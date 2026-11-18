// include/strata/program/conv_key.hpp - the conversation key: a client's hint about WHICH conversation a request
// belongs to, and the one rule it feeds in the serve loop (generate.cpp): do not park a conversation that the same
// client conversation has just rewritten itself.
//
// Why.  The engine has no conversation id: identity is the exact token prefix.  A client that rewrites its own history
// (a compaction, a prune of old tool results, a rewind to an earlier turn) sends a prompt that shares only a short
// prefix with the live session, so the serve loop parks the live session (K/V copied to the host, evicted on to the
// disk tier) as if another conversation were interleaved with it.  Nothing will ever continue the old branch - only
// the client knows - and each such park costs 250-630 ms, up to 1.5 GB of RAM and, once evicted, a disk write.
//
// The hint.  `conv=<hex>` between the sampling keys of a GEN line (serve/server.py derives it from the OpenAI
// `prompt_cache_key`, hashed, so no client text reaches the protocol; 0 = no key - and only for a client that opts in
// with the X-Strata-Conv-Scope header, because many OpenAI clients send one static key per user or per app).
// Requests that carry the same key are one conversation, in order.  The engine keeps the key of the request that left
// the live session (`live_key`, kept by Tracker below).
//
// The rule (opt-in: STRATA_CONV_KEY_MIN_SHARED=N, N > 0; unset or 0 = the rule is off and nothing changes).  A
// request that carries the key of the live session, does not continue it, and shares at least N tokens with it, has
// rewritten it: the live session is NOT parked and so never written to the disk tier.  Everything else behaves as
// before - no key, another key (a subagent, a fork: they keep their parent's parked image), the same key but fewer
// than N shared tokens (a title request with its own system prompt), a request that continues the live session.
//
// The invariant: a hint can only CANCEL a park (and with it a write).  It never picks or restores a state - a restore
// still needs the exact token prefix, as ever - so a wrong hint can cost speed (a conversation read again) and can
// never change an output.
#pragma once

#include <algorithm>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace strata::program::conv_key {

/// STRATA_CONV_KEY_MIN_SHARED: the shared-token threshold, 0 = the rule is off.  Unset, empty, negative, not a number
/// or out of range all mean off: a malformed opt-in must not switch a cache-dropping rule on.
inline int64_t min_shared(const char* text) {
    if (text == nullptr) return 0;
    while (*text == ' ' || *text == '\t') ++text;
    if (*text == '\0') return 0;
    errno = 0;
    char* end = nullptr;
    const long long v = std::strtoll(text, &end, 10);
    if (errno != 0 || end == text) return 0;
    while (*end == ' ' || *end == '\t' || *end == '\n' || *end == '\r') ++end;
    if (*end != '\0') return 0;   // "1024abc"
    return v > 0 ? (int64_t) v : 0;
}

/// The `conv=` value of a GEN line: 1 to 16 hex digits and nothing else (no sign, no 0x).  0 for anything else, and for
/// an explicit 0: no key.
inline uint64_t parse_key(const char* hex) {
    if (hex == nullptr) return 0;
    uint64_t v = 0;
    int n = 0;
    for (; *hex != '\0'; ++hex, ++n) {
        const char c = *hex;
        uint64_t d = 0;
        if (c >= '0' && c <= '9') d = (uint64_t) (c - '0');
        else if (c >= 'a' && c <= 'f') d = (uint64_t) (c - 'a' + 10);
        else if (c >= 'A' && c <= 'F') d = (uint64_t) (c - 'A' + 10);
        else return 0;
        if (n >= 16) return 0;
        v = (v << 4) | d;
    }
    return n == 0 ? 0 : v;
}

/// How many leading tokens the two sequences share.  The engine holds the live session as int32 and the prompt as
/// int64, so this is a template; ids are compared as the wide type they both fit.
template<class A, class B>
int64_t common_prefix(const std::vector<A>& a, const std::vector<B>& b) {
    const size_t n = std::min(a.size(), b.size());
    size_t i = 0;
    while (i < n && (int64_t) a[i] == (int64_t) b[i]) ++i;
    return (int64_t) i;
}

/// What the serve loop knows when it is about to park the outgoing session for a new request.
struct Request {
    uint64_t req_key = 0;          ///< `conv=` of this request (0 = none)
    uint64_t live_key = 0;         ///< the key of the request that left the live session (0 = none: an unkeyed request, a restored file)
    bool parkable = false;         ///< a park would happen: the cache is on, the session is complete, long enough to be wanted, and none of pin=N, the rewind rule, a replaced reply or --conversation-cache-park-gain already skips it
    bool continues_live = false;   ///< the prompt starts with the live session: no switch, nothing is parked
    bool other_start = false;      ///< the request starts from a batch slot or is admitted into one: that path parks by its own rules
    bool images = false;           ///< pictures in the prompt or the live session: not touched
    bool same_cvec = true;         ///< the control vector is as the live session had it (else the old state is not a rewrite of this one)
};

enum class Verdict {
    off,             ///< STRATA_CONV_KEY_MIN_SHARED is 0: the rule does not exist
    no_key,          ///< the request carries no key
    not_applicable,  ///< no park would happen, or one of the paths above (see Request)
    no_live_key,     ///< the live session was left by an unkeyed request: it may be anyone's, parks as before
    other_key,       ///< another client conversation (a subagent, a fork): parks as before
    short_prefix,    ///< the same key but fewer than min_shared tokens shared (a title request, a new chat): parks as before
    supersede,       ///< the same key and at least min_shared tokens shared: the client rewrote it, not parked
};

/// The rule.  `shared` is set to the common prefix when it was measured (short_prefix, supersede), else 0.  Pure:
/// Tracker::before_park adds the live key, the log line and the counter; the serve loop's own part is `live_ok = false`
/// for a supersede.
template<class A, class B>
Verdict decide(int64_t min_shared_tokens, const Request& r, const std::vector<A>& live, const std::vector<B>& ids,
               int64_t& shared) {
    shared = 0;
    if (min_shared_tokens <= 0) return Verdict::off;
    if (r.req_key == 0) return Verdict::no_key;
    if (!r.parkable || r.continues_live || r.other_start || r.images || !r.same_cvec || live.empty())
        return Verdict::not_applicable;
    if (r.live_key == 0) return Verdict::no_live_key;
    if (r.live_key != r.req_key) return Verdict::other_key;
    shared = common_prefix(live, ids);   // linear, under a millisecond at 100K tokens
    return shared >= min_shared_tokens ? Verdict::supersede : Verdict::short_prefix;
}

/// Whether the verdict is worth a line in the log: every outcome where a park was in play and the key was involved.
inline bool logged(Verdict v) {
    return v == Verdict::no_live_key || v == Verdict::other_key || v == Verdict::short_prefix || v == Verdict::supersede;
}

/// The words of a verdict that leaves the park alone, for the log ("... (another conversation's key); parks as before").
inline const char* reason(Verdict v) {
    switch (v) {
        case Verdict::no_live_key: return "the live conversation has no key";
        case Verdict::other_key: return "another conversation's key";
        case Verdict::short_prefix: return "too few tokens shared";
        case Verdict::supersede: return "the same conversation rewrote it";
        case Verdict::not_applicable: return "not applicable";
        case Verdict::no_key: return "no key";
        case Verdict::off: return "off";
    }
    return "";
}

/// The serve loop's own values at the park decision, by their names in generate.cpp, copied one to one (no logic at
/// the call site: what they mean for the rule is request_of below, which the tests cover).
struct LoopState {
    bool live_ok = false;         ///< live_ok: the session is complete and may be continued or parked
    bool live_empty = true;       ///< live.empty()
    bool cache_enabled = false;   ///< conversations.enabled()
    bool cache_wants = false;     ///< conversations.wants(live.size()): long enough for --conversation-cache-min-tokens
    bool pin_sibling = false;     ///< pin=N: the park is already skipped
    bool own_rewind = false;      ///< STRATA_PARK_REWIND_MAX: the park is already skipped
    bool reply_replaced = false;  ///< the live tail is only the reply this request replaces: the park is already skipped
    bool park_worth = true;       ///< --conversation-cache-park-gain: false = the park is already skipped
    bool from_live = false;       ///< the prompt starts with the live session
    int slot_source = -1;         ///< a --batch slot the request starts from (-1: none)
    int admit_slot = -1;          ///< the --batch slot the request is admitted into (-1: none)
    bool req_images = false;      ///< !req_imgs.empty()
    bool live_images = false;     ///< !live_imgs.empty()
    bool want_cvec = true;        ///< the control vector state this request asks for
    bool cvec_cached = true;      ///< the control vector state the session was read with
};

/// The rule's view of the loop's values.
inline Request request_of(uint64_t req_key, uint64_t live_key, const LoopState& s) {
    Request r;
    r.req_key = req_key;
    r.live_key = live_key;
    r.parkable = s.live_ok && !s.live_empty && s.cache_enabled && s.cache_wants && !s.pin_sibling && !s.own_rewind &&
                 !s.reply_replaced && s.park_worth;
    r.continues_live = s.from_live;
    r.other_start = s.slot_source >= 0 || s.admit_slot >= 0;
    r.images = s.req_images || s.live_images;
    r.same_cvec = s.want_cvec == s.cvec_cached;
    return r;
}

/// What the serve loop does about one request.
struct Outcome {
    Verdict verdict = Verdict::off;
    int64_t shared = 0;   ///< the common prefix, when measured
    bool drop = false;    ///< supersede: the loop sets live_ok = false, so no park site parks (nor writes) the session
    std::string line;     ///< the engine log line ("\n"-terminated), empty when there is nothing to say
};

/// The rule's state in the serve loop: the key of the request that left the live session, and the counter of
/// conversations it did not park.  The loop's glue is four calls (generate.cpp):
///   before_park(req_key, state, live, ids)  once per request, before the park sites; `drop` -> live_ok = false
///   taken(req_key)                          where the request starts rewriting the session (live_ok = false, "in
///                                           between"): whatever the session holds next is this request's
///                                           conversation, finished, cancelled (#1620) or ckpt=0 alike
///   reset()                                 where the session is replaced by what no request key describes: a
///                                           session file restored, the #879 flush
///   start_line(enabled)                     once at start
class Tracker {
public:
    explicit Tracker(int64_t min_shared_tokens) : min_(min_shared_tokens > 0 ? min_shared_tokens : 0) {}

    int64_t min_shared() const { return min_; }
    bool on() const { return min_ > 0; }
    uint64_t live_key() const { return live_key_; }
    size_t dropped() const { return dropped_; }

    void taken(uint64_t req_key) { live_key_ = req_key; }
    void reset() { live_key_ = 0; }

    /// The start-up line, empty with the rule off (the log then reads as before).
    std::string start_line(bool cache_enabled) const {
        if (!on()) return {};
        char buf[320];
        std::snprintf(buf, sizeof buf, "strata serve: conversation cache: conv key rule %s (STRATA_CONV_KEY_MIN_SHARED=%lld): "
                                       "a request with the live conversation's conv= key that shares that many tokens with "
                                       "it and does not continue it is not parked\n",
                      cache_enabled ? "on" : "inert (the conversation cache is off)", (long long) min_);
        return buf;
    }

    /// The decision for one request, its log line, and the counter.
    template<class A, class B>
    Outcome before_park(uint64_t req_key, const LoopState& s, const std::vector<A>& live, const std::vector<B>& ids) {
        Outcome o;
        o.verdict = decide(min_, request_of(req_key, live_key_, s), live, ids, o.shared);
        char buf[320];
        if (o.verdict == Verdict::supersede) {
            o.drop = true;
            ++dropped_;
            std::snprintf(buf, sizeof buf, "strata serve: conversation cache: dropped superseded conversation (same conv key, "
                                           "%lld shared tokens of its %zu) - not parked, not written to the disk tier; "
                                           "dropped_superseded=%zu\n", (long long) o.shared, live.size(), dropped_);
            o.line = buf;
        } else if (logged(o.verdict)) {
            char detail[64] = "";
            if (o.verdict == Verdict::short_prefix)
                std::snprintf(detail, sizeof detail, ": %lld shared tokens < %lld", (long long) o.shared, (long long) min_);
            std::snprintf(buf, sizeof buf, "strata serve: conversation cache: conv key does not supersede the outgoing "
                                           "conversation (%s%s); it parks as before\n", reason(o.verdict), detail);
            o.line = buf;
        }
        return o;
    }

private:
    int64_t min_ = 0;
    uint64_t live_key_ = 0;
    size_t dropped_ = 0;
};

}  // namespace strata::program::conv_key
