// src/program/conv_key_test.cpp - the conversation key rule (include/strata/program/conv_key.hpp): which requests may
// cancel the park of the outgoing session.  CPU only, header only.
//
// The truth table, in the serve loop's terms.  `live` is the session's tokens, `ids` the new prompt:
//
//   rule off (STRATA_CONV_KEY_MIN_SHARED unset / 0 / junk) ....... never: parks as before
//   no key on the request ........................................ never
//   the live session has no key .................................. never (an unkeyed request, a restored file)
//   another key (subagent, fork) with a long shared prefix ........ never: the parent's image stays parkable
//   same key, fewer than N shared tokens (title request) .......... never
//   same key, N or more shared tokens, the prompt diverges ........ supersede (compaction, prune)
//   same key, prompt is a strict prefix of the session (rewind) ... supersede
//   same key, exactly N shared / N - 1 shared ..................... supersede / not
//   same key, exact continuation (from_live) ...................... never: nothing is parked, nothing is dropped
//   batch slot / admit slot / pictures / other control vector /
//   nothing to park / park already skipped ........................ never
//
// then the replay of a measured window (seven parks of one compaction-heavy agent session, from its engine log): P1..P7
// drop, the title request and the subagent park as before.
//
// Then the serve loop's side (the Tracker generate.cpp calls):
//   request_of: every loop value the rule reads, one at a time;
//   the log lines, whole, with the numbers serve/test_monitor.py feeds its reader, and the counter;
//   a model of the serve loop, request after request, in which the live key is never set by hand - only the
//   Tracker's taken()/reset() at the places the loop calls them (main, title, subagent, an unkeyed client,
//   compaction, a session file, #879, ckpt=0, a cancelled request, the --batch paths);
//   the borrow after a request that did not park: the K/V retained from an earlier restore must go before the
//   donor's prefix overwrites the session, or the next park takes the old conversation's bytes (ConversationCache);
//   and the glue in generate.cpp itself (source checks: where the four calls sit relative to the park sites, that
//   live_ok = false makes park_current_body return, that a borrow drops the retained K/V before the restore, and a
//   tripwire on the places that make a session live).  What stays for a GPU run: the engine executing that glue.
#include "strata/program/conv_key.hpp"
#include "strata/core/conversation_cache.hpp"

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

using namespace strata::program::conv_key;

namespace {
int g_fail = 0;
int g_checks = 0;
void check(bool ok, const char* what) {
    ++g_checks;
    std::printf("  %-78s %s\n", what, ok ? "ok" : "FAIL");
    if (!ok) ++g_fail;
}

// a token stream: `shared` tokens that are `base`'s, then `extra` distinct tokens (tagged so two calls never agree past `shared`)
std::vector<int32_t> stream(int64_t shared, int64_t extra, int32_t tag, int32_t base = 1) {
    std::vector<int32_t> v;
    for (int64_t i = 0; i < shared; ++i) v.push_back(base + (int32_t) (i % 5000));
    for (int64_t i = 0; i < extra; ++i) v.push_back(100000 + tag * 1000 + (int32_t) (i % 900));
    return v;
}
std::vector<int64_t> wide(const std::vector<int32_t>& v) { return std::vector<int64_t>(v.begin(), v.end()); }

constexpr uint64_t KEY = 0x5ee5a10cca7e0040ull, SUB = 0x1111222233334444ull;

Request base_request() {
    Request r;
    r.req_key = KEY;
    r.live_key = KEY;
    r.parkable = true;
    return r;
}

Verdict run(int64_t min, const Request& r, const std::vector<int32_t>& live, const std::vector<int64_t>& ids,
            int64_t* shared = nullptr) {
    int64_t s = -1;
    const Verdict v = decide(min, r, live, ids, s);
    if (shared) *shared = s;
    return v;
}

template<class A, class B>
bool starts_with_seq(const std::vector<A>& seq, const std::vector<B>& prefix) {
    if (prefix.size() > seq.size()) return false;
    for (size_t i = 0; i < prefix.size(); ++i)
        if ((int64_t) seq[i] != (int64_t) prefix[i]) return false;
    return true;
}

// A model of the serve loop around the rule (generate.cpp), at the grain the rule sees: the main session (`live`,
// `live_ok`), whether a park site parks, and the Tracker's calls exactly where the loop makes them - before_park before
// the park sites, taken() where the request takes the session over, reset() for a session file and the #879 flush.
// The live key is never set here: it is whatever those calls leave in the Tracker.
struct SimLoop {
    Tracker rule;
    std::vector<int32_t> live;
    bool live_ok = false;
    int64_t min_tokens = 0;   // --conversation-cache-min-tokens
    int parks = 0;
    std::vector<std::string> lines;
    explicit SimLoop(int64_t min) : rule(min) {}

    struct Gen {
        std::vector<int32_t> prompt;
        uint64_t key = 0;
        int32_t reply_tag = 0;   // the reply's tokens (distinct per request)
        bool ckpt = true;        // ckpt=0: the request leaves nothing to continue or park (#830)
        bool cancel = false;     // #1620: stopped while reading, the session stays live at the prompt read so far
        int admit_slot = -1;     // BGEN: read in the main session, then copied into the slot
        int slot_source = -1;    // the request starts from an idle slot's conversation
        bool images = false;
    };

    Verdict request(const Gen& g) {
        LoopState s;
        s.live_ok = live_ok;
        s.live_empty = live.empty();
        s.cache_enabled = true;
        s.cache_wants = min_tokens <= 0 || (int64_t) live.size() >= min_tokens;
        s.from_live = live_ok && starts_with_seq(g.prompt, live);
        s.slot_source = g.slot_source;
        s.admit_slot = g.admit_slot;
        s.req_images = g.images;
        const Outcome o = rule.before_park(g.key, s, live, wide(g.prompt));
        if (!o.line.empty()) lines.push_back(o.line);
        if (o.drop) live_ok = false;
        // the park sites: (!from_live || slot_source >= 0) -> park_current, which parks a live_ok session that is long
        // enough to be wanted (park_current_body)
        if ((!s.from_live || g.slot_source >= 0) && live_ok && !live.empty() && s.cache_wants) ++parks;
        live_ok = false;   // "until this request has finished, the session is in between"
        rule.taken(g.key);
        live = g.prompt;
        if (!g.cancel) {
            for (int32_t i = 0; i < 50; ++i) live.push_back(900000 + g.reply_tag * 100 + i);
            live_ok = g.ckpt;
        } else {
            live_ok = true;   // #1620
        }
        return o.verdict;
    }
    void restore_file(std::vector<int32_t> ids) {   // a session file replaces the session
        live = std::move(ids);
        live_ok = true;
        rule.reset();
    }
    void nan_flush() {   // #879: the next request drops every cached state first
        live_ok = false;
        rule.reset();
    }
};

// The loop's calls around a borrow, on the real ConversationCache: the retained tokens the NEXT park takes for the
// prefix the new session shares with the conversation the K/V was retained from.
int64_t retained_at_next_park(bool parked_before_borrow, bool drop_before_borrow) {
    using strata::core::ConversationCache;
    using strata::core::ConversationKv;
    auto kv = [](size_t n) {
        std::vector<ConversationKv> v(2);
        for (auto& l : v) l.k.resize(n, 1);
        return v;
    };
    ConversationCache cache(64u << 20, 4);
    const auto A = stream(1500, 500, 1);                 // restored earlier: its K/V retained for the next park
    cache.retain(kv(4096), (int64_t) A.size(), {}, A);
    if (parked_before_borrow) (void) cache.take_reuse_for(A);   // the outgoing session's park took it
    const int64_t P = 1500;                              // the donor's checkpoint: its K/V overwrites the cells [0, P)
    if (drop_before_borrow) cache.take_reuse();          // generate.cpp: just before conversation_snapshot_restore_prefix
    cache.limit_reuse(P);                                // read_from = resume = P
    const auto live = stream(1500, 300, 1);              // the session read on: shares 1800 tokens with A
    return cache.take_reuse_for(live).unchanged_tokens;
}

size_t count_of(const std::string& s, const std::string& what, size_t from = 0, size_t to = std::string::npos) {
    size_t n = 0;
    for (size_t p = s.find(what, from); p != std::string::npos && p < to; p = s.find(what, p + what.size())) ++n;
    return n;
}

// the text before `pos`, back to the previous non-blank character's line, trimmed
std::string line_before(const std::string& s, size_t pos) {
    size_t e = pos;
    while (e > 0 && (s[e - 1] == ' ' || s[e - 1] == '\n' || s[e - 1] == '\t')) --e;
    size_t b = s.rfind('\n', e == 0 ? 0 : e - 1);
    b = b == std::string::npos ? 0 : b + 1;
    while (b < e && s[b] == ' ') ++b;
    return s.substr(b, e - b);
}
}  // namespace

int main() {
    std::printf("min_shared (STRATA_CONV_KEY_MIN_SHARED):\n");
    check(min_shared(nullptr) == 0, "unset: off");
    check(min_shared("") == 0, "empty: off");
    check(min_shared("0") == 0, "0: off");
    check(min_shared("1024") == 1024, "1024: on");
    check(min_shared("  1024 ") == 1024, "blanks around the number are fine");
    check(min_shared("1") == 1, "1: on (shares at least the first token)");
    check(min_shared("-5") == 0, "negative: off");
    check(min_shared("1024abc") == 0, "trailing junk: off, a malformed opt-in does not switch the rule on");
    check(min_shared("abc") == 0, "not a number: off");
    check(min_shared("1e3") == 0, "float spelling: off");
    check(min_shared("99999999999999999999") == 0, "out of range: off");

    std::printf("parse_key (conv=<hex> of the GEN line):\n");
    check(parse_key("5ee5a10cca7e0040") == 0x5ee5a10cca7e0040ull, "16 hex digits");
    check(parse_key("2A") == 0x2a, "upper case and a short value");
    check(parse_key("0000000000000001") == 1, "leading zeros");
    check(parse_key("0") == 0 && parse_key("0000000000000000") == 0, "an explicit 0 is no key");
    check(parse_key("") == 0 && parse_key(nullptr) == 0, "empty / null: no key");
    check(parse_key("0x2a") == 0, "a 0x prefix is not hex digits only: no key");
    check(parse_key("+2a") == 0 && parse_key("-2a") == 0, "a sign: no key");
    check(parse_key("2a ") == 0 && parse_key("2g") == 0, "a blank or a non-hex character: no key");
    check(parse_key("00000000000000001") == 0, "17 digits: no key");

    std::printf("common_prefix (int32 session against an int64 prompt):\n");
    {
        const auto a = stream(10, 5, 1), b = stream(10, 5, 2);
        check(common_prefix(a, wide(b)) == 10, "diverges after the shared 10");
        check(common_prefix(a, wide(a)) == (int64_t) a.size(), "equal sequences share everything");
        check(common_prefix(std::vector<int32_t>{}, wide(a)) == 0, "an empty session shares nothing");
        std::vector<int32_t> neg = {-1, 2};
        std::vector<int64_t> negw = {-1, 2};
        check(common_prefix(neg, negw) == 2, "a negative id is the same id in both widths");
        const auto aw = wide(a);
        check(common_prefix(a, std::vector<int64_t>(aw.begin(), aw.begin() + 4)) == 4, "the shorter one bounds it");
    }

    std::printf("the truth table (min shared = 1024):\n");
    const int64_t N = 1024;
    {
        const auto live = stream(9373, 55000, 1);
        const auto self_rewrite = wide(stream(9373 + 48000, 40000, 2));   // shares 57373 with live, then its own tail
        const auto base = [&] { return base_request(); };
        int64_t sh = 0;

        check(run(N, base(), live, self_rewrite, &sh) == Verdict::supersede && sh >= N, "same key, self-rewrite: supersede");
        check(run(0, base(), live, self_rewrite) == Verdict::off, "rule off (0): off, parks as before");
        {
            Request r = base(); r.req_key = 0;
            check(run(N, r, live, self_rewrite) == Verdict::no_key, "no key on the request: no_key");
        }
        {
            Request r = base(); r.live_key = 0;
            check(run(N, r, live, self_rewrite) == Verdict::no_live_key, "the live session has no key: no_live_key");
        }
        {
            Request r = base(); r.req_key = SUB;
            check(run(N, r, live, self_rewrite) == Verdict::other_key, "another key with a long shared prefix (a fork): other_key");
        }
        {
            const auto title = wide(stream(80, 300, 3));   // its own system prompt: shares 80 tokens
            check(run(N, base(), live, title, &sh) == Verdict::short_prefix && sh == 80,
                  "same key, only 80 shared (a title request): short_prefix");
        }
        {
            const auto root_only = wide(stream(9373, 20000, 4));
            check(run(N, base(), stream(9373, 100, 1), root_only, &sh) == Verdict::supersede && sh == 9373,
                  "same key, 9373 shared (the system prompt): supersede (P4 / P7 of the window)");
        }
        {   // exact continuation: the prompt starts with the whole session, the loop sets continues_live
            Request r = base(); r.continues_live = true;
            auto cont = live; cont.insert(cont.end(), {7, 8, 9});
            check(run(N, r, live, wide(cont)) == Verdict::not_applicable, "same key, exact continuation (from_live): not_applicable");
            check(common_prefix(live, wide(cont)) == (int64_t) live.size(),
                  "  ...and the whole session is the common prefix (so only continues_live protects it)");
        }
        {   // a rewind: the prompt is a strict prefix of the session (the client re-sent an earlier turn)
            const auto rewind = wide(std::vector<int32_t>(live.begin(), live.begin() + 30000));
            check(run(N, base(), live, rewind, &sh) == Verdict::supersede && sh == 30000, "same key, prompt a strict prefix of the session: supersede");
        }
        {   // the threshold is inclusive
            const auto exact = wide(stream(N, 5000, 5));
            const auto below = wide(stream(N - 1, 5000, 6));
            const auto L = stream(N + 500, 100, 1);   // both share only the first N / N - 1 tokens of L
            check(run(N, base(), L, exact, &sh) == Verdict::supersede && sh == N, "exactly min shared: supersede");
            check(run(N, base(), L, below, &sh) == Verdict::short_prefix && sh == N - 1, "min shared - 1: short_prefix");
        }
        {   // min shared 1: a single shared token is enough
            check(run(1, base(), stream(1, 10, 1), wide(stream(1, 10, 2)), &sh) == Verdict::supersede && sh == 1, "min 1: one shared token supersedes");
            check(run(1, base(), stream(0, 10, 1), wide(stream(0, 10, 2)), &sh) == Verdict::short_prefix && sh == 0, "min 1: none shared does not");
        }
        {
            Request r = base(); r.other_start = true;
            check(run(N, r, live, self_rewrite) == Verdict::not_applicable, "a batch slot start: not_applicable");
        }
        {
            Request r = base(); r.images = true;
            check(run(N, r, live, self_rewrite) == Verdict::not_applicable, "pictures: not_applicable");
        }
        {
            Request r = base(); r.same_cvec = false;
            check(run(N, r, live, self_rewrite) == Verdict::not_applicable, "another control vector: not_applicable");
        }
        {
            Request r = base(); r.parkable = false;   // cache off, session incomplete, too short, pin=N or the rewind rule skip it
            check(run(N, r, live, self_rewrite) == Verdict::not_applicable, "nothing to park (or already skipped): not_applicable");
        }
        check(run(N, base(), std::vector<int32_t>{}, self_rewrite) == Verdict::not_applicable, "an empty session: not_applicable");
        {   // the shared length is only measured when the key matched (no linear compare for every request)
            Request r = base(); r.req_key = SUB;
            run(N, r, live, self_rewrite, &sh);
            check(sh == 0, "another key: the prefix is not measured");
        }
        {
            Request r = base(); r.req_key = 0;
            run(N, r, live, self_rewrite, &sh);
            check(sh == 0, "no key: the prefix is not measured");
        }
        check(logged(Verdict::supersede) && logged(Verdict::short_prefix) && logged(Verdict::other_key) &&
              logged(Verdict::no_live_key), "the key involved: logged");
        check(!logged(Verdict::off) && !logged(Verdict::no_key) && !logged(Verdict::not_applicable),
              "off / no key / not applicable: silent");
        check(std::string(reason(Verdict::supersede)).size() > 0 && std::string(reason(Verdict::other_key)).size() > 0,
              "every logged verdict has words");
    }

    std::printf("the measured window (P1..P7, then a title request and a subagent):\n");
    {
        // the session after each request, as the loop sees it.  KEY is the agent's session id; the title request carries
        // it too but its prompt has its own system prompt; the subagent has its own key.
        const int64_t root = 9373;
        struct Step { const char* name; std::vector<int32_t> live; std::vector<int64_t> ids; uint64_t key, live_key; Verdict want; };
        const auto main_turn = stream(root, 58000, 1);                         // main, live after a turn
        std::vector<Step> steps;
        steps.push_back({"P1 rewind to the turn checkpoint (66774 of 67289)", stream(root + 57900, 600, 1),
                         wide(stream(root + 57900 - 0, 0, 1)), KEY, KEY, Verdict::supersede});
        steps.push_back({"P2 prune rewrote the middle, the prompt restarts at the root", stream(root, 77000, 2),
                         wide(stream(root, 60000, 3)), KEY, KEY, Verdict::supersede});
        steps.push_back({"P3 summarizer request: main superseded at ~58K", stream(root + 48000, 38000, 4),
                         wide(stream(root + 48000, 400, 5)), KEY, KEY, Verdict::supersede});
        steps.push_back({"P4 post-compaction main: the summarizer's state, root only", stream(root, 55000, 6),
                         wide(stream(root, 3000, 7)), KEY, KEY, Verdict::supersede});
        steps.push_back({"P5 rewind (the donor of the window: lost, the price of the rule)", stream(root + 43000, 600, 8),
                         wide(stream(root + 43000, 0, 8)), KEY, KEY, Verdict::supersede});
        steps.push_back({"P6 second summarizer request", stream(root + 48000, 30000, 9),
                         wide(stream(root + 48000, 300, 10)), KEY, KEY, Verdict::supersede});
        steps.push_back({"P7 post-compaction main again, root only", stream(root, 50000, 11),
                         wide(stream(root, 2000, 12)), KEY, KEY, Verdict::supersede});
        steps.push_back({"title request after main: its own system prompt, ~80 shared", main_turn,
                         wide(stream(80, 400, 13)), KEY, KEY, Verdict::short_prefix});
        steps.push_back({"subagent (own child key) after main: parent's prefix, parks as before", main_turn,
                         wide(stream(root + 20000, 800, 14)), SUB, KEY, Verdict::other_key});
        steps.push_back({"main after the subagent: the live session carries the subagent's key", stream(root + 20000, 900, 14),
                         wide(stream(root + 20000, 5000, 15)), KEY, SUB, Verdict::other_key});
        steps.push_back({"an unkeyed client interleaved (curl): no key, parks as before", main_turn,
                         wide(stream(root, 500, 16)), 0, KEY, Verdict::no_key});
        steps.push_back({"main right after the unkeyed one: the live session has no key", stream(root, 600, 16),
                         wide(stream(root + 57000, 100, 1)), KEY, 0, Verdict::no_live_key});
        int fired = 0;
        for (const Step& s : steps) {
            Request r = base_request();
            r.req_key = s.key;
            r.live_key = s.live_key;
            int64_t sh = 0;
            const Verdict v = decide(1024, r, s.live, s.ids, sh);
            check(v == s.want, s.name);
            fired += v == Verdict::supersede;
        }
        check(fired == 7, "7 of the 7 parks of the window are dropped");
    }

    std::printf("request_of (the loop's values, one at a time):\n");
    {
        LoopState all;   // everything that lets a park happen, nothing that hands it to another path
        all.live_ok = true;
        all.live_empty = false;
        all.cache_enabled = true;
        all.cache_wants = true;
        const Request r0 = request_of(KEY, SUB, all);
        check(r0.req_key == KEY && r0.live_key == SUB, "the two keys go through as they are");
        check(r0.parkable && !r0.continues_live && !r0.other_start && !r0.images && r0.same_cvec,
              "a complete, wanted session with no other path: parkable, nothing else set");
        auto with = [&](auto set) { LoopState s = all; set(s); return request_of(KEY, KEY, s); };
        check(!with([](LoopState& s) { s.live_ok = false; }).parkable, "live_ok false: not parkable");
        check(!with([](LoopState& s) { s.live_empty = true; }).parkable, "an empty session: not parkable");
        check(!with([](LoopState& s) { s.cache_enabled = false; }).parkable, "the cache off: not parkable");
        check(!with([](LoopState& s) { s.cache_wants = false; }).parkable, "below --conversation-cache-min-tokens: not parkable");
        check(!with([](LoopState& s) { s.pin_sibling = true; }).parkable, "pin=N already skips the park: not parkable");
        check(!with([](LoopState& s) { s.own_rewind = true; }).parkable, "STRATA_PARK_REWIND_MAX already skips it: not parkable");
        check(!with([](LoopState& s) { s.reply_replaced = true; }).parkable, "a replaced reply already skips it: not parkable");
        check(!with([](LoopState& s) { s.park_worth = false; }).parkable,
              "--conversation-cache-park-gain already skips it: not parkable");
        check(with([](LoopState& s) { s.from_live = true; }).continues_live, "from_live: continues the session");
        check(with([](LoopState& s) { s.slot_source = 0; }).other_start, "a slot source (slot 0 counts): another path");
        check(with([](LoopState& s) { s.admit_slot = 0; }).other_start, "admitted into a slot (slot 0 counts): another path");
        check(!with([](LoopState& s) { s.slot_source = -1; s.admit_slot = -1; }).other_start, "no slot: not another path");
        check(with([](LoopState& s) { s.req_images = true; }).images, "pictures in the prompt");
        check(with([](LoopState& s) { s.live_images = true; }).images, "pictures in the session");
        check(!with([](LoopState& s) { s.want_cvec = false; }).same_cvec, "another control vector asked for");
        check(!with([](LoopState& s) { s.cvec_cached = false; }).same_cvec, "the session read with another control vector");
        check(with([](LoopState& s) { s.want_cvec = false; s.cvec_cached = false; }).same_cvec, "the same (off) control vector");
    }

    std::printf("Tracker: the log lines (whole, as serve/test_monitor.py reads them) and the counter:\n");
    {
        Tracker off(min_shared(nullptr));
        check(!off.on() && off.start_line(true).empty(), "rule off: no start line (the log reads as before)");
        LoopState s;
        s.live_ok = true;
        s.live_empty = false;
        s.cache_enabled = true;
        s.cache_wants = true;
        const auto live = stream(9373 + 48000, 87209 - 9373 - 48000, 1);
        const auto rewrite = wide(stream(9373 + 48000, 400, 2));
        off.taken(KEY);
        const Outcome o0 = off.before_park(KEY, s, live, rewrite);
        check(o0.verdict == Verdict::off && !o0.drop && o0.line.empty() && off.dropped() == 0,
              "rule off: verdict off, no drop, no line, no count");

        Tracker t(min_shared("1024"));
        check(t.on() && t.min_shared() == 1024 && t.live_key() == 0 && t.dropped() == 0, "a new tracker: on, no live key");
        check(t.start_line(true) == "strata serve: conversation cache: conv key rule on (STRATA_CONV_KEY_MIN_SHARED=1024): a "
                                    "request with the live conversation's conv= key that shares that many tokens with it "
                                    "and does not continue it is not parked\n", "the start line");
        check(t.start_line(false).find("conv key rule inert (the conversation cache is off)") != std::string::npos,
              "the start line with the cache off says inert");
        t.taken(KEY);
        check(t.live_key() == KEY, "taken(): the live key is the request's");
        const Outcome o1 = t.before_park(KEY, s, live, rewrite);
        check(o1.verdict == Verdict::supersede && o1.drop && o1.shared == 57373 && t.dropped() == 1, "supersede: drop, counted");
        check(o1.line == "strata serve: conversation cache: dropped superseded conversation (same conv key, 57373 shared "
                         "tokens of its 87209) - not parked, not written to the disk tier; dropped_superseded=1\n",
              "the supersede line, whole");
        const Outcome o2 = t.before_park(KEY, s, live, rewrite);
        check(o2.line.find("; dropped_superseded=2\n") != std::string::npos && t.dropped() == 2, "the count is cumulative");
        const Outcome o3 = t.before_park(KEY, s, live, wide(stream(80, 300, 3)));
        check(o3.verdict == Verdict::short_prefix && !o3.drop && t.dropped() == 2, "a title request: no drop, not counted");
        check(o3.line == "strata serve: conversation cache: conv key does not supersede the outgoing conversation (too few "
                         "tokens shared: 80 shared tokens < 1024); it parks as before\n", "the short-prefix line, whole");
        const Outcome o4 = t.before_park(SUB, s, live, rewrite);
        check(o4.verdict == Verdict::other_key && !o4.drop &&
              o4.line == "strata serve: conversation cache: conv key does not supersede the outgoing conversation (another "
                         "conversation's key); it parks as before\n", "another key: its line, no drop");
        const Outcome o5 = t.before_park(0, s, live, rewrite);
        check(o5.verdict == Verdict::no_key && !o5.drop && o5.line.empty(), "no key: silent");
        LoopState cont = s;
        cont.from_live = true;
        const Outcome o6 = t.before_park(KEY, cont, live, rewrite);
        check(o6.verdict == Verdict::not_applicable && !o6.drop && o6.line.empty(), "a continuation: silent");
        t.reset();
        const Outcome o7 = t.before_park(KEY, s, live, rewrite);
        check(t.live_key() == 0 && o7.verdict == Verdict::no_live_key && !o7.drop &&
              o7.line.find("(the live conversation has no key); it parks as before") != std::string::npos,
              "reset(): no live key, parks as before");
        check(t.dropped() == 2, "only supersedes count");
        check(Tracker(-5).min_shared() == 0 && !Tracker(-5).on(), "a negative threshold is off");
    }

    std::printf("the serve loop, request after request (the live key only from taken() / reset()):\n");
    {
        SimLoop L(1024);
        const int64_t root = 9373;
        auto grow = [](std::vector<int32_t> v, int64_t extra, int32_t tag) {
            for (int64_t i = 0; i < extra; ++i) v.push_back(200000 + tag * 1000 + (int32_t) (i % 900));
            return v;
        };
        int want_parks = 0, want_drops = 0;
        auto step = [&](const char* what, const SimLoop::Gen& g, Verdict want, bool parks) {
            const int before = L.parks;
            const Verdict v = L.request(g);
            want_parks += parks;
            want_drops += v == Verdict::supersede;
            char buf[200];
            std::snprintf(buf, sizeof buf, "%s: %s%s", what, reason(v), L.parks - before == 1 ? ", parked" : "");
            check(v == want && L.parks - before == (parks ? 1 : 0) && L.rule.live_key() == g.key, buf);
        };
        SimLoop::Gen g;
        g = {}; g.prompt = stream(root, 2000, 1); g.key = KEY; g.reply_tag = 1;
        step("main, first turn (nothing live yet)", g, Verdict::not_applicable, false);
        g = {}; g.prompt = grow(L.live, 500, 2); g.key = KEY; g.reply_tag = 2;
        step("main, next turn (continues the session)", g, Verdict::not_applicable, false);
        const auto main2 = L.live;
        g = {}; g.prompt = stream(80, 300, 3); g.key = KEY; g.reply_tag = 3;
        step("title request (the session's key, 80 shared)", g, Verdict::short_prefix, true);
        g = {}; g.prompt = grow(main2, 600, 4); g.key = KEY; g.reply_tag = 4;
        step("main after the title (the title's state holds the key)", g, Verdict::short_prefix, true);
        const auto main3 = L.live;
        g = {}; g.prompt = stream(root, 3000, 5); g.key = SUB; g.reply_tag = 5;
        step("subagent (its own key) on the root", g, Verdict::other_key, true);
        g = {}; g.prompt = grow(main3, 400, 6); g.key = KEY; g.reply_tag = 6;
        step("main after the subagent: the live key is the SUBAGENT's now", g, Verdict::other_key, true);
        const auto main4 = L.live;
        g = {}; g.prompt = grow(std::vector<int32_t>(main4.begin(), main4.end() - 400), 395, 7); g.key = KEY; g.reply_tag = 7;
        step("compaction summarizer (main's prefix + the instruction)", g, Verdict::supersede, false);
        g = {}; g.prompt = stream(root, 4000, 8); g.key = KEY; g.reply_tag = 8;
        step("post-compaction main (the summarizer's state, root only)", g, Verdict::supersede, false);
        const auto post = L.live;
        g = {}; g.prompt = stream(root, 100, 9); g.key = 0; g.reply_tag = 9;
        step("an unkeyed client (curl) on the same system prompt", g, Verdict::no_key, true);
        g = {}; g.prompt = grow(post, 300, 10); g.key = KEY; g.reply_tag = 10;
        step("main after the unkeyed client: the live session has no key", g, Verdict::no_live_key, true);
        const auto main5 = L.live;
        g = {}; g.prompt = stream(80, 200, 11); g.key = KEY; g.reply_tag = 11; g.ckpt = false;
        step("a ckpt=0 title request: parks main, leaves nothing live", g, Verdict::short_prefix, true);
        g = {}; g.prompt = grow(main5, 200, 12); g.key = KEY; g.reply_tag = 12;
        step("main after the ckpt=0 request: nothing to park", g, Verdict::not_applicable, false);
        g = {}; g.prompt = grow(L.live, 300, 13); g.key = KEY; g.reply_tag = 13; g.cancel = true;
        step("main, cancelled while reading (#1620: stays live with its key)", g, Verdict::not_applicable, false);
        g = {}; g.prompt = stream(root, 1000, 14); g.key = KEY; g.reply_tag = 14;
        step("a rewrite after the cancelled request: its key was kept", g, Verdict::supersede, false);
        L.restore_file(stream(root, 5000, 15));
        check(L.rule.live_key() == 0, "a session file restored: no live key");
        g = {}; g.prompt = stream(root, 6000, 16); g.key = KEY; g.reply_tag = 16;
        step("main after the session file: parks it as before", g, Verdict::no_live_key, true);
        L.nan_flush();
        check(L.rule.live_key() == 0, "#879 flush: no live key");
        g = {}; g.prompt = stream(root, 6000, 16); g.key = KEY; g.reply_tag = 17;
        step("main after the #879 flush: nothing to park", g, Verdict::not_applicable, false);
        g = {}; g.prompt = stream(root, 700, 18); g.key = KEY; g.reply_tag = 18; g.admit_slot = 0;
        step("a BGEN request admitted into slot 0: its own path, parks as before", g, Verdict::not_applicable, true);
        g = {}; g.prompt = stream(root, 800, 19); g.key = KEY; g.reply_tag = 19;
        step("the same key rewriting after the BGEN request: it left its key", g, Verdict::supersede, false);
        g = {}; g.prompt = stream(root, 900, 20); g.key = KEY; g.reply_tag = 20; g.slot_source = 1;
        step("a request starting from an idle slot: its own path, parks as before", g, Verdict::not_applicable, true);
        g = {}; g.prompt = stream(root + 20000, 500, 21); g.key = SUB; g.reply_tag = 21;
        step("a subagent", g, Verdict::other_key, true);
        g = {}; g.prompt = stream(root + 20000, 100, 22); g.key = SUB; g.reply_tag = 22;
        step("the subagent rewriting itself (its own compaction)", g, Verdict::supersede, false);
        g = {}; g.prompt = stream(root, 900, 23); g.key = KEY; g.reply_tag = 23; g.images = true;
        step("a request with pictures: not touched, parks as before", g, Verdict::not_applicable, true);
        check(L.parks == want_parks, "parks: as counted step by step");
        check((int) L.rule.dropped() == want_drops && want_drops == 5, "5 supersedes, the Tracker's counter agrees");
        size_t logged_lines = 0, drop_lines = 0;
        for (const auto& l : L.lines) {
            ++logged_lines;
            drop_lines += l.find("dropped superseded conversation") != std::string::npos;
        }
        check(drop_lines == 5 && logged_lines == 5 + 8, "a line per supersede and per keyed park (8), none for the rest");
        SimLoop off(min_shared("0"));
        g = {}; g.prompt = stream(root, 2000, 1); g.key = KEY; g.reply_tag = 1;
        off.request(g);
        g = {}; g.prompt = stream(root, 900, 2); g.key = KEY; g.reply_tag = 2;
        check(off.request(g) == Verdict::off && off.parks == 1 && off.lines.empty(), "rule off: a same-key rewrite parks, no line");
    }

    std::printf("a borrow after a request that did not park (retained K/V, ConversationCache):\n");
    check(retained_at_next_park(true, false) == 0, "today's path: the outgoing park took the retained K/V");
    check(retained_at_next_park(false, false) == 1500,
          "no park, no drop: the next park would take 1500 tokens of the OLD conversation's bytes (the hazard)");
    check(retained_at_next_park(false, true) == 0, "no park, dropped before the borrow: nothing stale is taken");
    check(retained_at_next_park(true, true) == 0, "the drop is a no-op when the park already took it");

    std::printf("the glue in generate.cpp (source checks):\n");
#ifdef STRATA_SOURCE_DIR
    {
        std::ifstream f(std::string(STRATA_SOURCE_DIR) + "/src/program/generate.cpp", std::ios::binary);
        std::stringstream buf;
        buf << f.rdbuf();
        const std::string src = buf.str();
        if (src.empty()) {
            std::printf("  (skipped: no src/program/generate.cpp under %s)\n", STRATA_SOURCE_DIR);
        } else {
            const size_t npos = std::string::npos;
            check(count_of(src, "conv_rule.") == 5 && count_of(src, "conv_rule.start_line(") == 1 &&
                  count_of(src, "conv_rule.before_park(") == 1 && count_of(src, "conv_rule.taken(req_conv);") == 1 &&
                  count_of(src, "conv_rule.reset();") == 2, "the Tracker's calls: start_line, before_park, taken, reset x2");
            check(src.find("else if (key == \"conv\") req_conv = strata::program::conv_key::parse_key(") != npos,
                  "conv= is parsed into req_conv");
            const size_t rewind = src.find("bool own_rewind = false;");
            const size_t dec = src.find("conv_rule.before_park(req_conv, ks, live, ids);");
            const size_t taken = src.find("conv_rule.taken(req_conv);");
            check(rewind != npos && dec != npos && taken != npos && rewind < dec && dec < taken,
                  "before_park after pin=N / the rewind rule are decided, before the request takes the session");
            const size_t ks0 = src.rfind("strata::program::conv_key::LoopState ks;", dec);
            const char* fields[] = {"ks.live_ok = live_ok;", "ks.live_empty = live.empty();",
                                    "ks.cache_enabled = conversations.enabled();",
                                    "ks.cache_wants = conversations.wants((int64_t) live.size());",
                                    "ks.pin_sibling = pin_sibling;", "ks.own_rewind = own_rewind;",
                                    "ks.reply_replaced = reply_replaced;", "ks.park_worth = park_worth;", "ks.from_live = from_live;",
                                    "ks.slot_source = slot_source;", "ks.admit_slot = admit_slot;",
                                    "ks.req_images = !req_imgs.empty();", "ks.live_images = !live_imgs.empty();",
                                    "ks.want_cvec = want_cvec;", "ks.cvec_cached = cvec_cached;"};
            bool fields_ok = ks0 != npos && ks0 > rewind && count_of(src, "ks.", ks0, dec) == 15;
            for (const char* fld : fields) fields_ok = fields_ok && count_of(src, fld, ks0, dec) == 1;
            check(fields_ok, "every LoopState field, from its own loop variable, once");
            const size_t line_at = src.find("if (!kd.line.empty()) std::fputs(kd.line.c_str(), stderr);", dec);
            const size_t drop_at = src.find("if (kd.drop) live_ok = false;", dec);
            check(line_at != npos && drop_at != npos && drop_at < dec + 300 && line_at < drop_at,
                  "the line is written and a supersede sets live_ok = false right after the decision");
            check(count_of(src, "park_current(", rewind, dec) == 0 && count_of(src, "park_current(", drop_at, taken) == 3,
                  "all three park sites come after it (none between the rewind rule and it)");
            check(count_of(src, "live_ok = true", dec, taken) == 0 && count_of(src, "continue;", dec, taken) == 0,
                  "nothing makes the session live again, and no request skips the hand-over, before it is taken");
            check(line_before(src, taken) ==
                      "live_ok = false;   // until this request has finished, the session is in between",
                  "taken() right where the request takes the session over");
            const size_t body = src.find("auto park_current_body = [&](size_t held) -> bool {");
            size_t first = body == npos ? npos : body + std::string("auto park_current_body = [&](size_t held) -> bool {").size();
            while (first != npos && first < src.size() && (src[first] == ' ' || src[first] == '\n')) ++first;
            const std::string early = "if (!conversations.enabled() || !live_ok || live.empty()) return true;";
            check(first != npos && src.compare(first, early.size(), early) == 0,
                  "live_ok = false makes park_current_body return before it parks anything");
            bool resets_ok = true;
            for (size_t p = src.find("conv_rule.reset();"); p != npos; p = src.find("conv_rule.reset();", p + 1))
                resets_ok = resets_ok && line_before(src, p).rfind("live_ok = ", 0) == 0;
            const size_t file_at = src.find("live = std::move(image.live.ids);");
            const size_t flush_at = src.find("const size_t n_checks = checks.size();");
            const size_t r1 = src.find("conv_rule.reset();");
            const size_t r2 = r1 == npos ? npos : src.find("conv_rule.reset();", r1 + 1);
            check(resets_ok && file_at != npos && r1 > file_at && r1 < file_at + 600 && flush_at != npos && r2 > flush_at &&
                  r2 < flush_at + 200, "reset() where a session file is restored and in the #879 flush, beside live_ok");
            // the tripwire: a new place that makes a session live must say whose conversation it is
            bool live_sites_ok = count_of(src, "live_ok = true;") == 2 &&
                                 count_of(src, "live_ok = o.prompt_cache > 0 && req_ckpt;") == 1;
            for (size_t p = src.find("live_ok = true;"); p != npos; p = src.find("live_ok = true;", p + 1))
                live_sites_ok = live_sites_ok && (p > taken || src.compare(src.find_first_not_of(" \n", p + 15), 18,
                                                                           "conv_rule.reset();") == 0);
            live_sites_ok = live_sites_ok && src.find("live_ok = o.prompt_cache > 0 && req_ckpt;") > taken;
            check(live_sites_ok, "the session becomes live in 3 places, each after taken() or with reset() (else: give "
                                 "the new place its key, and this count)");
            const size_t restore_prefix = src.find("conversation_snapshot_restore_prefix(");
            const size_t borrow_block = restore_prefix == npos ? npos : src.rfind("if (borrow) {", restore_prefix);
            const size_t drop_reuse = restore_prefix == npos ? npos : src.rfind("conversations.take_reuse();", restore_prefix);
            check(restore_prefix != npos && borrow_block != npos && drop_reuse != npos && drop_reuse > borrow_block &&
                  borrow_block > src.find("park_current(0);", dec),
                  "a borrow drops the retained K/V before the donor's prefix overwrites the session (after the park)");
        }
    }
#else
    std::printf("  (skipped: built without STRATA_SOURCE_DIR)\n");
#endif

    if (g_fail) std::printf("FAIL (%d of %d checks)\n", g_fail, g_checks);
    else std::printf("PASS (%d checks)\n", g_checks);
    return g_fail ? 1 : 0;
}
