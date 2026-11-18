// The disk tier of --serve's parked conversations (--conversation-disk-dir): the RAM cache's evictions
// (ConversationCache::set_on_evict) are written to session files (conversation_file.hpp, one per layer-split stage)
// by a background thread and read back when a request continues them, within a byte limit (and, for the one-file
// format of a non-persistent run, a time limit).
// By default only this process reads its files back, and the folder's files of this tier are removed at start and at
// the end.  `persist`: the files are bound to the session files' identity (model and settings) instead, a small index
// file beside each conversation lists its tokens, and the next start picks them up (no time limit, the budget only).
// (xrip, upstream PR #1937 + b0c4d361 / 65f3fd00.)
//
// strata-nvfp4 additions (extras/dialog-park): an engine killed without QUIT (the tray stops it with taskkill /F /T)
// never reaches the QUIT-time save, so the engine also writes on its own while it is idle (write_now: on the
// caller's thread, given up within one 16 MiB block when a request arrives) - the live conversation and every parked
// one that has no copy on the disk yet.  An entry a request reads back may stay on the disk (take(..., keep)): it is
// still a valid resume point if the engine dies before its next save, and a newer copy of the same conversation
// removes it once that copy is written (supersede, the RAM cache's #342 rule).  The index carries a hash.  The
// background writer starts a job only while the engine is idle (set_busy) unless RAM is waited for.
#pragma once

#include "strata/core/conversation_cache.hpp"
#include "strata/core/conversation_chunks.hpp"
#include "strata/core/conversation_file.hpp"

#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <system_error>
#include <thread>
#include <vector>

namespace strata::core {

/// --conversation-disk-live (chunked format only): whether the live state of a conversation is written along with its
/// checkpoints.  The live state is the deepest point the conversation reached (its K/V and a running state of ~118 MB);
/// a checkpoint within a few thousand tokens of it resumes the same conversation after re-reading only that gap.
///   always: always written (the default, as before);
///   skip:   not written when a checkpoint lies within `live_gap` tokens of it - the image is written "as of" the
///           deepest checkpoint (its K/V cut to that point, no live running state);
///   auto:   skip, but only for background writes (the RAM cache's evictions, the idle save) - never for the writes of
///           RAM pressure and of the QUIT-time save, which keep every token they can.
enum class LivePolicy { always, skip, automatic };
/// Pure policy: true when a write with `policy` should drop the live state, given the live state's tokens and the
/// deepest checkpoint's (0 = none).  No checkpoint, or none behind the live state: false.  `auto` is `skip` here: the
/// caller (the tier) passes `always` for a write that must not skip.
bool conversation_disk_skip_live(LivePolicy policy, int64_t live_tokens, int64_t deepest_checkpoint_tokens, int64_t gap);
/// "always", "skip" or "auto".  False + `error` otherwise.
bool conversation_disk_parse_live(const std::string& text, LivePolicy& policy, std::string& error);

/// --conversation-disk-mib's default: the one storage limit of the disk tier, in MiB (10 GB; v1 and chunked alike).
constexpr int64_t kConversationDiskDefaultMib = 10240;

struct ConversationDiskOptions {
    std::string dir;                 ///< empty: the tier is off
    /// THE storage limit: at most this many bytes of files on the disk - the v1 files, the manifests and the whole chunk
    /// pool, unreferenced chunks included; 0: off.  Over it, the OLDEST data goes first (least recently used: an
    /// entry's last write, hit or read; a chunk's last release, hit or read): first the unreferenced chunks, then the
    /// oldest entries (their chunks become unreferenced and go in the same order).  Removing an entry that is still
    /// valid logs a "WARNING [disk-limit]" (rate limited) and counts in ConversationDiskStats::forced_evictions: the
    /// limit is too small.  --serve's default is 10240 MiB.
    uint64_t budget_bytes = 0;
    /// A file older than this is removed (<= 0: kept until the limit evicts it).  v1 (one file per conversation) only:
    /// the chunked format has no time limit and ignores it.
    int64_t ttl_s = 30 * 60;
    uint64_t min_free_bytes = 0;     ///< a write needs the file plus this much free on the disk (session preflight)
    bool persist = false;            ///< keep the files for later runs, bound to `identity`
    SessionFileIdentity identity;    ///< persist: the session files' identity (model and settings fingerprints)
    /// (strata-nvfp4) several engines use the folder at once - the replicas of one server, one per GPU - as one pool:
    /// rescan() takes in what the others wrote, and open() leaves alone the files of other identities and the young
    /// leftovers another engine may still be writing.  Needs `persist` (the files are bound to the identity).
    bool shared = false;
    /// (strata-nvfp4) the chunked, deduplicated format v2 (conversation_chunks.hpp): a manifest per stage plus
    /// content-addressed chunks in <dir>/strata-pool-<identity>/ (non-persist: strata-pool-n<nonce>).  The reader takes
    /// v1 and v2 files alike whatever this says; it picks what is WRITTEN.  Falls back to v1 on Windows.
    bool chunked = false;
    uint32_t chunk_tokens = 512;     ///< chunked: tokens of K/V per slab (a multiple of 4, 64..4096)
    ChunkCodec codec = ChunkCodec::raw;   ///< chunked: how chunks are stored (zstd falls back to raw when not built in)
    int codec_level = 1;             ///< chunked: the zstd level
    int io_workers = 2;              ///< chunked: threads of the chunk pipeline (hash, codec, write; parallel decode on read)
    LivePolicy live_policy = LivePolicy::always;   ///< chunked: see LivePolicy
    int64_t live_gap = 4096;         ///< chunked, skip/auto: the live state is dropped when a checkpoint is at most this many tokens behind it
    /// Called with every diagnostic line (no trailing newline), from the caller's or the writer's thread.
    std::function<void(const std::string&)> log;
    /// Tests: called on the writer thread before each write starts.
    std::function<void()> before_write;
    /// Tests (chunked): called on the writing thread after a manifest's chunks are written or hit, before it is published.
    std::function<void()> before_manifest;
};

/// What the tier did since it started (diagnostics).
struct ConversationDiskStats {
    size_t queued = 0, written = 0, write_failed = 0, ram_hits = 0, disk_hits = 0, read_failed = 0;
    size_t expired = 0, evicted = 0, waits = 0;   ///< evicted = forced_evictions (entries removed over the limit)
    size_t synced = 0, sync_cancelled = 0, superseded = 0, kept = 0;   ///< write_now; supersede; take(keep)
    uint64_t bytes_written = 0, bytes_read = 0;
    uint64_t tokens_back = 0;                 ///< prompt tokens given back (not read again) by ram or disk hits
    double write_ms = 0, read_ms = 0, wait_ms = 0;
    // chunked (format v2): chunks looked up by the writes, and what they cost
    uint64_t chunks_hit = 0, chunks_new = 0;
    uint64_t raw_bytes = 0, dedup_bytes = 0, new_file_bytes = 0;   ///< K/V + state bytes seen; of them already stored; new chunk file bytes
    uint64_t chunks_swept = 0, swept_bytes = 0;                    ///< unreferenced chunks evicted for the limit (oldest first)
    /// the storage limit was too small: entries still valid removed to fit it (and the bytes they held), WARNING lines
    /// logged (rate limited: one a minute), and images refused because one alone is over the limit
    uint64_t forced_evictions = 0, forced_evicted_bytes = 0, limit_warnings = 0, limit_rejected = 0;
    bool direct_io = false;                                        ///< the last chunk write used O_DIRECT
    uint64_t prefix_reads = 0, prefix_raw_bytes = 0;               ///< takes that read a prefix of the entry; the bytes they read
    uint64_t live_skipped = 0, live_skipped_tokens = 0;            ///< writes made "as of" a checkpoint; the tokens they left out
};

/// --conversation-disk-codec: "raw" or "zstd" / "zstd:LEVEL" (1..19; plain "zstd" is level 1).  False + `error` otherwise.
/// A zstd asked for in a build without it is accepted (the tier then stores raw and says so in its start line).
bool conversation_disk_parse_codec(const std::string& text, ChunkCodec& codec, int& level, std::string& error);

class ConversationDiskTier {
public:
    using Clock = std::chrono::steady_clock;

    explicit ConversationDiskTier(ConversationDiskOptions options, std::function<Clock::time_point()> now = Clock::now);
    ~ConversationDiskTier();
    ConversationDiskTier(const ConversationDiskTier&) = delete;
    ConversationDiskTier& operator=(const ConversationDiskTier&) = delete;

    bool enabled() const { return open_; }
    bool persistent() const { return open_ && options_.persist; }
    bool shared() const { return open_ && options_.shared; }
    /// Creates the folder (0700 on POSIX) and starts the writer.  This tier's files a previous run left there are
    /// removed, or with `persist` taken in when their identity matches (the others removed).  False (and the tier
    /// stays off) when the folder cannot be used.
    bool open(std::string& error);

    /// Queues `image` for the writer; its RAM is freed when its write ends, stored or not.  Every checkpoint is written
    /// with it (they hold no K/V), so a prompt that leaves the conversation early still resumes from one.  False when
    /// the tier is off or over budget.  `flush` (RAM pressure): the file is flushed before its RAM is counted as back -
    /// unflushed, Windows keeps the written bytes as modified file cache, which the available figure leaves out.
    bool put(SavedConversation&& image, bool flush = false);

    struct Match {
        uint64_t id = 0;        ///< the entry (stable while the writer runs; 0 = none)
        int64_t tokens = 0;
        bool live = false;
    };
    /// The longest prefix of `prompt` an entry holds (its live state or its checkpoint), queued or on the disk, as
    /// ConversationCache::best; ties prefer the newest entry.
    template<class Token>
    Match best(const std::vector<Token>& prompt, const std::vector<ConversationImageKey>& images, bool cvec) const {
        std::lock_guard<std::mutex> lock(m_);
        Match best;
        auto scan = [&](const Entry& e) {
            if (e.cvec != cvec) return;
            auto consider = [&](const ConversationCheckpoint& c, bool live) {
                const int64_t n = conversation_prefix(c, prompt, images);
                if (n > best.tokens || (n == best.tokens && n > 0 && e.id > best.id)) best = {e.id, n, live};
            };
            consider(e.live, true);
            for (const auto& c : e.checkpoints) consider(c, false);
        };
        for (const auto& e : entries_) scan(e);
        for (const auto& j : queue_) scan(j.entry);
        if (writing_) scan(writing_entry_);   // its file appears when the write ends: take() waits for it
        return best;
    }

    /// Hands entry `id` back in `out` (its stage images included) and removes it from the tier either way: from RAM
    /// when it is still queued, else from the disk (waiting for its write when it is being written).  `admit(need,
    /// why)` is asked before each file's payload is allocated (session_file_read's RAM preflight).  `tokens` is what
    /// the request reuses (diagnostics).  On failure `out` is empty and `error` says why.
    /// `keep`: an entry on the disk stays there (its files are read, not removed) and becomes the newest - a request
    /// that borrows a sibling's prefix, or (persist) a resume point that must outlive a crash until a newer copy of
    /// the conversation is written.  A queued entry still leaves the queue (its RAM image is handed over).
    ///
    /// `upto` (chunked entries, with `keep`): the request reuses a checkpoint of exactly `upto` tokens, shorter than the
    /// entry's live state: only the K/V and the running states up to that point are read (chunked_read's prefix image:
    /// the checkpoint becomes the live state, the checkpoints of at most `upto` tokens stay).  The image `out` is then
    /// `upto` tokens long and the engine restores it as any other.  Ignored (the whole entry is read) for a v1 entry, an
    /// entry still queued, without `keep`, or when no checkpoint of that length is held.
    bool take(uint64_t id, int64_t tokens, SavedConversation& out, std::string& error,
              const std::function<bool(uint64_t need_bytes, std::string& why)>& admit = {}, bool keep = false,
              int64_t upto = 0);

    /// Writes `image` NOW on the caller's thread (no queue), then registers it, makes room for the budget and removes
    /// the older entries it supersedes - in that order, so a crash at any point leaves at least one copy.  `sources`
    /// (optional): the first stage's K/V streamed from the device instead of `image.kv` (session_file_write's
    /// streaming form; `image.kv` must then be empty).  `cancel` is asked before every 16 MiB block: true gives the
    /// write up (`*cancelled`, temporary files removed, nothing registered).  An entry with exactly these keys already
    /// on the disk is not written again: its id comes back.  `id` = the entry.  False with `error` on a failure.
    /// `background`: a write that can give way (the idle save), not the QUIT-time save (LivePolicy::automatic).
    /// `as_of_out`: the tokens of the checkpoint the entry was written as of (live-skip), 0 when it holds the whole live state.
    bool write_now(const SavedConversation& image, const std::vector<SessionKvSource>* sources,
                   const std::function<bool()>& cancel, uint64_t& id, std::string& error, bool* cancelled = nullptr,
                   bool background = false, int64_t* as_of_out = nullptr);
    /// Whether entry `id` is on the disk (or queued / being written).
    bool has(uint64_t id) const;
    /// `shared`: the entries other engines published since the last look come in (their index files), and the ones
    /// another engine removed (index gone) leave this list.  Cheap (one folder listing); returns how many came in.
    size_t rescan();
    /// The engine runs a request (true) or is idle (false): the background writer starts a queued job only while
    /// idle, or when someone waits for it (wait_idle, take).
    void set_busy(bool busy);

    /// Whether images wait in the queue or are being written (their RAM is not back yet).
    bool pending() const;
    /// Blocks until the queue is empty and no write runs; returns the ms it waited.
    double wait_idle();
    /// #879: every conversation goes - queued, being written (waited for) and on the disk, files too.  Returns how many.
    size_t clear();

    /// Removes the entries older than the time limit (v1 only: the chunked format has none); returns how many.
    size_t expire();

    size_t size() const;                    ///< entries on the disk
    uint64_t bytes() const;                 ///< their files' bytes (chunked: manifests + the chunk pool, unreferenced chunks included)
    size_t pool_chunks() const;             ///< chunked: chunk files the pool knows (tests); 0 otherwise
    ConversationDiskStats stats() const;
    /// One line of totals for the log.
    std::string summary() const;
    /// The file name prefix of this tier (tests).
    static const char* file_prefix() { return "strata-conv-"; }

private:
    struct Entry {
        uint64_t id = 0;
        std::vector<std::string> files;          ///< the first stage's image, then one per later stage
        ConversationCheckpoint live;             ///< ids and images only: what best() matches
        std::vector<ConversationCheckpoint> checkpoints;
        bool cvec = true;
        uint64_t bytes = 0;                      ///< the files' bytes (queued: the image's estimate)
        Clock::time_point at;
        bool chunked = false;                    ///< format v2: `files` are manifests, `refs` the chunks they name
        std::vector<ChunkRef> refs;              ///< every chunk of every manifest, duplicates included (each ref()'d once)
        uint64_t manifest_bytes = 0;
        int64_t idx_ns = 0;                      ///< persist: the index file's mtime at scan (ns): the LRU order a restart rebuilds
        uint64_t written_bytes = 0;              ///< bytes this write put on the disk (chunked: new chunks + manifests)
        std::string chunk_note;                  ///< the write's chunk counters, for its log line
        std::string index;                       ///< persist: the index file
        int64_t written_s = 0;                   ///< persist: when it was written (seconds since the epoch)
    };
    struct Job {
        Entry entry;
        SavedConversation image;
        bool flush = false;                      ///< put(flush): written durably
    };
    void writer();
    void write(Job& job);                                    // without the lock
    // the files of `img` (one per stage) under entry `e`'s id; `sources`: stage 0's K/V streamed; without the lock
    bool write_files(Entry& e, const SavedConversation& img, const std::vector<SessionKvSource>* sources,
                     const std::function<bool()>& cancel, std::string& error, bool flush = false, int64_t as_of = 0);
    // live-skip: the tokens of the checkpoint `img` is written "as of" (0: written whole); a background write can skip
    // under `automatic`, the others only under `skip`
    int64_t live_skip_tokens(const SavedConversation& img, bool background) const;
    // a written entry joins the list: the budget and the entries it supersedes go after it (m_ held); log part back
    std::string publish_locked(Entry&& e, size_t& made_room, size_t& dropped);
    void remove_locked(size_t index, bool delete_files);     // m_ held
    size_t expire_locked();                                  // m_ held
    size_t find_locked(uint64_t id) const;                   // m_ held; entries_.size() when absent
    void log(const std::string& line) const;
    SessionFileIdentity file_identity() const;
    bool write_index(Entry& e, std::string& error) const;
    bool read_index(const std::string& path, Entry& e, bool* transient = nullptr) const;
    std::string scan_folder(size_t& stale, std::error_code& ec);   // open(): take in or remove; its log part
    // chunked (format v2); all with m_ held unless noted
    bool bind_chunks(Entry& e) const;                // a scanned entry's manifests: refs taken, chunks checked; false = refuse it
    void release_refs(Entry& e) const;               // unref + forget
    uint64_t pool_bytes_all() const;                 // pool.bytes() or 0
    // The limit loop: total = the files' bytes + the manifests + the whole pool, ghosts included.  Over the limit: the
    // oldest unreferenced chunks go (ChunkPool::evict_ghosts), then the oldest entry (LRU; a WARNING each time), and so on.
    void make_room_locked(uint64_t protect_a, uint64_t protect_b, size_t& evicted);
    // (m_ held) one rate-limited "WARNING [disk-limit]" line (queued; flush_warnings() logs it with m_ released)
    enum class LimitEvent { evicted, rejected, remaining, undecided };
    void warn_limit_locked(LimitEvent what, int64_t tokens, uint64_t entry_bytes, uint64_t demand_bytes);
    void flush_warnings();                           // without m_
    uint64_t footprint(const Entry& e) const;        // the bytes removing `e` can free at most: manifests + its chunks
    void touch_index(const Entry& e) const;          // persist: the index file's mtime = now (the LRU order a restart rebuilds)
    // shared: the entries other engines publish or use move in the list by their index files' mtime
    void sort_by_use_locked();
    // shared: every chunk any manifest of this identity names (this engine's entries from memory, the others' parsed
    // from their manifests, cached per index file); false: it could not be told
    bool named_chunks_locked(std::set<ChunkKey>& out);
    void remove_foreign_pools() const;               // open(): other runs' (non-persist) or identities' pools
    std::string pool_dir() const;
    std::unique_ptr<ChunkPool> pool_;
    std::map<std::string, std::vector<ChunkKey>> named_cache_;   ///< shared: index path -> the chunks its manifests name
    std::map<std::string, int> named_fail_;          ///< shared: index path -> consecutive passes its manifest did not parse
    std::vector<std::string> warn_lines_;            ///< queued by warn_limit_locked
    bool limit_warned_ = false;                      ///< a WARNING was logged ...
    Clock::time_point limit_warn_at_{};              ///< ... at this time (now_()): at most one a minute
    size_t limit_warn_pending_ = 0;                  ///< events since the last line
    std::set<std::string> seen_index_;   ///< shared: index files looked at already (taken in, or another identity's)

    ConversationDiskOptions options_;
    std::function<Clock::time_point()> now_;
    bool open_ = false;
    uint64_t nonce_ = 0, next_id_ = 0, bytes_ = 0;
    mutable std::mutex m_;
    std::condition_variable cv_;
    std::deque<Entry> entries_;   ///< on the disk, oldest first
    std::deque<Job> queue_;       ///< waiting for the writer, oldest first
    bool writing_ = false;        ///< the writer holds a job (writing_entry_ is its keys)
    Entry writing_entry_;
    bool busy_ = false;           ///< set_busy: the engine runs a request; the writer starts no job then
    int urgent_ = 0;              ///< callers waiting for the writer (wait_idle, take): it starts a job even when busy
    uint64_t reading_ = 0;        ///< take(keep) reads this entry's files: the budget and supersede leave it alone
    bool stop_ = false;
    std::thread thread_;
    ConversationDiskStats stats_;
};

}  // namespace strata::core
