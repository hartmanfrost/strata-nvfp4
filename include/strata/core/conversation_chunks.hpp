// Chunked, deduplicated disk format of the parked conversations (format v2; the disk tier only, see
// docs/NVFP4.md "Chunked parked dialogs").  A parked image is a small MANIFEST file (strata-conv-*.bin, magic
// "STRSESS\x02") plus content-addressed CHUNKS in a pool folder shared by every conversation of one identity:
//   <dir>/<xx>/<key32hex>.chk      xx = the key's first byte; 4096-byte header block, payload at 4096, size rounded
//                                  up to 4096 (O_DIRECT-aligned); key = SessionHasher::digest128 of the RAW bytes
// A zstd chunk's payload is a container of per-part segments (conversation_codec.hpp: raw or zstd, byte-shuffled where
// the element width says so); raw is the default.
// A K/V slab (T tokens of every K/V layer) and a 4 MiB slice of a checkpoint's running state are chunks; an image that
// continues a conversation re-uses every chunk whose bytes did not change.  Chunks are immutable and published by a
// no-replace rename, so writers need no lock; whether a chunk is still wanted is the tier's reference count (a view
// of the folder, rebuilt from the manifests).  There is no time limit: an unreferenced chunk (a "ghost") stays as a
// dedup cache until the tier's storage limit needs the room, and then the OLDEST ghost (least recently used: its last
// release, hit or read) goes first; a referenced or pinned chunk is never deleted by the pool.  Several engines on one
// folder (`shared`) announce the chunks a write or a take is about to use in a lease file (ChunkLease), and a deleter
// never removes a chunk a lease or a manifest names.
// Every read checks the chunk's size, header hash, the lengths the manifest expects and the hash of the raw bytes
// against the key.  Little-endian only.  Pure host code: no CUDA.  POSIX only (the pool refuses to open elsewhere).
#pragma once

#include "strata/core/conversation_codec.hpp"
#include "strata/core/conversation_file.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <set>
#include <string>
#include <vector>

namespace strata::core {

constexpr size_t kChunkAlign = 4096;
constexpr uint64_t kMaxChunkRaw = uint64_t(64) << 20;
constexpr uint64_t kMaxChunkFile = kChunkAlign + kMaxChunkRaw;   // 4096 + round_up(64 MiB, 4096)
constexpr uint32_t kStateSlice = 4u << 20;                        // bytes of a checkpoint's running state per chunk

struct ChunkKey {
    std::array<uint8_t, 16> b{};
    bool operator==(const ChunkKey& o) const { return b == o.b; }
    bool operator<(const ChunkKey& o) const { return b < o.b; }
    std::string hex() const;
};
struct ChunkRef {
    ChunkKey key;
    uint32_t raw_len = 0, stored_len = 0;
    bool operator==(const ChunkRef& o) const {
        return key == o.key && raw_len == o.raw_len && stored_len == o.stored_len;
    }
};
enum class ChunkKind : uint16_t { kv_slab = 0, state_slice = 1 };

// The file size of a chunk whose payload is `stored_len` bytes: 4096 + round_up(stored_len, 4096).
uint64_t chunk_file_size(uint32_t stored_len);
// The chunk's key: SessionHasher(0).digest128() over the raw bytes, one pass.
ChunkKey chunk_key(const void* raw, size_t n);

// An aligned staging buffer for one chunk file: the raw bytes are produced at payload() (offset 4096), so a raw chunk
// goes to the disk without another copy.
class ChunkBuffer {
public:
    ChunkBuffer();
    ~ChunkBuffer();
    ChunkBuffer(ChunkBuffer&&) noexcept;
    ChunkBuffer& operator=(ChunkBuffer&&) noexcept;
    ChunkBuffer(const ChunkBuffer&) = delete;
    ChunkBuffer& operator=(const ChunkBuffer&) = delete;
    // room for a payload of `raw_cap` bytes (the file is padded to 4096); false when out of memory
    bool reserve(size_t raw_cap);
    uint8_t* payload() { return data_ + kChunkAlign; }
    size_t capacity() const { return cap_; }
private:
    friend class ChunkPool;
    uint8_t* data_ = nullptr;
    size_t cap_ = 0, alloc_ = 0;
};

struct ChunkPoolOptions {
    std::string dir;                    // .../strata-pool-<ident16>
    ChunkCodec codec = ChunkCodec::raw;
    int zstd_level = 1;
    bool direct = true;                 // O_DIRECT; false or STRATA_SESSION_BUFFERED=1: buffered + sync_file_range + FADV_DONTNEED
    uint64_t min_free_bytes = 0;        // a new chunk needs its file plus this much free (free-space preflight)
    bool shared = false;                // other engines use the folder: leases (ChunkLease) and the census (ChunkPool::census)
    // tests: return an errno value to make a step fail, 0 to go on: "chunk_write" (EINVAL while direct: the buffered
    // fallback), "chunk_sync" (buffered mode: the writeback step), "chunk_flush" (durable fdatasync), "chunk_rename",
    // "shard_sync"
    std::function<int(const char* step)> fault;
};

// Shared folders: the chunks one write (or one non-keep take) is about to hit, write or read.  An engine that deletes
// ghosts reads every lease before it reads the manifests, and never deletes a chunk a lease names.  The file is created
// under a temporary name, flock()ed, and renamed to ".lease-<rand>": a deleter never sees an unlocked half-made lease,
// and a lease whose flock can be taken belongs to a dead process (ignored and removed).  Destroying the object (or
// release()) removes the file.  Thread-safe.
class ChunkLease {
public:
    ~ChunkLease();
    ChunkLease(const ChunkLease&) = delete;
    ChunkLease& operator=(const ChunkLease&) = delete;
    // Appends the keys (a key already announced is skipped).  Must come BEFORE the chunk is hit, written or read.
    void announce(const ChunkKey&);
    void announce(const ChunkRef*, size_t n);
    void release();
private:
    friend class ChunkPool;
    ChunkLease() = default;
    struct Impl;
    std::unique_ptr<Impl> p_;
};

struct ChunkPutInfo {
    bool hit = false;                   // the chunk was already there (nothing written)
    uint32_t stored_len = 0;
    ChunkCodec codec = ChunkCodec::raw;
    bool shuffled = false;
    uint64_t new_file_bytes = 0;        // file bytes this call wrote
    double encode_ms = 0;               // time in the codec
    bool direct_io = false;             // the write went through O_DIRECT
    SessionError error_kind = SessionError::none;   // after a failure
};

class ChunkPool {
public:
    explicit ChunkPool(ChunkPoolOptions);
    ~ChunkPool();
    ChunkPool(const ChunkPool&) = delete;
    ChunkPool& operator=(const ChunkPool&) = delete;
    // Creates the folder (0700) and indexes the chunks already there (unreferenced: they are ghosts, ordered by their
    // files' mtime - nanoseconds - until the owner refs them or the limit needs the room); removes the leftover
    // temporary files (crash hygiene, not retention) older than 10 minutes.
    bool open(std::string& error);
    // Thread-safe.  The chunk is referenced (+1) for the caller on return; a writer that fails unref()s its whole list.
    // `buf` holds the raw bytes at payload().
    // `segments` (optional): how the raw bytes split for the codec (lengths add up to raw_len), else one segment of
    // `shuffle_width`.
    bool put_buffer(ChunkKind, ChunkBuffer& buf, size_t raw_len, uint8_t shuffle_width, ChunkRef& out,
                    ChunkPutInfo& info, std::string& error, bool durable = false,
                    const std::vector<CodecSegment>* segments = nullptr, ChunkLease* lease = nullptr);
    // The same from raw bytes elsewhere (one copy into a staging buffer).
    bool put(ChunkKind, const void* raw, size_t raw_len, uint8_t shuffle_width, ChunkRef& out, ChunkPutInfo& info,
             std::string& error, bool durable = false);
    // Thread-safe.  `dst` has room for ref.raw_len.  Verifies the size formula, the header hash, raw_len / stored_len /
    // key against `ref`, decodes with exactly raw_len, and the hash of the raw bytes against the key.
    bool get(ChunkKind, const ChunkRef&, void* dst, size_t dst_cap, std::string& error, SessionError* kind = nullptr);
    bool exists(const ChunkRef&) const;                     // stat + the size formula, no data read
    // References are this tier's view of the folder; a pin is one more ref.
    void ref(const ChunkRef*, size_t n);
    void unref(const ChunkRef*, size_t n);                  // at 0 the chunk becomes the NEWEST ghost
    // Shared: a lease for one write or take (null when the pool is not shared).
    std::unique_ptr<ChunkLease> begin_lease();
    // Deletes ghosts, OLDEST first (by last release / hit / read), until at least `want_bytes` of file bytes are freed
    // (UINT64_MAX: all of them).  Never a referenced or pinned chunk (those are not ghosts).  Not shared: unlinked.
    // Shared: other engines' references are invisible, so `named` is asked for the keys every manifest in the folder
    // names (false: it could not tell - nothing is deleted); a victim named by a manifest or a lease is kept, the
    // others are renamed away, the snapshot is taken again, and the ones it names now are put back (a writer or a
    // taker raced this deleter); the rest are unlinked.  Returns the chunks removed.  Each round reads the leases, the
    // manifests, then the leases AGAIN: a take announces its chunks before it removes its index, so a lease that
    // appears between the first lease read and the manifest read is caught by the second.  `undecided` (optional) is
    // set when `named` could not tell (shared): nothing was deleted for that reason, and the caller should not conclude
    // that the room cannot be made.
    size_t evict_ghosts(uint64_t want_bytes, const std::function<bool(std::set<ChunkKey>&)>& named = {},
                        uint64_t* freed_bytes = nullptr, bool* undecided = nullptr);
    // Shared: takes the chunk files other engines wrote into the index as ghosts (ordered by mtime) and forgets the
    // unreferenced ones that are gone.  Cheap (a folder listing, an lstat per new file).  No-op when not shared.
    void census();
    // durable writes: fsync the shard folders touched since the last call
    bool sync_touched_shards(std::string& error);
    uint64_t bytes() const;                                 // file bytes of every known chunk, ghosts included
    uint64_t live_bytes() const;                            // the same for the referenced or pinned chunks only
    uint64_t ghost_bytes() const;                           // the same for the unreferenced ones (the next to go)
    size_t ghost_count() const;
    // tests: the ghosts' keys, oldest first
    std::vector<ChunkKey> ghosts_oldest_first() const;
    uint32_t refs_of(const ChunkKey&) const;                // tests and diagnostics (0 for an unknown chunk)
    size_t count() const;
    void remove_all();                                      // the whole folder: non-persist end, clear()
    std::string path_of(const ChunkKey&) const;
    const ChunkPoolOptions& options() const;
private:
    struct Impl;
    std::unique_ptr<Impl> p_;
};

struct ChunkedWriteOptions {
    uint32_t chunk_tokens = 512;        // multiple of 4, 64..4096
    bool durable = false;
    uint64_t min_free_bytes = 0;        // (the pool's own option decides the per-chunk preflight; reserved)
    int workers = 1;                    // 1: everything on the calling thread; N: N threads hash, encode and write (1..8)
    bool low_priority = true;           // the worker threads run at nice +5 (Linux)
    ChunkLease* lease = nullptr;        // shared: every chunk is announced in it before it is looked at or written
    std::function<bool()> cancel;       // asked before every chunk (with workers, from the worker threads too, never two calls at once): true gives the write up (error "cancelled")
    SessionProgress progress;           // (raw bytes done, raw bytes of the image) after every chunk
    SessionPhase phase;                 // "flush" and "publish" before each starts
    // asked after the last chunk and before the manifest is published (false + why: the write is given up); the
    // shared tier re-checks there that every chunk it hit is still on the disk
    std::function<bool(const std::vector<ChunkRef>&, std::string& why)> before_publish;
    std::function<int(const char* step)> fault;   // the manifest's steps as SessionWriteOptions::fault: "write", "file_flush", "rename", "dir_flush"
    // > 0: write the image "as of" the checkpoint of exactly this many tokens (live-skip): that checkpoint becomes the
    // live state, the deeper checkpoints and the live state's own running state are left out, and the K/V is cut to
    // those tokens (the source is read from offset 0 as before, only less of it).  The result is the image the prefix
    // read builds for the same length.  An error when no checkpoint of that length leads the live state.
    int64_t as_of_tokens = 0;
};
struct ChunkedWriteStats {
    uint64_t manifest_bytes = 0, raw_bytes = 0, hit_raw_bytes = 0, new_file_bytes = 0;
    uint32_t chunks = 0, hits = 0, misses = 0;
    double hash_ms = 0, encode_ms = 0, io_ms = 0;
    bool direct_io = false;
};
// Writes the chunks (through `pool`) and the manifest at `manifest_path` (temp + rename, like session_file_write).
// `image`: everything in RAM, or the meta (image.kv empty) with `sources` as session_file_write(sources).  `refs_out`:
// every chunk the manifest names, in manifest order, duplicates included (each ref()'d once on return; on failure
// all are unref()'d and the chunks only this write created stay as ghosts).  Layer-split state is refused.
bool chunked_write(ChunkPool&, const std::string& manifest_path, const SavedConversation& image,
                   const std::vector<SessionKvSource>* sources, const SessionFileIdentity&,
                   const ChunkedWriteOptions&, std::vector<ChunkRef>& refs_out, ChunkedWriteStats&, std::string& error,
                   SessionStatus* = nullptr);

struct ChunkedReadOptions {
    // > 0 (and below the live state's length): the prefix image of that many tokens - the checkpoint of exactly that
    // length becomes the live state, the checkpoints of at most that length stay (the one at `upto` too: the engine
    // mounts it), the K/V is cut to those tokens (cells up to the page that holds the last one, pooled rows
    // upto/4 + 1 with the spare row from the checkpoint's `dead`), and only the slabs and state slices that image needs
    // are read.  restorable by conversation_snapshot_restore unchanged.  An error when no checkpoint has that length;
    // upto == the live length is the whole image.
    int64_t upto = 0;
    int workers = 1;
};
// Full validation (manifest header, identity, hashes, bounded parse, every chunk) before `out` is replaced.
// limits.admit is asked once, with chunked_read_peak_bytes, before the image is allocated.
bool chunked_read(ChunkPool&, const std::string& manifest_path, const SessionFileIdentity&, SavedConversation& out,
                  size_t& manifest_bytes, uint64_t& raw_bytes_read, std::string& error, const SessionReadLimits&,
                  const ChunkedReadOptions& = {}, SessionStatus* = nullptr);
// The manifest's header and chunk references only (the tier's scan and rescan).
bool chunked_manifest_refs(const std::string& manifest_path, const SessionFileIdentity&, std::vector<ChunkRef>& refs,
                           uint64_t& manifest_bytes, std::string& error, bool* transient = nullptr);
// transient (optional): set on a failure that says nothing about the file itself (an I/O or resource error: EMFILE,
// ENOMEM, EIO) as against a file that was read and is invalid.
bool is_chunked_manifest(const std::string& path);          // the first 8 bytes are "STRSESS\x02"
// What the RAM preflight is asked for: the parsed image, three chunk buffers per worker (file, raw, decode temporary),
// the small vectors.  `max_chunk_raw`: the largest chunk of the manifest when it is parsed (0: a guess from chunk_tokens).
uint64_t chunked_read_peak_bytes(uint64_t raw_total, int workers, uint32_t chunk_tokens, uint64_t max_chunk_raw = 0);

} // namespace strata::core
