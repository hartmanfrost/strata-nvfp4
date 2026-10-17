// Chunked, deduplicated disk format of the parked conversations (format v2; the disk tier only, see
// docs/NVFP4.md "Chunked parked dialogs").  A parked image is a small MANIFEST file (strata-conv-*.bin, magic
// "STRSESS\x02") plus content-addressed CHUNKS in a pool folder shared by every conversation of one identity:
//   <dir>/<xx>/<key32hex>.chk      xx = the key's first byte; 4096-byte header block, payload at 4096, size rounded
//                                  up to 4096 (O_DIRECT-aligned); key = SessionHasher::digest128 of the RAW bytes
// A K/V slab (T tokens of every K/V layer) and a 4 MiB slice of a checkpoint's running state are chunks; an image that
// continues a conversation re-uses every chunk whose bytes did not change.  Chunks are immutable and published by a
// no-replace rename, so writers need no lock; whether a chunk is still wanted is the tier's reference count (a view
// of the folder, rebuilt from the manifests), and an unreferenced chunk waits out a grace period before it is deleted.
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
    int64_t grace_s = 600;              // an unreferenced chunk is kept this long (from its last touch)
    std::function<int64_t()> clock_s;   // tests: seconds since the epoch for the grace comparisons (default: the system clock)
    // tests: return an errno value to make a step fail, 0 to go on: "chunk_write" (EINVAL while direct: the buffered
    // fallback), "chunk_sync" (buffered mode: the writeback step), "chunk_flush" (durable fdatasync), "chunk_rename",
    // "shard_sync"
    std::function<int(const char* step)> fault;
};
struct ChunkPutInfo {
    bool hit = false;                   // the chunk was already there (nothing written)
    uint32_t stored_len = 0;
    ChunkCodec codec = ChunkCodec::raw;
    bool shuffled = false;
    uint64_t new_file_bytes = 0;        // file bytes this call wrote
    bool direct_io = false;             // the write went through O_DIRECT
    SessionError error_kind = SessionError::none;   // after a failure
};

class ChunkPool {
public:
    explicit ChunkPool(ChunkPoolOptions);
    ~ChunkPool();
    ChunkPool(const ChunkPool&) = delete;
    ChunkPool& operator=(const ChunkPool&) = delete;
    // Creates the folder (0700) and indexes the chunks already there (unreferenced: they are ghosts until the owner
    // refs them or the grace runs out); removes the leftover temporary files older than 10 minutes.
    bool open(std::string& error);
    // Thread-safe.  The chunk is referenced (+1) for the caller on return; a writer that fails unref()s its whole list.
    // `buf` holds the raw bytes at payload().
    bool put_buffer(ChunkKind, ChunkBuffer& buf, size_t raw_len, uint8_t shuffle_width, ChunkRef& out,
                    ChunkPutInfo& info, std::string& error, bool durable = false);
    // The same from raw bytes elsewhere (one copy into a staging buffer).
    bool put(ChunkKind, const void* raw, size_t raw_len, uint8_t shuffle_width, ChunkRef& out, ChunkPutInfo& info,
             std::string& error, bool durable = false);
    // Thread-safe.  `dst` has room for ref.raw_len.  Verifies the size formula, the header hash, raw_len / stored_len /
    // key against `ref`, decodes with exactly raw_len, and the hash of the raw bytes against the key.
    bool get(ChunkKind, const ChunkRef&, void* dst, size_t dst_cap, std::string& error, SessionError* kind = nullptr);
    bool exists(const ChunkRef&) const;                     // stat + the size formula, no data read
    // References are this tier's view of the folder; a pin is one more ref.
    void ref(const ChunkRef*, size_t n);
    void unref(const ChunkRef*, size_t n);                  // at 0 the chunk becomes a ghost (its grace starts now)
    // Deletes ghosts.  Not shared: the owner of the folder - an unreferenced chunk goes when its grace ran out, or
    // right away under `pressure`; never a referenced or pinned one.  `shared`: other engines use the folder and their
    // references are invisible, so only a chunk that no manifest names (`foreign_refs`: every key any manifest in the
    // folder names) and whose file is older than the grace goes, re-checked by stat just before the unlink.
    size_t sweep(bool shared, bool pressure, const std::set<ChunkKey>* foreign_refs, size_t* freed_bytes = nullptr);
    // durable writes: fsync the shard folders touched since the last call
    bool sync_touched_shards(std::string& error);
    uint64_t bytes() const;                                 // file bytes of every known chunk, ghosts included
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
    int workers = 1;
    std::function<bool()> cancel;       // asked before every chunk: true gives the write up (error "cancelled")
    SessionProgress progress;           // (raw bytes done, raw bytes of the image) after every chunk
    SessionPhase phase;                 // "flush" and "publish" before each starts
    std::function<int(const char* step)> fault;   // the manifest's steps as SessionWriteOptions::fault: "write", "file_flush", "rename", "dir_flush"
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
    int64_t upto = 0;                   // > 0: the prefix image (not implemented yet: refused)
    int workers = 1;
};
// Full validation (manifest header, identity, hashes, bounded parse, every chunk) before `out` is replaced.
// limits.admit is asked once, with chunked_read_peak_bytes, before the image is allocated.
bool chunked_read(ChunkPool&, const std::string& manifest_path, const SessionFileIdentity&, SavedConversation& out,
                  size_t& manifest_bytes, uint64_t& raw_bytes_read, std::string& error, const SessionReadLimits&,
                  const ChunkedReadOptions& = {}, SessionStatus* = nullptr);
// The manifest's header and chunk references only (the tier's scan and rescan).
bool chunked_manifest_refs(const std::string& manifest_path, const SessionFileIdentity&, std::vector<ChunkRef>& refs,
                           uint64_t& manifest_bytes, std::string& error);
bool is_chunked_manifest(const std::string& path);          // the first 8 bytes are "STRSESS\x02"
// What the RAM preflight is asked for: the parsed image, three chunk buffers per worker (file, raw, decode temporary),
// the small vectors.  `max_chunk_raw`: the largest chunk of the manifest when it is parsed (0: a guess from chunk_tokens).
uint64_t chunked_read_peak_bytes(uint64_t raw_total, int workers, uint32_t chunk_tokens, uint64_t max_chunk_raw = 0);

} // namespace strata::core
