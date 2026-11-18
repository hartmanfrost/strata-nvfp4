#include "strata/core/conversation_chunks.hpp"
#include "strata/platform/aux_cpus.hpp"

#include <algorithm>
#include <bit>
#include <atomic>
#include <cerrno>
#include <condition_variable>
#include <deque>
#include <functional>
#include <thread>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <limits>
#include <mutex>
#include <new>
#include <random>
#include <system_error>
#include <set>
#include <unordered_map>
#ifdef _WIN32
#include <malloc.h>
#else
#include <fcntl.h>
#include <sys/file.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <sys/statvfs.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <unistd.h>
#endif

static_assert(std::endian::native == std::endian::little, "chunked session files (format v2) are little-endian only");
static_assert(std::numeric_limits<float>::is_iec559 && std::numeric_limits<double>::is_iec559,
              "chunked session files (format v2) need IEEE-754 floats");

namespace strata::core {
namespace {
namespace fs = std::filesystem;

constexpr char kChunkMagic[8] = {'S', 'T', 'R', 'C', 'H', 'N', 'K', '1'};
constexpr char kManMagic[8] = {'S', 'T', 'R', 'S', 'E', 'S', 'S', '\x02'};
constexpr char kManEnd[8] = {'S', 'T', 'R', 'S', 'E', 'N', 'D', '\x02'};
constexpr uint32_t kManVersion = 2;
constexpr size_t kHeader = 64, kTrailer = 16;
constexpr uint64_t kMaxManifest = uint64_t(256) << 20;   // the largest manifest file read into memory
constexpr uint64_t kMaxRow = uint64_t(1) << 20;          // the largest pooled row (the spare row rides inline)
constexpr int64_t kTmpMaxAgeS = 600;                     // a leftover temporary file older than this is removed
constexpr size_t kLayerMin = 7 * 8 + 5 * 8 + 5;          // the fewest manifest bytes one K/V layer takes
constexpr size_t kCheckpointMin = 6 * 8 + 8 + 8 + 8;     // six counts, used, gdn_len, slice_len + n_slices (72)
constexpr size_t kRefBytes = 24;

uint64_t round_up(uint64_t x, uint64_t a) { return (x + a - 1) / a * a; }
uint64_t sat_add(uint64_t a, uint64_t b) { return a > UINT64_MAX - b ? UINT64_MAX : a + b; }
uint64_t sat_mul(uint64_t a, uint64_t b) { return a && b > UINT64_MAX / a ? UINT64_MAX : a * b; }

SessionError errno_kind(int e) {
    switch (e) {
        case ENOSPC:
#ifdef EDQUOT
        case EDQUOT:
#endif
        case EFBIG: return SessionError::storage;
        case ENOMEM: return SessionError::memory;
        default: return SessionError::io;
    }
}

int injected(const std::function<int(const char*)>& fault, const char* step) { return fault ? fault(step) : 0; }

uint64_t random64() {
    std::random_device rd;
    uint64_t r = (uint64_t(rd()) << 32) ^ rd();
    r ^= (uint64_t) std::chrono::steady_clock::now().time_since_epoch().count() * 0x9E3779B185EBCA87ull;
#ifndef _WIN32
    r ^= (uint64_t) ::getpid() * 0xC2B2AE3D27D4EB4Full;
#endif
    return r;
}
std::string hex64(uint64_t v) {
    char h[17];
    std::snprintf(h, sizeof h, "%016llx", (unsigned long long) v);
    return h;
}

#ifdef _WIN32
uint8_t* alloc_aligned(size_t n) { return static_cast<uint8_t*>(_aligned_malloc(n, kChunkAlign)); }
void free_aligned(void* p) { _aligned_free(p); }
#else
uint8_t* alloc_aligned(size_t n) { return static_cast<uint8_t*>(std::aligned_alloc(kChunkAlign, n)); }
void free_aligned(void* p) { std::free(p); }
#endif
struct AlignedFree { void operator()(uint8_t* p) const { free_aligned(p); } };
using Aligned = std::unique_ptr<uint8_t, AlignedFree>;
Aligned aligned_bytes(size_t n) {
    return Aligned(alloc_aligned(round_up(std::max<size_t>(n, 1), kChunkAlign)));
}

template<class T> void put_le(uint8_t* p, T v) { std::memcpy(p, &v, sizeof v); }
template<class T> T get_le(const uint8_t* p) { T v; std::memcpy(&v, p, sizeof v); return v; }

// ------------------------------------------------------------------------------------------------ chunk header

void make_header(uint8_t* h, uint32_t raw_len, uint32_t stored_len, ChunkCodec codec, bool shuffled, ChunkKind kind,
                 const ChunkKey& key) {
    std::memset(h, 0, kChunkAlign);
    std::memcpy(h, kChunkMagic, 8);
    put_le<uint32_t>(h + 8, raw_len);
    put_le<uint32_t>(h + 12, stored_len);
    h[16] = (uint8_t) codec;
    h[17] = (codec != ChunkCodec::raw ? 2 : 0) | (shuffled ? 1 : 0);   // bit1: segmented container (every non-raw codec)
    put_le<uint16_t>(h + 18, (uint16_t) kind);
    std::memcpy(h + 24, key.b.data(), 16);
    put_le<uint64_t>(h + 40, session_hash64(h, 40, 0));
}

struct Hdr {
    uint32_t raw_len = 0, stored_len = 0;
    ChunkCodec codec = ChunkCodec::raw;
    bool shuffled = false;
    uint16_t kind = 0;
    ChunkKey key;
};

// the header block's own checks (no outside expectation); null when fine
const char* parse_header(const uint8_t* h, Hdr& out) {
    if (std::memcmp(h, kChunkMagic, 8) != 0) return "not a chunk (magic)";
    if (get_le<uint64_t>(h + 40) != session_hash64(h, 40, 0)) return "header checksum mismatch";
    if (get_le<uint32_t>(h + 20) != 0) return "invalid header fields";
    out.raw_len = get_le<uint32_t>(h + 8);
    out.stored_len = get_le<uint32_t>(h + 12);
    const uint8_t codec = h[16], flags = h[17];
    if (codec != (uint8_t) ChunkCodec::raw && codec != (uint8_t) ChunkCodec::zstd) return "unknown codec";
    if (flags & ~3u) return "invalid header flags";
    if (((flags & 2u) != 0) != (codec != (uint8_t) ChunkCodec::raw)) return "invalid header flags";
    out.codec = (ChunkCodec) codec;
    out.shuffled = (flags & 1u) != 0;
    out.kind = get_le<uint16_t>(h + 18);
    if (out.kind > (uint16_t) ChunkKind::state_slice) return "unknown chunk kind";
    std::memcpy(out.key.b.data(), h + 24, 16);
    if (out.raw_len > kMaxChunkRaw) return "raw length over the limit";
    if (out.stored_len > out.raw_len) return "stored length over the raw length";
    if (out.codec == ChunkCodec::raw && (out.stored_len != out.raw_len || out.shuffled)) return "invalid raw chunk";
    return nullptr;
}

#ifndef _WIN32
#ifndef O_DIRECT
#define O_DIRECT 0
#endif
bool buffered_forced() {
    const char* e = std::getenv("STRATA_SESSION_BUFFERED");
    return e != nullptr && e[0] == '1';
}
// O_DIRECT is switched on the open descriptor: a filesystem that refuses it keeps buffered I/O
bool try_direct(int fd) {
    if (O_DIRECT == 0 || buffered_forced()) return false;
    const int fl = ::fcntl(fd, F_GETFL);
    return fl >= 0 && ::fcntl(fd, F_SETFL, fl | O_DIRECT) == 0;
}
bool leave_direct(int fd, bool& direct) {
    const int fl = ::fcntl(fd, F_GETFL);
    if (fl < 0 || ::fcntl(fd, F_SETFL, fl & ~O_DIRECT) != 0) return false;
    direct = false;
    return true;
}
// buffered I/O must not leave the file's pages in the page cache (charged to the memory cgroup): write them back,
// then drop them
void writeback_and_drop(int fd) {
#if defined(__linux__)
    ::sync_file_range(fd, 0, 0, SYNC_FILE_RANGE_WRITE | SYNC_FILE_RANGE_WAIT_AFTER);
#endif
    ::posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
}

class Fd {
public:
    Fd() = default;
    explicit Fd(int fd) : fd_(fd) {}
    Fd(Fd&& o) noexcept : fd_(o.fd_) { o.fd_ = -1; }
    Fd& operator=(Fd&& o) noexcept { if (this != &o) { reset(); fd_ = o.fd_; o.fd_ = -1; } return *this; }
    ~Fd() { reset(); }
    int get() const { return fd_; }
    explicit operator bool() const { return fd_ >= 0; }
    int release() { const int f = fd_; fd_ = -1; return f; }
    bool close() { const int f = fd_; fd_ = -1; return f < 0 || ::close(f) == 0; }
private:
    void reset() { if (fd_ >= 0) ::close(fd_); fd_ = -1; }
    int fd_ = -1;
};

int write_file_all(int fd, const uint8_t* p, size_t n, bool& direct, const ChunkPoolOptions& o) {
    if (const int inj = injected(o.fault, "chunk_write")) {
        if (inj == EINVAL && direct) leave_direct(fd, direct);   // the injected "this transfer is refused"
        else return inj;
    }
    size_t off = 0;
    while (off < n) {
        const ssize_t w = ::pwrite(fd, p + off, n - off, (off_t) off);
        if (w < 0 && errno == EINTR) continue;
        // a filesystem that took O_DIRECT at open but refuses this transfer: nothing was written, go buffered
        if (w < 0 && errno == EINVAL && direct && leave_direct(fd, direct)) continue;
        if (w <= 0) return w < 0 ? errno : EIO;
        off += (size_t) w;
    }
    return 0;
}

bool pread_all(int fd, uint8_t* p, size_t n, bool& direct, std::string& err) {
    size_t off = 0;
    while (off < n) {
        const ssize_t r = ::pread(fd, p + off, n - off, (off_t) off);
        if (r < 0 && errno == EINTR) continue;
        if (r < 0 && errno == EINVAL && direct && leave_direct(fd, direct)) continue;
        if (r <= 0) { err = r < 0 ? std::strerror(errno) : "file ends early"; return false; }
        off += (size_t) r;
    }
    return true;
}

bool free_bytes(const std::string& dir, uint64_t& out) {
    struct statvfs s {};
    if (::statvfs(dir.c_str(), &s) != 0) return false;
    out = (uint64_t) s.f_bavail * (uint64_t) s.f_frsize;
    return true;
}

// an open regular file with one name, never a link; size via fstat
bool open_regular(const std::string& path, Fd& fd, struct stat& st, std::string& err, bool& refused, int* eno = nullptr) {
    refused = false;
    fd = Fd(::open(path.c_str(), O_RDONLY | O_NOFOLLOW | O_NONBLOCK | O_CLOEXEC));
    if (!fd) {
        const int e = errno;
        if (eno) *eno = e;
        refused = e == ELOOP;
        err = e == ELOOP ? std::string("a symbolic link, not a file") : std::string(std::strerror(e));
        return false;
    }
    if (::fstat(fd.get(), &st) != 0) { if (eno) *eno = errno; err = std::strerror(errno); return false; }
    if (!S_ISREG(st.st_mode)) { refused = true; err = "not a regular file"; return false; }
    if (st.st_nlink != 1) { refused = true; err = "the file has more than one name (hard link)"; return false; }
    const int fl = ::fcntl(fd.get(), F_GETFL);
    if (fl >= 0) ::fcntl(fd.get(), F_SETFL, fl & ~O_NONBLOCK);
    return true;
}
#endif

std::string hex_of(const std::array<uint8_t, 16>& b) {
    static const char* d = "0123456789abcdef";
    std::string s;
    for (uint8_t c : b) { s += d[c >> 4]; s += d[c & 15]; }
    return s;
}
bool key_from_hex(const std::string& s, ChunkKey& k) {
    if (s.size() != 32) return false;
    auto nib = [](char c) { return c >= '0' && c <= '9' ? c - '0' : c >= 'a' && c <= 'f' ? c - 'a' + 10 : -1; };
    for (size_t i = 0; i < 16; ++i) {
        const int a = nib(s[2 * i]), b = nib(s[2 * i + 1]);
        if (a < 0 || b < 0) return false;
        k.b[i] = (uint8_t) (a << 4 | b);
    }
    return true;
}
std::string shard_of(const ChunkKey& k) {
    char b[3];
    std::snprintf(b, sizeof b, "%02x", k.b[0]);
    return b;
}

struct KeyHash {
    size_t operator()(const ChunkKey& k) const { uint64_t v; std::memcpy(&v, k.b.data(), 8); return (size_t) v; }
};
} // namespace

std::string ChunkKey::hex() const { return hex_of(b); }

uint64_t chunk_file_size(uint32_t stored_len) { return kChunkAlign + round_up(stored_len, kChunkAlign); }

ChunkKey chunk_key(const void* raw, size_t n) {
    SessionHasher h(0);
    h.update(raw, n);
    ChunkKey k;
    k.b = h.digest128();
    return k;
}

// ------------------------------------------------------------------------------------------------ ChunkBuffer

ChunkBuffer::ChunkBuffer() = default;
ChunkBuffer::~ChunkBuffer() { free_aligned(data_); }
ChunkBuffer::ChunkBuffer(ChunkBuffer&& o) noexcept : data_(o.data_), cap_(o.cap_), alloc_(o.alloc_) {
    o.data_ = nullptr; o.cap_ = o.alloc_ = 0;
}
ChunkBuffer& ChunkBuffer::operator=(ChunkBuffer&& o) noexcept {
    if (this == &o) return *this;
    free_aligned(data_);
    data_ = o.data_; cap_ = o.cap_; alloc_ = o.alloc_;
    o.data_ = nullptr; o.cap_ = o.alloc_ = 0;
    return *this;
}
bool ChunkBuffer::reserve(size_t raw_cap) {
    if (data_ && cap_ >= raw_cap) return true;
    if (raw_cap > kMaxChunkRaw) return false;
    const size_t alloc = kChunkAlign + round_up(std::max<size_t>(raw_cap, 1), kChunkAlign);
    uint8_t* d = alloc_aligned(alloc);
    if (!d) return false;
    free_aligned(data_);
    data_ = d; alloc_ = alloc; cap_ = alloc - kChunkAlign;
    return true;
}

// ------------------------------------------------------------------------------------------------ ChunkPool

struct ChunkPool::Impl {
    struct Info {
        uint64_t file_bytes = 0;
        uint32_t refs = 0, stored_len = 0;
        bool stored_known = false;
        uint64_t seq = 0;           // last use (release, hit, read): wall-clock nanoseconds, strictly increasing; a ghost's LRU key
    };
    explicit Impl(ChunkPoolOptions opt) : o(std::move(opt)) {}
    ChunkPoolOptions o;
    mutable std::mutex mu;
    std::unordered_map<ChunkKey, Info, KeyHash> idx;
    uint64_t bytes = 0;
    uint64_t ghost_bytes = 0;           // file bytes of the unreferenced chunks (exactly ghost_lru's)
    uint64_t last_tick = 0;
    std::set<std::pair<uint64_t, ChunkKey>> ghost_lru;   // every chunk with refs == 0, oldest (smallest seq) first
    std::set<std::string> touched;      // shard names; "" = the pool folder itself
    std::mutex sync_mu;                 // one sync_touched_shards at a time, held until its fsyncs are done
    AdaptiveSkip skip;                  // which kinds of data the codec is tried on
    bool opened = false;

    // (mu held) a strictly increasing wall-clock nanosecond stamp: comparable with the files' mtimes, so the ghosts
    // this process released and the ones it found on the disk (a restart, another engine) share one order
    uint64_t tick() {
        const uint64_t t = (uint64_t) std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
        last_tick = t > last_tick ? t : last_tick + 1;
        return last_tick;
    }
    // (mu held) the LRU bookkeeping: a ghost is in ghost_lru exactly while refs == 0
    void ghost_add(const ChunkKey& k, Info& e, uint64_t seq) {
        e.seq = seq;
        ghost_lru.insert({seq, k});
        ghost_bytes += e.file_bytes;
    }
    void ghost_del(const ChunkKey& k, const Info& e) {
        if (ghost_lru.erase({e.seq, k})) ghost_bytes -= std::min(ghost_bytes, e.file_bytes);
    }
    void ref_inc(const ChunkKey& k, Info& e) {
        if (e.refs == 0) ghost_del(k, e);
        ++e.refs;
    }
    // refs 1 -> 0: the chunk becomes the newest ghost
    void ref_dec(const ChunkKey& k, Info& e) {
        if (e.refs == 0) return;
        if (--e.refs == 0) ghost_add(k, e, tick());
    }
    // forgets the index entry (the file is gone or being removed): bytes and the ghost set follow
    std::unordered_map<ChunkKey, Info, KeyHash>::iterator drop(std::unordered_map<ChunkKey, Info, KeyHash>::iterator it) {
        if (it->second.refs == 0) ghost_del(it->first, it->second);
        bytes -= std::min(bytes, it->second.file_bytes);
        return idx.erase(it);
    }
    std::string path_of(const ChunkKey& k) const { return o.dir + "/" + shard_of(k) + "/" + k.hex() + ".chk"; }
    std::string shard_dir(const ChunkKey& k) const { return o.dir + "/" + shard_of(k); }

#ifndef _WIN32
    // sets the file's mtime to now: the persisted LRU order (a restart, other engines' censuses read it back)
    void stamp_file(const std::string& path) const {
        struct timespec ts[2];
        ts[0].tv_sec = 0; ts[0].tv_nsec = UTIME_OMIT;
        ts[1].tv_sec = 0; ts[1].tv_nsec = UTIME_NOW;
        ::utimensat(AT_FDCWD, path.c_str(), ts, 0);
    }

    // every key the live leases of the folder name; a lease whose flock can be taken belongs to a dead process: removed
    void read_leases(std::set<ChunkKey>& out) const {
        std::error_code ec;
        for (const auto& f : fs::directory_iterator(o.dir, ec)) {
            const std::string name = f.path().filename().string();
            if (name.rfind(".lease-", 0) != 0 || name.rfind(".lease-tmp-", 0) == 0) continue;
            Fd fd(::open(f.path().c_str(), O_RDONLY | O_CLOEXEC));
            if (!fd) continue;
            if (::flock(fd.get(), LOCK_EX | LOCK_NB) == 0) { ::unlink(f.path().c_str()); continue; }   // stale
            uint8_t buf[16 * 64];
            off_t off = 0;
            for (;;) {
                const ssize_t n = ::pread(fd.get(), buf, sizeof buf, off);
                if (n <= 0) break;
                for (ssize_t i = 0; i + 16 <= n; i += 16) {
                    ChunkKey k;
                    std::memcpy(k.b.data(), buf + i, 16);
                    out.insert(k);
                }
                off += (n / 16) * 16;
                if (n < (ssize_t) sizeof buf) break;
            }
        }
    }

    // what a look at an existing chunk file found: a valid chunk, no file, a file that is definitely not this chunk
    // (wrong header, size, name; a link), or a failure of the look itself (resources, I/O) that says nothing about the file
    enum class Verdict { ok, absent, invalid, error };

    // does a valid chunk with this key (and raw length) exist: stat, header block, size formula.  `durable`: a valid
    // chunk is also flushed (fdatasync) and its shard folder queued for the next sync_touched_shards, so a durable
    // write that only hits a chunk an earlier non-durable write made leaves it as safe as one it wrote itself
    Verdict validate_existing(const ChunkKey& key, uint32_t raw_len, uint32_t& stored_len, uint64_t& file_bytes,
                              std::string& why, SessionError& kind, bool durable) {
        Fd fd;
        struct stat st {};
        bool refused = false;
        int eno = 0;
        if (const int inj = injected(o.fault, "chunk_validate")) { why = std::strerror(inj); kind = errno_kind(inj); return Verdict::error; }
        if (!open_regular(path_of(key), fd, st, why, refused, &eno)) {
            if (refused) return Verdict::invalid;
            if (eno == ENOENT) return Verdict::absent;
            kind = errno_kind(eno ? eno : EIO);
            return Verdict::error;
        }
        const uint64_t size = (uint64_t) st.st_size;
        if (size < kChunkAlign || size > kMaxChunkFile || size % kChunkAlign) { why = "bad chunk file size"; return Verdict::invalid; }
        Aligned h = aligned_bytes(kChunkAlign);
        if (!h) { why = "out of memory"; kind = SessionError::memory; return Verdict::error; }
        bool direct = false;
        if (!pread_all(fd.get(), h.get(), kChunkAlign, direct, why)) { kind = SessionError::io; return Verdict::error; }
        ::posix_fadvise(fd.get(), 0, kChunkAlign, POSIX_FADV_DONTNEED);
        Hdr hd;
        if (const char* bad = parse_header(h.get(), hd)) { why = bad; return Verdict::invalid; }
        if (!(hd.key == key) || hd.raw_len != raw_len) { why = "chunk header does not match its name"; return Verdict::invalid; }
        if (chunk_file_size(hd.stored_len) != size) { why = "chunk size does not match its header"; return Verdict::invalid; }
        if (durable) {
            int e = injected(o.fault, "chunk_flush");
            if (!e && ::fdatasync(fd.get()) != 0) e = errno;
            if (e) { why = std::string("flushing a chunk: ") + std::strerror(e); kind = errno_kind(e); return Verdict::error; }
            std::lock_guard<std::mutex> g(mu);
            touched.insert(shard_of(key));
        }
        stored_len = hd.stored_len;
        file_bytes = size;
        return Verdict::ok;
    }

    // A chunk whose payload failed its hash (or whose header is wrong): moved out of the way so the next put of
    // these bytes writes a fresh file instead of hitting the bad one.  Only the file that was read (same inode) goes -
    // another writer may have replaced it since.  The index entry stays (the manifests that name the chunk still hold
    // their references; the replacement joins the same entry), unless nothing references it.
    void quarantine(const ChunkKey& key, const struct stat& read_st) {
        const std::string path = path_of(key);
        struct stat now_st {};
        if (::lstat(path.c_str(), &now_st) != 0 || now_st.st_ino != read_st.st_ino || now_st.st_dev != read_st.st_dev) return;
        const std::string aside = shard_dir(key) + "/.tmp-bad-" + hex64(random64());   // open() sweeps .tmp-* leftovers
        if (::rename(path.c_str(), aside.c_str()) != 0) ::unlink(path.c_str());
        std::lock_guard<std::mutex> g(mu);
        auto it = idx.find(key);
        if (it != idx.end() && it->second.refs == 0) drop(it);
    }

    bool shard_ready(const ChunkKey& key, std::string& error, SessionError& kind) {
        const std::string d = shard_dir(key);
        if (::mkdir(d.c_str(), 0700) == 0) {
            std::lock_guard<std::mutex> g(mu);
            touched.insert("");
        } else if (errno != EEXIST) {
            kind = errno_kind(errno);
            error = "chunk folder " + d + ": " + std::strerror(errno);
            return false;
        }
        return true;
    }

    // writes `buf` (header + payload, padded) as a new chunk file; the hit/EEXIST answer is in `exists`
    bool write_new(const ChunkKey& key, ChunkBuffer& buf, uint32_t stored_len, ChunkPutInfo& info, bool durable,
                   bool& exists, std::string& error) {
        exists = false;
        const uint64_t file_size = chunk_file_size(stored_len);
        const std::string dir = shard_dir(key);
        Fd dirfd(::open(dir.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC));
        if (!dirfd) { info.error_kind = errno_kind(errno); error = "chunk folder " + dir + ": " + std::strerror(errno); return false; }
        std::string tmp;
        Fd fd;
        for (int attempt = 0; attempt < 8 && !fd; ++attempt) {
            tmp = ".tmp-" + hex64(random64());
            fd = Fd(::openat(dirfd.get(), tmp.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC, 0600));
            if (!fd && errno != EEXIST) break;
        }
        if (!fd) { info.error_kind = errno_kind(errno); error = "cannot create a chunk file in " + dir + ": " + std::strerror(errno); return false; }
        bool published = false;
        auto cleanup = [&] { if (!published) ::unlinkat(dirfd.get(), tmp.c_str(), 0); };
        bool direct = o.direct && try_direct(fd.get());
        int e = write_file_all(fd.get(), buf.data_, (size_t) file_size, direct, o);
        if (e) { cleanup(); info.error_kind = errno_kind(e); error = std::string("writing a chunk: ") + std::strerror(e); return false; }
        if (!direct) {
            // the buffered fallback: the same file, written back and dropped so dirty pages do not pile up in the cgroup
            if (const int inj = injected(o.fault, "chunk_sync")) { cleanup(); info.error_kind = errno_kind(inj); error = std::string("chunk writeback: ") + std::strerror(inj); return false; }
            writeback_and_drop(fd.get());
        }
        if (durable) {
            e = injected(o.fault, "chunk_flush");
            if (!e && ::fdatasync(fd.get()) != 0) e = errno;
            if (e) { cleanup(); info.error_kind = errno_kind(e); error = std::string("flushing a chunk: ") + std::strerror(e); return false; }
        }
        if (!fd.close()) { e = errno; cleanup(); info.error_kind = errno_kind(e); error = std::string("closing a chunk: ") + std::strerror(e); return false; }
        info.direct_io = direct;
        e = injected(o.fault, "chunk_rename");
        const std::string final_name = key.hex() + ".chk";
        if (!e) {
#if defined(__linux__) && defined(RENAME_NOREPLACE)
            if (::renameat2(dirfd.get(), tmp.c_str(), dirfd.get(), final_name.c_str(), RENAME_NOREPLACE) != 0) {
                e = errno;
                if (e == ENOSYS || e == EINVAL) {   // a kernel or filesystem without RENAME_NOREPLACE: link + unlink
                    e = ::linkat(dirfd.get(), tmp.c_str(), dirfd.get(), final_name.c_str(), 0) == 0 ? 0 : errno;
                    if (e == 0) ::unlinkat(dirfd.get(), tmp.c_str(), 0);
                }
            }
#else
            e = ::linkat(dirfd.get(), tmp.c_str(), dirfd.get(), final_name.c_str(), 0) == 0 ? 0 : errno;
            if (e == 0) ::unlinkat(dirfd.get(), tmp.c_str(), 0);
#endif
        }
        if (e == EEXIST) {   // another writer published the same key first
            ::unlinkat(dirfd.get(), tmp.c_str(), 0);
            published = true;
            exists = true;
            return true;
        }
        if (e) { cleanup(); info.error_kind = errno_kind(e); error = std::string("publishing a chunk: ") + std::strerror(e); return false; }
        published = true;
        if (durable) { std::lock_guard<std::mutex> g(mu); touched.insert(shard_of(key)); }
        info.new_file_bytes = file_size;
        return true;
    }
#endif
};

#ifndef _WIN32
namespace {
uint64_t mtime_ns(const struct stat& st) {
#ifdef __APPLE__
    return (uint64_t) st.st_mtimespec.tv_sec * 1000000000ull + (uint64_t) st.st_mtimespec.tv_nsec;
#else
    return (uint64_t) st.st_mtim.tv_sec * 1000000000ull + (uint64_t) st.st_mtim.tv_nsec;
#endif
}
}  // namespace
#endif

ChunkPool::ChunkPool(ChunkPoolOptions o) : p_(new Impl(std::move(o))) {}
ChunkPool::~ChunkPool() = default;
const ChunkPoolOptions& ChunkPool::options() const { return p_->o; }
std::string ChunkPool::path_of(const ChunkKey& k) const { return p_->path_of(k); }

#ifdef _WIN32
bool ChunkPool::open(std::string& error) { error = "the chunked disk format is POSIX only"; return false; }
bool ChunkPool::put_buffer(ChunkKind, ChunkBuffer&, size_t, uint8_t, ChunkRef&, ChunkPutInfo&, std::string& e, bool,
                           const std::vector<CodecSegment>*, ChunkLease*) { e = "the chunked disk format is POSIX only"; return false; }
bool ChunkPool::get(ChunkKind, const ChunkRef&, void*, size_t, std::string& e, SessionError*) { e = "the chunked disk format is POSIX only"; return false; }
bool ChunkPool::put(ChunkKind, const void*, size_t, uint8_t, ChunkRef&, ChunkPutInfo&, std::string& e, bool) { e = "the chunked disk format is POSIX only"; return false; }
bool ChunkPool::exists(const ChunkRef&) const { return false; }
void ChunkPool::ref(const ChunkRef*, size_t) {}
void ChunkPool::unref(const ChunkRef*, size_t) {}
size_t ChunkPool::evict_ghosts(uint64_t, const std::function<bool(std::set<ChunkKey>&)>&, uint64_t*, bool*) { return 0; }
void ChunkPool::census() {}
std::unique_ptr<ChunkLease> ChunkPool::begin_lease() { return nullptr; }
struct ChunkLease::Impl {};
ChunkLease::~ChunkLease() = default;
void ChunkLease::announce(const ChunkKey&) {}
void ChunkLease::announce(const ChunkRef*, size_t) {}
void ChunkLease::release() {}
bool ChunkPool::sync_touched_shards(std::string&) { return true; }
void ChunkPool::remove_all() {}
#else
bool ChunkPool::open(std::string& error) {
    auto& p = *p_;
    std::error_code ec;
    fs::create_directories(p.o.dir, ec);
    if (ec) { error = "chunk pool " + p.o.dir + ": " + ec.message(); return false; }
    ::chmod(p.o.dir.c_str(), 0700);
    std::lock_guard<std::mutex> g(p.mu);
    p.idx.clear();
    p.ghost_lru.clear();
    p.bytes = p.ghost_bytes = 0;
    const int64_t now = (int64_t) std::time(nullptr);
    std::vector<std::pair<uint64_t, ChunkKey>> found;   // (mtime ns, key): the ghosts' order, oldest first
    for (const auto& shard : fs::directory_iterator(p.o.dir, ec)) {
        std::error_code e2;
        const std::string sname = shard.path().filename().string();
        if (sname.size() != 2 || !shard.is_directory(e2)) continue;
        for (const auto& f : fs::directory_iterator(shard.path(), e2)) {
            const std::string name = f.path().filename().string();
            struct stat st {};
            if (::lstat(f.path().c_str(), &st) != 0 || !S_ISREG(st.st_mode)) continue;
            if (name.rfind(".tmp-", 0) == 0 || name.rfind(".del-", 0) == 0) {   // (.del-: a victim an evicting engine died over)
                if (now - (int64_t) st.st_mtime > kTmpMaxAgeS) ::unlink(f.path().c_str());
                continue;
            }
            ChunkKey k;
            if (name.size() != 36 || name.compare(32, 4, ".chk") != 0 || !key_from_hex(name.substr(0, 32), k) ||
                shard_of(k) != sname) continue;
            const uint64_t size = (uint64_t) st.st_size;
            if (size < kChunkAlign || size > kMaxChunkFile || size % kChunkAlign) continue;
            Impl::Info info;
            info.file_bytes = size;
            p.idx[k] = info;
            p.bytes += size;
            found.emplace_back(mtime_ns(st), k);
        }
    }
    // every chunk is a ghost until a manifest refs it, ordered by the persisted mtime (ns): the order before the restart
    std::sort(found.begin(), found.end());
    for (const auto& f : found) p.ghost_add(f.second, p.idx[f.second], f.first);
    p.opened = true;
    return true;
}

bool ChunkPool::put_buffer(ChunkKind kind, ChunkBuffer& buf, size_t raw_len, uint8_t shuffle_width, ChunkRef& out,
                           ChunkPutInfo& info, std::string& error, bool durable,
                           const std::vector<CodecSegment>* segments, ChunkLease* lease) {
    auto& p = *p_;
    info = {};
    error.clear();
    if (raw_len > kMaxChunkRaw || raw_len > buf.capacity()) {
        info.error_kind = SessionError::invalid;
        error = "chunk: the payload is over the limit";
        return false;
    }
    const ChunkKey key = chunk_key(buf.payload(), raw_len);
    out = {};
    out.key = key;
    out.raw_len = (uint32_t) raw_len;
    // shared: the key is announced before the chunk is looked at or written, so a deleter that already snapshotted the
    // manifests still finds it in the leases when it checks again (ChunkLease)
    if (lease) lease->announce(key);

    // an existing chunk: pinned in the index first (a sweep cannot take it now), then checked on the disk
    using Verdict = Impl::Verdict;
    std::string hit_why;
    auto try_hit = [&]() -> Verdict {
        bool pinned = false;
        {
            std::lock_guard<std::mutex> g(p.mu);
            auto it = p.idx.find(key);
            if (it != p.idx.end()) { p.ref_inc(key, it->second); pinned = true; }
        }
        uint32_t stored = 0;
        uint64_t file_bytes = 0;
        SessionError why_kind = SessionError::io;
        hit_why.clear();
        const Verdict verdict = p.validate_existing(key, out.raw_len, stored, file_bytes, hit_why, why_kind, durable);
        std::lock_guard<std::mutex> g(p.mu);
        auto it = p.idx.find(key);
        if (verdict != Verdict::ok) {
            if (pinned && it != p.idx.end()) {
                // the chunk is not what the index thought (gone or invalid): the unpin forgets it when nothing else
                // references it (it was pinned, so it is not in the LRU set)
                if (--it->second.refs == 0) p.drop(it);
            }
            if (verdict == Verdict::error) info.error_kind = why_kind;
            return verdict;
        }
        // a hit is a use: the chunk is the newest in the LRU order, in memory and on the disk (every hit: the mtime is
        // the order a restart and the other engines see)
        if (it == p.idx.end()) {
            Impl::Info n;
            n.file_bytes = file_bytes; n.refs = 1; n.stored_len = stored; n.stored_known = true; n.seq = p.tick();
            p.idx[key] = n;
            p.bytes += file_bytes;
        } else {
            auto& e = it->second;
            if (!pinned) p.ref_inc(key, e);
            if (e.file_bytes != file_bytes) { p.bytes = p.bytes - std::min(p.bytes, e.file_bytes) + file_bytes; e.file_bytes = file_bytes; }
            e.stored_len = stored; e.stored_known = true;
            e.seq = p.tick();
        }
        p.stamp_file(p.path_of(key));
        out.stored_len = stored;
        info.hit = true;
        info.stored_len = stored;
        return Verdict::ok;
    };
    auto look_failed = [&]() {   // the look itself failed: nothing is known about the file, so nothing is done to it
        if (info.error_kind == SessionError::none) info.error_kind = SessionError::io;
        error = "chunk " + key.hex() + ": " + hit_why;
        return false;
    };
    {
        const Verdict v = try_hit();
        if (v == Verdict::ok) return true;
        if (v == Verdict::error) return look_failed();
        // absent: a miss; invalid: the write below finds the name taken and replaces the file
    }

    // a miss: encode, write, publish
    uint32_t stored_len = (uint32_t) raw_len;
    ChunkCodec codec = ChunkCodec::raw;
    bool shuffled = false;
    if (p.o.codec != ChunkCodec::raw && codec_available(p.o.codec) && raw_len) {
        // the segments the codec treats separately (K/V parts differ in element width and compressibility); without a
        // plan the chunk is one segment of the given width
        std::vector<CodecSegment> one;
        if (!segments) {
            one.push_back({(uint32_t) raw_len, shuffle_width, (uint8_t) (kind == ChunkKind::state_slice ? 5 : 6)});
            segments = &one;
        }
        std::vector<uint8_t> enc;
        bool any_shuffled = false;
        const auto t0 = std::chrono::steady_clock::now();
        if (!segmented_encode(p.o.zstd_level, segments->data(), segments->size(), buf.payload(), raw_len, &p.skip, enc,
                              any_shuffled)) {
            info.error_kind = SessionError::io;
            error = "chunk: the encoder failed";
            return false;
        }
        info.encode_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        if (enc.size() < raw_len) {
            std::memcpy(buf.payload(), enc.data(), enc.size());
            stored_len = (uint32_t) enc.size();
            codec = p.o.codec;
            shuffled = any_shuffled;
        }
    }
    const uint64_t file_size = chunk_file_size(stored_len);
    if (!p.shard_ready(key, error, info.error_kind)) return false;
    if (p.o.min_free_bytes) {
        uint64_t avail = 0;
        if (!free_bytes(p.o.dir, avail)) { info.error_kind = SessionError::io; error = "chunk: cannot read the free disk space"; return false; }
        if (avail < file_size || avail - file_size < p.o.min_free_bytes) {
            info.error_kind = SessionError::storage;
            error = "chunk: not enough disk space (" + std::to_string(file_size >> 10) + " KiB plus a reserve of " +
                    std::to_string(p.o.min_free_bytes >> 20) + " MiB needed, " + std::to_string(avail >> 20) + " MiB free)";
            return false;
        }
    }
    make_header(buf.data_, (uint32_t) raw_len, stored_len, codec, shuffled, kind, key);
    std::memset(buf.data_ + kChunkAlign + stored_len, 0, (size_t) (file_size - kChunkAlign - stored_len));
    bool exists = false;
    if (!p.write_new(key, buf, stored_len, info, durable, exists, error)) return false;
    if (exists) {
        // another writer published this name first: a valid chunk is a hit
        Verdict v = try_hit();
        if (v == Verdict::ok) return true;
        if (v == Verdict::error) return look_failed();
        if (v == Verdict::invalid) {
            // the file under that name is not a valid chunk (a torn or foreign one): replace it - but only on a second,
            // definite look (a writer racing this one may just have published the valid file), never on a failed look
            v = try_hit();
            if (v == Verdict::ok) return true;
            if (v == Verdict::error) return look_failed();
            if (v == Verdict::invalid) ::unlink(p.path_of(key).c_str());
        }
        if (!p.write_new(key, buf, stored_len, info, durable, exists, error)) return false;
        if (exists) { info.error_kind = SessionError::io; error = "chunk: cannot replace an invalid chunk file"; return false; }
    }
    {
        std::lock_guard<std::mutex> g(p.mu);
        auto it = p.idx.find(key);
        if (it == p.idx.end()) {
            p.idx[key] = {file_size, 1, stored_len, true, p.tick()};
            p.bytes += file_size;
        } else {
            auto& e = it->second;
            p.ref_inc(key, e);
            p.bytes = p.bytes - std::min(p.bytes, e.file_bytes) + file_size;
            e.file_bytes = file_size; e.stored_len = stored_len; e.stored_known = true; e.seq = p.tick();
        }
    }
    out.stored_len = stored_len;
    info.stored_len = stored_len;
    info.codec = codec;
    info.shuffled = shuffled;
    return true;
}

bool ChunkPool::put(ChunkKind kind, const void* raw, size_t raw_len, uint8_t shuffle_width, ChunkRef& out,
                    ChunkPutInfo& info, std::string& error, bool durable) {
    ChunkBuffer buf;
    if (raw_len > kMaxChunkRaw || !buf.reserve(raw_len)) {
        info = {};
        info.error_kind = raw_len > kMaxChunkRaw ? SessionError::invalid : SessionError::memory;
        error = raw_len > kMaxChunkRaw ? "chunk: the payload is over the limit" : "chunk: out of memory for the staging buffer";
        return false;
    }
    if (raw_len) std::memcpy(buf.payload(), raw, raw_len);
    return put_buffer(kind, buf, raw_len, shuffle_width, out, info, error, durable);
}

bool ChunkPool::get(ChunkKind kind, const ChunkRef& ref, void* dst, size_t dst_cap, std::string& error,
                    SessionError* kind_out) {
    auto fail = [&](SessionError k, const std::string& m) {
        if (kind_out) *kind_out = k;
        error = "chunk " + ref.key.hex() + ": " + m;
        return false;
    };
    if (ref.raw_len > kMaxChunkRaw || ref.stored_len > ref.raw_len) return fail(SessionError::invalid, "invalid reference");
    if (dst_cap < ref.raw_len) return fail(SessionError::invalid, "the destination is too small");
    Fd fd;
    struct stat st {};
    std::string why;
    bool refused = false;
    int eno = 0;
    if (!open_regular(p_->path_of(ref.key), fd, st, why, refused, &eno))
        return fail(refused || eno == ENOENT ? SessionError::invalid : errno_kind(eno), why);
    const uint64_t size = (uint64_t) st.st_size;
    if (size > kMaxChunkFile || size != chunk_file_size(ref.stored_len))
        return fail(SessionError::invalid, "file size " + std::to_string(size) + " does not match the reference");
    Aligned file = aligned_bytes((size_t) size);
    if (!file) return fail(SessionError::memory, "out of memory for the read buffer");
    bool direct = p_->o.direct && try_direct(fd.get());
    if (!pread_all(fd.get(), file.get(), (size_t) size, direct, why)) return fail(SessionError::io, "read error: " + why);
    if (!direct) ::posix_fadvise(fd.get(), 0, 0, POSIX_FADV_DONTNEED);
    // a file that was read in full and is wrong in itself (bit rot, a torn write): moved aside, so the next put of
    // these bytes writes a fresh chunk instead of hitting this one (a hit only looks at the header block)
    auto corrupt = [&](const std::string& m) {
        (void) injected(p_->o.fault, "chunk_quarantine");   // tests: a writer replaces the file between the read and the move
        p_->quarantine(ref.key, st);
        return fail(SessionError::invalid, m);
    };
    Hdr h;
    if (const char* bad = parse_header(file.get(), h)) return corrupt(bad);
    // The kind is not part of the key, so the same bytes written once as a K/V slab and once as a state slice are one
    // file: the kind is advisory (it tells a codec which shuffle width the writer used), never a reason to refuse.
    (void) kind;
    if (!(h.key == ref.key)) return corrupt("key mismatch");
    if (h.raw_len != ref.raw_len || h.stored_len != ref.stored_len)
        return fail(SessionError::invalid, "lengths do not match the manifest");
    // a successful read is a use: the chunk becomes the newest in the LRU order (memory and file mtime)
    auto used = [&]() {
        {
            std::lock_guard<std::mutex> g(p_->mu);
            auto it = p_->idx.find(ref.key);
            if (it != p_->idx.end()) {
                auto& e = it->second;
                if (e.refs == 0) { p_->ghost_del(ref.key, e); p_->ghost_add(ref.key, e, p_->tick()); }
                else e.seq = p_->tick();
            }
        }
        p_->stamp_file(p_->path_of(ref.key));
    };
    const uint8_t* payload = file.get() + kChunkAlign;
    if (h.codec == ChunkCodec::raw) {
        if (!(chunk_key(payload, h.raw_len) == ref.key)) return corrupt("content hash mismatch");
        if (h.raw_len) std::memcpy(dst, payload, h.raw_len);
        used();
        return true;
    }
    if (!codec_available(h.codec)) return fail(SessionError::invalid, "codec not available in this build");
    bool decoder_unavailable = false;
    if (!segmented_decode(payload, h.stored_len, dst, h.raw_len, &decoder_unavailable)) {
        // the decoder could not get memory: nothing is known about the file, so it stays where it is
        if (decoder_unavailable) return fail(SessionError::memory, "out of memory for the decoder");
        return corrupt("decode failed");
    }
    if (!(chunk_key(dst, h.raw_len) == ref.key)) return corrupt("content hash mismatch");
    used();
    return true;
}

bool ChunkPool::exists(const ChunkRef& ref) const {
    struct stat st {};
    if (::lstat(p_->path_of(ref.key).c_str(), &st) != 0 || !S_ISREG(st.st_mode)) return false;
    return (uint64_t) st.st_size == chunk_file_size(ref.stored_len);
}

void ChunkPool::ref(const ChunkRef* refs, size_t n) {
    auto& p = *p_;
    std::lock_guard<std::mutex> g(p.mu);
    for (size_t i = 0; i < n; ++i) {
        auto it = p.idx.find(refs[i].key);
        if (it == p.idx.end()) {
            // seen on the disk by a manifest scan after open() (another engine wrote it): index it (a manifest names it
            // now, so it is referenced from the start and never a ghost)
            Impl::Info info;
            struct stat st {};
            info.seq = p.tick();
            if (::lstat(p.path_of(refs[i].key).c_str(), &st) == 0 && S_ISREG(st.st_mode)) info.file_bytes = (uint64_t) st.st_size;
            it = p.idx.emplace(refs[i].key, info).first;
            p.bytes += info.file_bytes;
            ++it->second.refs;
            continue;
        }
        p.ref_inc(refs[i].key, it->second);
    }
}

void ChunkPool::unref(const ChunkRef* refs, size_t n) {
    auto& p = *p_;
    std::lock_guard<std::mutex> g(p.mu);
    for (size_t i = 0; i < n; ++i) {
        auto it = p.idx.find(refs[i].key);
        if (it == p.idx.end() || it->second.refs == 0) continue;
        p.ref_dec(refs[i].key, it->second);
        // a ghost: the newest one now (the order a restart and the other engines read back is the file's mtime)
        if (it->second.refs == 0) p.stamp_file(p.path_of(refs[i].key));
    }
}

struct ChunkLease::Impl {
    std::mutex mu;
    int fd = -1;
    std::string path;
    std::set<ChunkKey> seen;
};
ChunkLease::~ChunkLease() { release(); }
void ChunkLease::release() {
    if (!p_) return;
    std::lock_guard<std::mutex> g(p_->mu);
    if (p_->fd >= 0) {
        ::unlink(p_->path.c_str());   // first the name, then the lock
        ::close(p_->fd);
        p_->fd = -1;
    }
}
void ChunkLease::announce(const ChunkKey& k) {
    if (!p_) return;
    std::lock_guard<std::mutex> g(p_->mu);
    if (p_->fd < 0 || !p_->seen.insert(k).second) return;
    (void) ::write(p_->fd, k.b.data(), 16);   // O_APPEND; a failure only weakens the protection (the other checks remain)
}
void ChunkLease::announce(const ChunkRef* refs, size_t n) {
    for (size_t i = 0; i < n; ++i) announce(refs[i].key);
}

std::unique_ptr<ChunkLease> ChunkPool::begin_lease() {
    if (!p_->o.shared) return nullptr;
    for (int attempt = 0; attempt < 8; ++attempt) {
        const std::string tmp = p_->o.dir + "/.lease-tmp-" + hex64(random64());
        const int fd = ::open(tmp.c_str(), O_RDWR | O_CREAT | O_EXCL | O_CLOEXEC | O_APPEND, 0600);
        if (fd < 0) { if (errno == EEXIST) continue; break; }
        if (::flock(fd, LOCK_EX) != 0) { ::close(fd); ::unlink(tmp.c_str()); break; }
        const std::string fin = p_->o.dir + "/.lease-" + hex64(random64());
        if (::rename(tmp.c_str(), fin.c_str()) != 0) { ::close(fd); ::unlink(tmp.c_str()); break; }
        std::unique_ptr<ChunkLease> l(new ChunkLease());
        l->p_.reset(new ChunkLease::Impl());
        l->p_->fd = fd;
        l->p_->path = fin;
        return l;
    }
    return nullptr;   // no lease: the caller goes on (a failure here only weakens the protection against a deleter race)
}

size_t ChunkPool::evict_ghosts(uint64_t want_bytes, const std::function<bool(std::set<ChunkKey>&)>& named,
                               uint64_t* freed_out, bool* undecided) {
    auto& p = *p_;
    if (freed_out) *freed_out = 0;
    if (undecided) *undecided = false;
    if (want_bytes == 0) return 0;
    if (!p.o.shared) {
        std::lock_guard<std::mutex> g(p.mu);
        size_t removed = 0;
        uint64_t freed = 0;
        std::vector<std::pair<ChunkKey, uint64_t>> stuck;   // a file that cannot be unlinked stays a ghost, in its place
        while (freed < want_bytes && !p.ghost_lru.empty()) {
            const ChunkKey key = p.ghost_lru.begin()->second;
            auto it = p.idx.find(key);
            if (it == p.idx.end() || it->second.refs > 0) { p.ghost_lru.erase(p.ghost_lru.begin()); continue; }   // (cannot happen)
            if (::unlink(p.path_of(key).c_str()) != 0 && errno != ENOENT) {
                stuck.emplace_back(key, it->second.seq);
                p.ghost_del(key, it->second);
                continue;
            }
            freed += it->second.file_bytes;
            ++removed;
            p.drop(it);
        }
        for (const auto& st : stuck) {
            auto it = p.idx.find(st.first);
            if (it != p.idx.end()) p.ghost_add(st.first, it->second, st.second);
        }
        if (freed_out) *freed_out = freed;
        return removed;
    }
    // ---- shared.  Rounds: each takes the oldest ghosts not looked at yet, until enough is freed or none is left.
    size_t removed = 0;
    uint64_t freed = 0;
    std::set<ChunkKey> examined;
    for (int round = 0; round < 64 && freed < want_bytes; ++round) {
        struct Victim { ChunkKey key; uint64_t bytes; std::string aside; bool moved = false; };
        std::vector<Victim> victims;
        // S1: the leases FIRST, then the manifests (a writer publishes its manifest, then drops its lease: a lease gone
        // by now means its manifest is visible to the manifest read that follows)
        // lease, manifests, lease: a take announces (lease) and then removes its index, a write publishes its index and
        // then drops its lease - whichever order a racing owner works in, one of the three reads sees its chunk.
        // Read BEFORE the victims are picked, so that the chunks other engines' manifests name (usually the front of the
        // order in shared mode) are skipped for free instead of using up the rounds
        std::set<ChunkKey> prot;
        p.read_leases(prot);
        if (!named || !named(prot)) { if (undecided) *undecided = true; break; }
        p.read_leases(prot);
        {
            std::lock_guard<std::mutex> g(p.mu);
            uint64_t sum = 0;
            for (const auto& gk : p.ghost_lru) {
                if (sum >= want_bytes - freed) break;
                if (examined.count(gk.second) || prot.count(gk.second)) continue;
                auto it = p.idx.find(gk.second);
                if (it == p.idx.end() || it->second.refs > 0) continue;
                victims.push_back({gk.second, it->second.file_bytes, {}, false});
                sum += it->second.file_bytes;
            }
        }
        if (victims.empty()) break;
        for (const auto& v : victims) examined.insert(v.key);
        // the unprotected victims leave their names (atomic): a writer that looks now finds no file and writes it again
        for (auto& v : victims) {
            v.aside = p.shard_dir(v.key) + "/.del-" + hex64(random64());
            if (::rename(p.path_of(v.key).c_str(), v.aside.c_str()) == 0) v.moved = true;
        }
        // S2, once for the batch: a victim a lease or a manifest names now is put back
        std::set<ChunkKey> again;
        p.read_leases(again);
        const bool again_ok = named(again);   // false: cannot tell, put every one back
        if (!again_ok && undecided) *undecided = true;
        std::lock_guard<std::mutex> g(p.mu);
        for (auto& v : victims) {
            if (!v.moved) continue;
            auto it = p.idx.find(v.key);
            if (!again_ok || again.count(v.key) || (it != p.idx.end() && it->second.refs > 0)) {
                // link() never replaces: a writer that recreated the name wins, and the trashed copy is dropped
                (void) ::link(v.aside.c_str(), p.path_of(v.key).c_str());
                ::unlink(v.aside.c_str());
                continue;
            }
            ::unlink(v.aside.c_str());
            if (it != p.idx.end()) p.drop(it);   // (a local writer that found it missing forgot it already)
            freed += v.bytes;
            ++removed;
        }
    }
    if (freed_out) *freed_out = freed;
    return removed;
}

void ChunkPool::census() {
    auto& p = *p_;
    if (!p.o.shared) return;
    std::set<ChunkKey> present;
    std::error_code ec;
    for (const auto& shard : fs::directory_iterator(p.o.dir, ec)) {
        std::error_code e2;
        const std::string sname = shard.path().filename().string();
        if (sname.size() != 2 || !shard.is_directory(e2)) continue;
        for (const auto& f : fs::directory_iterator(shard.path(), e2)) {
            const std::string name = f.path().filename().string();
            ChunkKey k;
            if (name.size() != 36 || name.compare(32, 4, ".chk") != 0 || !key_from_hex(name.substr(0, 32), k) ||
                shard_of(k) != sname) continue;
            present.insert(k);
        }
    }
    if (ec) return;   // the listing failed: nothing is concluded from it
    std::lock_guard<std::mutex> g(p.mu);
    for (const auto& k : present) {
        if (p.idx.count(k)) continue;
        struct stat st {};
        if (::lstat(p.path_of(k).c_str(), &st) != 0 || !S_ISREG(st.st_mode)) continue;
        const uint64_t size = (uint64_t) st.st_size;
        if (size < kChunkAlign || size > kMaxChunkFile || size % kChunkAlign) continue;
        Impl::Info info;
        info.file_bytes = size;
        auto it = p.idx.emplace(k, info).first;
        p.bytes += size;
        p.ghost_add(k, it->second, mtime_ns(st));   // another engine's chunk: a ghost, in the order of its file's mtime
    }
    for (auto it = p.idx.begin(); it != p.idx.end();)
        if (it->second.refs == 0 && !present.count(it->first)) it = p.drop(it);
        else ++it;
}

bool ChunkPool::sync_touched_shards(std::string& error) {
    auto& p = *p_;
    // One sync at a time, held until its fsyncs are done: a second durable writer must not find the set emptied by a
    // first one that is still flushing (or that failed) and publish its manifest as if its shards were safe.
    std::lock_guard<std::mutex> serial(p.sync_mu);
    std::set<std::string> todo;
    {
        std::lock_guard<std::mutex> g(p.mu);
        todo.swap(p.touched);
    }
    bool ok = true;
    std::set<std::string> failed;
    for (const auto& s : todo) {
        int e = injected(p.o.fault, "shard_sync");
        if (!e) {
            Fd d(::open((s.empty() ? p.o.dir : p.o.dir + "/" + s).c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC));
            if (!d) e = errno;
            else if (::fsync(d.get()) != 0) e = errno;
        }
        if (e && e != EINVAL) {   // EINVAL: a filesystem that cannot flush a folder
            if (ok) error = std::string("flushing a chunk folder: ") + std::strerror(e);
            ok = false;
            failed.insert(s);
        }
    }
    if (!failed.empty()) {   // a retry (the chunks it hits are hits now) must flush them again
        std::lock_guard<std::mutex> g(p.mu);
        p.touched.insert(failed.begin(), failed.end());
    }
    return ok;
}

void ChunkPool::remove_all() {
    auto& p = *p_;
    std::lock_guard<std::mutex> g(p.mu);
    std::error_code ec;
    fs::remove_all(p.o.dir, ec);
    p.idx.clear();
    p.ghost_lru.clear();
    p.bytes = p.ghost_bytes = 0;
    p.touched.clear();
}
#endif

uint64_t ChunkPool::bytes() const { std::lock_guard<std::mutex> g(p_->mu); return p_->bytes; }
uint64_t ChunkPool::live_bytes() const {
    std::lock_guard<std::mutex> g(p_->mu);
    uint64_t n = 0;
    for (const auto& kv : p_->idx)
        if (kv.second.refs > 0) n += kv.second.file_bytes;
    return n;
}
uint64_t ChunkPool::ghost_bytes() const { std::lock_guard<std::mutex> g(p_->mu); return p_->ghost_bytes; }
size_t ChunkPool::ghost_count() const { std::lock_guard<std::mutex> g(p_->mu); return p_->ghost_lru.size(); }
std::vector<ChunkKey> ChunkPool::ghosts_oldest_first() const {
    std::lock_guard<std::mutex> g(p_->mu);
    std::vector<ChunkKey> v;
    for (const auto& gk : p_->ghost_lru) v.push_back(gk.second);
    return v;
}
size_t ChunkPool::count() const { std::lock_guard<std::mutex> g(p_->mu); return p_->idx.size(); }
uint32_t ChunkPool::refs_of(const ChunkKey& k) const {
    std::lock_guard<std::mutex> g(p_->mu);
    auto it = p_->idx.find(k);
    return it == p_->idx.end() ? 0 : it->second.refs;
}

// ------------------------------------------------------------------------------------------------ geometry and slabs

namespace {
struct LayerGeom {
    int64_t format = 0, cells = 0, heads = 0, head_dim = 0, page_size = 0, pooled_rows = 0, idx_dim = 0;
    std::array<uint64_t, 5> sizes{};
    std::array<uint8_t, 5> widths{0, 0, 2, 2, 4};   // shuffle element width per part: k_scale / v_scale fp16, pooled f32
};

// How K/V layers are cut into slabs.  Slab i holds cells [i*T, (i+1)*T) of every layer (clamped to the layer's cells)
// and pooled rows [i*T/4, (i+1)*T/4) of the rows that are not the spare one (the last), part-major: all K, all V,
// all k_scale, all v_scale, all pooled.
struct Plan {
    uint32_t T = 0;
    uint64_t max_cells = 0, n_slabs = 0, kv_total = 0;
    std::vector<LayerGeom> L;
    std::vector<uint64_t> unit;       // [layer*4 + part]: bytes per cell of parts 0..3
    std::vector<uint64_t> row;        // pooled row bytes per layer
    std::vector<uint64_t> data_rows;  // pooled rows in slabs (pooled_rows - 1; the spare row is inline)

    bool build(std::string& err) {
        auto fail = [&](const std::string& m) { err = "chunked session: " + m; return false; };
        if (T < 64 || T > 4096 || T % 4) return fail("chunk_tokens must be a multiple of 4 between 64 and 4096");
        unit.assign(L.size() * 4, 0);
        row.assign(L.size(), 0);
        data_rows.assign(L.size(), 0);
        max_cells = 0;
        kv_total = 0;
        for (size_t l = 0; l < L.size(); ++l) {
            const auto& g = L[l];
            if (g.cells < 0 || g.pooled_rows < 0) return fail("negative K/V extent");
            for (size_t part = 0; part < 4; ++part) {
                if (g.cells == 0) { if (g.sizes[part] != 0) return fail("K/V part has bytes but no cells"); continue; }
                if (g.sizes[part] % (uint64_t) g.cells) return fail("K/V part size is not a whole number of cells");
                unit[l * 4 + part] = g.sizes[part] / (uint64_t) g.cells;
            }
            if (g.pooled_rows == 0) {
                if (g.sizes[4] != 0) return fail("pooled part has bytes but no rows");
            } else {
                if (g.sizes[4] % (uint64_t) g.pooled_rows) return fail("pooled part size is not a whole number of rows");
                row[l] = g.sizes[4] / (uint64_t) g.pooled_rows;
                if (row[l] > kMaxRow) return fail("pooled row too large");
                data_rows[l] = (uint64_t) g.pooled_rows - 1;
            }
            for (uint8_t w : g.widths) if (w != 0 && w != 1 && w != 2 && w != 4 && w != 8) return fail("invalid shuffle width");
            max_cells = std::max<uint64_t>(max_cells, (uint64_t) g.cells);
            for (uint64_t s : g.sizes) kv_total = sat_add(kv_total, s);
        }
        if (kv_total == UINT64_MAX) return fail("K/V too large");
        n_slabs = (max_cells + T - 1) / T;
        if (n_slabs > UINT32_MAX) return fail("too many slabs");
        for (size_t l = 0; l < L.size(); ++l)
            if (data_rows[l] > n_slabs * (T / 4)) return fail("pooled rows are not covered by the cells' slabs");
        return true;
    }
    // [off, off+len) of part `part` of layer `l` that slab `i` carries
    void slice(size_t l, size_t part, uint64_t i, uint64_t& off, uint64_t& len) const {
        if (part < 4) {
            const uint64_t cells = (uint64_t) L[l].cells;
            const uint64_t c0 = std::min(i * T, cells), c1 = std::min((i + 1) * T, cells);
            off = c0 * unit[l * 4 + part];
            len = (c1 - c0) * unit[l * 4 + part];
        } else {
            const uint64_t r0 = std::min(i * (T / 4), data_rows[l]), r1 = std::min((i + 1) * (T / 4), data_rows[l]);
            off = r0 * row[l];
            len = (r1 - r0) * row[l];
        }
    }
    uint64_t slab_len(uint64_t i) const {
        uint64_t n = 0;
        for (size_t part = 0; part < 5; ++part)
            for (size_t l = 0; l < L.size(); ++l) { uint64_t o, c; slice(l, part, i, o, c); n += c; }
        return n;
    }
};

struct CkptMeta {
    uint64_t gdn_len = 0, inline_bytes = 0;
    uint32_t slice_len = kStateSlice, n_slices = 0;
    std::vector<ChunkRef> refs;
};

uint64_t inline_bytes_of(const ConversationCheckpoint& c) {
    return sat_add(sat_add(sat_mul(c.ids.size(), 4), sat_mul(c.imgs.size(), 16)),
                   sat_add(sat_add(c.ple.size(), c.tails.size()), sat_add(c.dead.size(), c.block_pos.size())));
}

// The pooled index's block: a pooled row covers this many tokens (the slab cut and the spare row are in these terms).
constexpr int64_t kIdxBlock = 4;

// The layer `g` of an image of `n` tokens as it is for the first `upto` tokens (0 < upto < n): the cells up to the page
// that holds token upto-1, the pooled rows up to the spare row of `upto` (rows [0, upto/4) are the prefix of the full
// image's rows, the spare row is not in the image: it comes from the checkpoint at `upto`).  Every part is linear in
// the cells (or rows), so the prefix's bytes are the first bytes of the full image's, part by part - the same
// transformation serves the prefix read and the write "as of a checkpoint" (live-skip).
bool prefix_layer(const LayerGeom& g, int64_t n, int64_t upto, LayerGeom& out, std::string& why) {
    if (upto < 1 || upto >= n) { why = "prefix length outside the image"; return false; }
    if (g.page_size <= 0 || g.cells <= 0) { why = "K/V layer without cells"; return false; }
    const int64_t page = g.page_size;
    if (upto > INT64_MAX - page) { why = "prefix length overflow"; return false; }
    const int64_t cells = (upto + page - 1) / page * page;
    if (cells > g.cells) { why = "prefix is beyond the layer's cells"; return false; }
    out = g;
    out.cells = cells;
    for (size_t part = 0; part < 4; ++part) {
        if (g.sizes[part] % (uint64_t) g.cells) { why = "K/V part size is not a whole number of cells"; return false; }
        out.sizes[part] = g.sizes[part] / (uint64_t) g.cells * (uint64_t) cells;
    }
    if (g.pooled_rows > 0) {
        if (g.pooled_rows != n / kIdxBlock + 1) { why = "pooled rows do not match the token count"; return false; }
        if (g.sizes[4] % (uint64_t) g.pooled_rows) { why = "pooled part size is not a whole number of rows"; return false; }
        out.pooled_rows = upto / kIdxBlock + 1;
        out.sizes[4] = g.sizes[4] / (uint64_t) g.pooled_rows * (uint64_t) out.pooled_rows;
    }
    return true;
}

bool has_stage_parts(const SavedConversation& s) {
    if (!s.live.stage_parts.empty()) return true;
    for (const auto& c : s.checkpoints) if (!c.stage_parts.empty()) return true;
    return false;
}

bool write_buffer_at(ConversationBuffer& b, uint64_t off, const uint8_t* src, uint64_t n) {
    return b.visit((size_t) off, (size_t) n, [&](uint8_t* p, size_t c, size_t at) {
        std::memcpy(p, src + (at - (size_t) off), c);
        return true;
    });
}

// ------------------------------------------------------------------------------------------------ manifest bytes

class Wr {
public:
    void raw(const void* p, size_t n) { const auto* s = static_cast<const uint8_t*>(p); b.insert(b.end(), s, s + n); }
    void u8(uint8_t v) { b.push_back(v); }
    void u32(uint32_t v) { raw(&v, 4); }
    void u64(uint64_t v) { raw(&v, 8); }
    void i64(int64_t v) { raw(&v, 8); }
    template<class T> void vec(const std::vector<T>& v) { u64(v.size()); raw(v.data(), v.size() * sizeof(T)); }
    void ref(const ChunkRef& r) { raw(r.key.b.data(), 16); u32(r.raw_len); u32(r.stored_len); }
    std::vector<uint8_t> b;
};

void put_checkpoint(Wr& w, const ConversationCheckpoint& c, const CkptMeta& m, bool zero_used = false) {
    w.vec(c.ids);
    w.u64(c.imgs.size());
    for (const auto& k : c.imgs) { w.i64(k.start); w.u64(k.hash); }
    w.vec(c.ple); w.vec(c.tails); w.vec(c.dead); w.vec(c.block_pos);
    w.u64(zero_used ? 0 : c.used);
    w.u64(m.gdn_len);
    w.u32(m.slice_len);
    w.u32(m.n_slices);
    for (const auto& r : m.refs) w.ref(r);
}

class Cur {
public:
    Cur(const uint8_t* p, size_t n, std::string& err, const SessionReadLimits& lim) : p_(p), left_(n), err_(err), lim_(lim) {}
    bool fail(const std::string& m) { if (err_.empty()) err_ = "chunked session: " + m; return false; }
    bool raw(void* d, size_t c) {
        if (c > left_) return fail("body ends early");
        if (c) std::memcpy(d, p_, c);
        p_ += c; left_ -= c;
        return true;
    }
    bool skip(uint64_t c) {
        if (c > left_) return fail("body ends early");
        p_ += c; left_ -= (size_t) c;
        return true;
    }
    bool u8(uint8_t& v) { return raw(&v, 1); }
    bool u32(uint32_t& v) { return raw(&v, 4); }
    bool u64(uint64_t& v) { return raw(&v, 8); }
    bool i64(int64_t& v) { return raw(&v, 8); }
    // `elem`: the fewest body bytes one element takes; `limit`: the most elements the caller accepts
    bool count(uint64_t& n, size_t elem, uint64_t limit = UINT64_MAX) {
        if (!u64(n)) return false;
        if (elem && n > left_ / elem) return fail("element count exceeds the body");
        if (n > limit) return fail("element count " + std::to_string(n) + " exceeds this engine's limit " + std::to_string(limit));
        return true;
    }
    template<class T> bool vec(std::vector<T>& v, bool skip_it, uint64_t limit, uint64_t& n) {
        if (!count(n, sizeof(T), limit)) return false;
        if (skip_it) return skip(n * sizeof(T));
        v.resize((size_t) n);
        return raw(v.data(), (size_t) n * sizeof(T));
    }
    bool ref(ChunkRef& r) {
        return raw(r.key.b.data(), 16) && u32(r.raw_len) && u32(r.stored_len);
    }
    size_t left() const { return left_; }
    const SessionReadLimits& limits() const { return lim_; }
private:
    const uint8_t* p_;
    size_t left_;
    std::string& err_;
    const SessionReadLimits& lim_;
};

struct Parsed {
    SavedConversation s;                         // geometry, identity, checkpoints' inline state; K/V empty
    Plan plan;
    std::vector<std::vector<uint8_t>> spare;     // per layer: the spare pooled row
    std::vector<ChunkRef> slabs;
    std::vector<CkptMeta> cps;                   // [0] = live, then the checkpoints
    uint64_t raw_total = 0;
};

bool get_checkpoint(Cur& c, ConversationCheckpoint& cp, CkptMeta& m, bool skip_inline) {
    const auto& lim = c.limits();
    uint64_t n = 0, bytes = 0;
    if (!c.vec(cp.ids, skip_inline, lim.max_tokens, n)) return false;
    bytes = sat_mul(n, 4);
    if (!c.count(n, 16, lim.max_tokens)) return false;
    bytes = sat_add(bytes, sat_mul(n, 16));
    if (skip_inline) { if (!c.skip(n * 16)) return false; }
    else {
        cp.imgs.resize((size_t) n);
        for (auto& k : cp.imgs) if (!c.i64(k.start) || !c.u64(k.hash)) return false;
    }
    std::vector<uint8_t>* arrays[4] = {&cp.ple, &cp.tails, &cp.dead, &cp.block_pos};
    for (int i = 0; i < 4; ++i) {
        if (!c.vec(*arrays[i], skip_inline, lim.max_state_bytes[i + 1], n)) return false;
        bytes = sat_add(bytes, n);
    }
    uint64_t gdn_len = 0;
    if (!c.u64(cp.used) || !c.u64(gdn_len) || !c.u32(m.slice_len) || !c.u32(m.n_slices)) return false;
    if (gdn_len > lim.max_state_bytes[0]) return c.fail("gdn length exceeds this engine's limit");
    if (gdn_len > (uint64_t) std::numeric_limits<size_t>::max()) return c.fail("gdn length too large");
    if (m.slice_len == 0 || m.slice_len % 4 || m.slice_len > kMaxChunkRaw) return c.fail("invalid state slice length");
    const uint64_t want = (gdn_len + m.slice_len - 1) / m.slice_len;
    if (m.n_slices != want) return c.fail("state slice count does not match the state length");
    if (want > c.left() / kRefBytes) return c.fail("state slice count exceeds the body");
    m.gdn_len = gdn_len;
    m.inline_bytes = bytes;
    m.refs.resize((size_t) want);
    for (uint64_t j = 0; j < want; ++j) {
        auto& r = m.refs[(size_t) j];
        if (!c.ref(r)) return false;
        const uint64_t expect = std::min<uint64_t>(m.slice_len, gdn_len - j * m.slice_len);
        if (r.raw_len != expect) return c.fail("state slice length does not match");
        if (r.stored_len > r.raw_len) return c.fail("stored length over the raw length");
    }
    return true;
}

bool parse_body(const uint8_t* body, size_t len, uint32_t T, const SessionReadLimits& lim, bool skip_inline,
                Parsed& out, std::string& err) {
    Cur c(body, len, err, lim);
    SavedConversation& s = out.s;
    for (auto& g : s.geometry) if (!c.i64(g)) return false;
    if (lim.geometry && *lim.geometry != s.geometry) return c.fail("saved with another model geometry than this runtime's");
    uint64_t flags = 0;
    uint32_t n_layers = 0;
    if (!c.i64(s.layer_lo) || !c.i64(s.layer_hi) || !c.u64(flags) || !c.u32(n_layers)) return false;
    if (lim.layer_range && (lim.layer_range->first != s.layer_lo || lim.layer_range->second != s.layer_hi))
        return c.fail("saved with another layer range than this runtime's");
    const uint64_t rope_code = flags >> 1;
    if (rope_code != 0 && (rope_code < 1048576 || rope_code > (1ull << 40))) return c.fail("invalid cvec flag");
    s.cvec = (flags & 1) == 1;
    s.rope = rope_code == 0 ? 1.0 : (double) rope_code / 1048576.0;
    if (n_layers > lim.max_kv_layers) return c.fail("K/V layer count exceeds this engine's limit");
    if (n_layers > c.left() / kLayerMin) return c.fail("K/V layer count exceeds the body");
    out.plan.T = T;
    out.plan.L.resize(n_layers);
    s.kv.resize(n_layers);
    out.spare.resize(n_layers);
    for (uint32_t l = 0; l < n_layers; ++l) {
        auto& g = out.plan.L[l];
        if (!c.i64(g.format) || !c.i64(g.cells) || !c.i64(g.heads) || !c.i64(g.head_dim) || !c.i64(g.page_size) ||
            !c.i64(g.pooled_rows) || !c.i64(g.idx_dim)) return false;
        if (g.format < INT32_MIN || g.format > INT32_MAX) return c.fail("invalid K/V format");
        for (auto& sz : g.sizes) if (!c.u64(sz)) return false;
        for (auto& w : g.widths) if (!c.u8(w)) return false;
        for (size_t part = 0; part < 5; ++part)
            if (l < lim.max_kv_bytes.size() && g.sizes[part] > lim.max_kv_bytes[l][part])
                return c.fail("K/V part exceeds this engine's limit");
        if (g.pooled_rows > 0) {
            if (g.sizes[4] % (uint64_t) g.pooled_rows) return c.fail("pooled part size is not a whole number of rows");
            const uint64_t row = g.sizes[4] / (uint64_t) g.pooled_rows;
            if (row > kMaxRow) return c.fail("pooled row too large");
            out.spare[l].resize((size_t) row);
            if (!c.raw(out.spare[l].data(), (size_t) row)) return false;
        }
        auto& k = s.kv[l];
        k.format = (int) g.format; k.cells = g.cells; k.heads = g.heads; k.head_dim = g.head_dim;
        k.page_size = g.page_size; k.pooled_rows = g.pooled_rows; k.idx_dim = g.idx_dim;
    }
    if (!out.plan.build(err)) return false;
    uint32_t n_slabs = 0;
    if (!c.u32(n_slabs)) return false;
    if (n_slabs != out.plan.n_slabs) return c.fail("slab count does not match the K/V extent");
    if (n_slabs > c.left() / kRefBytes) return c.fail("slab count exceeds the body");
    out.slabs.resize(n_slabs);
    for (uint32_t i = 0; i < n_slabs; ++i) {
        auto& r = out.slabs[i];
        if (!c.ref(r)) return false;
        if (r.raw_len != out.plan.slab_len(i) || r.raw_len > kMaxChunkRaw) return c.fail("slab length does not match the geometry");
        if (r.stored_len > r.raw_len) return c.fail("stored length over the raw length");
    }
    out.cps.resize(1);
    if (!get_checkpoint(c, s.live, out.cps[0], skip_inline)) return false;
    uint64_t n = 0;
    if (!c.count(n, kCheckpointMin, lim.max_checkpoints)) return false;
    s.checkpoints.resize((size_t) n);
    out.cps.resize((size_t) n + 1);
    for (size_t i = 0; i < s.checkpoints.size(); ++i)
        if (!get_checkpoint(c, s.checkpoints[i], out.cps[i + 1], skip_inline)) return false;
    if (c.left()) return c.fail("the body has " + std::to_string(c.left()) + " unparsed bytes");
    out.raw_total = out.plan.kv_total;
    for (const auto& m : out.cps) out.raw_total = sat_add(out.raw_total, sat_add(m.gdn_len, m.inline_bytes));
    return true;
}

std::vector<ChunkRef> all_refs(const Parsed& p) {
    std::vector<ChunkRef> refs = p.slabs;
    for (const auto& m : p.cps) refs.insert(refs.end(), m.refs.begin(), m.refs.end());
    return refs;
}

// header v2, little-endian: magic[8] version:u32 header_size:u32 model:u64 config:u64 body_len:u64 chunk_tokens:u32
// flags:u32 raw_total:u64 header_hash:u64 (session_hash64 of bytes 0..55, seed 0)
void manifest_header(uint8_t* h, const SessionFileIdentity& id, uint64_t body_len, uint32_t T, uint64_t raw_total) {
    std::memset(h, 0, kHeader);
    std::memcpy(h, kManMagic, 8);
    put_le<uint32_t>(h + 8, kManVersion);
    put_le<uint32_t>(h + 12, (uint32_t) kHeader);
    put_le<uint64_t>(h + 16, id.model);
    put_le<uint64_t>(h + 24, id.config);
    put_le<uint64_t>(h + 32, body_len);
    put_le<uint32_t>(h + 40, T);
    put_le<uint32_t>(h + 44, 0);
    put_le<uint64_t>(h + 48, raw_total);
    put_le<uint64_t>(h + 56, session_hash64(h, 56, 0));
}

std::string tmp_leaf(const std::string& leaf) {
    size_t keep = std::min<size_t>(leaf.size(), 64);
    while (keep > 0 && keep < leaf.size() && (uint8_t(leaf[keep]) & 0xC0) == 0x80) --keep;
    return "." + leaf.substr(0, keep) + "." + hex64(random64()) + ".tmp";
}

#ifndef _WIN32
// temp + rename over the final name, in the destination's own folder; durable: flush the file, then the folder
bool publish_manifest(const std::string& path, const std::vector<uint8_t>& bytes, bool durable,
                      const std::function<int(const char*)>& fault, std::string& error, SessionStatus& st) {
    const size_t cut = path.find_last_of('/');
    const std::string dir = cut == std::string::npos ? "." : path.substr(0, cut == 0 ? 1 : cut);
    const std::string leaf = cut == std::string::npos ? path : path.substr(cut + 1);
    if (leaf.empty() || leaf == "." || leaf == "..") { error = "chunked session: " + path + " names no file"; st.error = SessionError::invalid; return false; }
    Fd dirfd(::open(dir.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC));
    if (!dirfd) { st.error = errno_kind(errno); error = "chunked session: cannot open the directory of " + path + ": " + std::strerror(errno); return false; }
    std::string tmp;
    Fd fd;
    for (int attempt = 0; attempt < 8 && !fd; ++attempt) {
        tmp = tmp_leaf(leaf);
        fd = Fd(::openat(dirfd.get(), tmp.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC, 0600));
        if (!fd && errno != EEXIST) break;
    }
    if (!fd) { st.error = errno_kind(errno); error = "chunked session: cannot create a temporary file beside " + path + ": " + std::strerror(errno); return false; }
    bool renamed = false;
    auto fail = [&](int e, const std::string& what) {
        if (!renamed) ::unlinkat(dirfd.get(), tmp.c_str(), 0);
        st.error = errno_kind(e);
        error = "chunked session: writing " + path + " failed: " + what + std::strerror(e);
        return false;
    };
    int e = injected(fault, "write");
    for (size_t off = 0; !e && off < bytes.size();) {
        const ssize_t w = ::write(fd.get(), bytes.data() + off, bytes.size() - off);
        if (w < 0 && errno == EINTR) continue;
        if (w <= 0) { e = w < 0 ? errno : EIO; break; }
        off += (size_t) w;
    }
    if (e) return fail(e, "");
    if (durable) {
        e = injected(fault, "file_flush");
        if (!e && ::fsync(fd.get()) != 0) e = errno;
        if (e) return fail(e, "");
    }
    if (!fd.close()) return fail(errno, "");
    e = injected(fault, "rename");
    if (!e && ::renameat(dirfd.get(), tmp.c_str(), dirfd.get(), leaf.c_str()) != 0) e = errno;
    if (e) return fail(e, "");
    renamed = true;
    st.published = true;
    if (!durable) return true;
    e = injected(fault, "dir_flush");
    if (!e && ::fsync(dirfd.get()) != 0) e = errno;
    if (e == EINVAL) { st.dir_flush_unsupported = true; return true; }
    if (e) {
        st.error = errno_kind(e);
        error = std::string("chunked session: writing ") + path + " failed: flushing the directory: " + std::strerror(e) +
                " (the new file has already replaced the old one; its folder entry may not survive a power loss)";
        return false;
    }
    return true;
}
#endif

using Clock = std::chrono::steady_clock;
double ms_since(Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); }
} // namespace

// ------------------------------------------------------------------------------------------------ write

namespace {
#ifndef _WIN32
bool write_impl(ChunkPool& pool, const std::string& path, const SavedConversation& image,
                const std::vector<SessionKvSource>* sources, const SessionFileIdentity& id,
                const ChunkedWriteOptions& opt, std::vector<ChunkRef>& refs_out, ChunkedWriteStats& stats,
                std::string& error, SessionStatus& st) {
    st = {};
    st.error = SessionError::invalid;   // refusals before any disk access
    stats = {};
    refs_out.clear();
    error.clear();
    if (has_stage_parts(image)) { error = "chunked session: layer-split state cannot be saved"; return false; }
    if (sources && !image.kv.empty()) { error = "chunked session: K/V given both in the image and as sources"; return false; }
    Plan plan;
    plan.T = opt.chunk_tokens;
    if (sources) {
        for (const auto& k : *sources) {
            LayerGeom g;
            g.format = k.format; g.cells = k.cells; g.heads = k.heads; g.head_dim = k.head_dim;
            g.page_size = k.page_size; g.pooled_rows = k.pooled_rows; g.idx_dim = k.idx_dim;
            for (size_t i = 0; i < 5; ++i) g.sizes[i] = k.sizes[i];
            plan.L.push_back(g);
        }
    } else {
        for (const auto& k : image.kv) {
            LayerGeom g;
            g.format = k.format; g.cells = k.cells; g.heads = k.heads; g.head_dim = k.head_dim;
            g.page_size = k.page_size; g.pooled_rows = k.pooled_rows; g.idx_dim = k.idx_dim;
            g.sizes = {k.k.size(), k.v.size(), k.k_scale.size(), k.v_scale.size(), k.pooled.size()};
            plan.L.push_back(g);
        }
    }
    // "as of a checkpoint" (live-skip): the image written is the one of `as_of_tokens` tokens - the checkpoint of exactly
    // that length is the live state, the shallower checkpoints stay, the K/V is cut to those tokens (a prefix of every
    // part, so the source is read from offset 0 as before) and the spare pooled rows come from the checkpoint's `dead`
    std::vector<const ConversationCheckpoint*> cps;
    bool as_of = false;
    if (opt.as_of_tokens > 0) {
        const int64_t n = (int64_t) image.live.ids.size();
        const ConversationCheckpoint* at = nullptr;
        for (const auto& c : image.checkpoints) if ((int64_t) c.ids.size() == opt.as_of_tokens) at = &c;
        if (at == nullptr || opt.as_of_tokens >= n ||
            !std::equal(at->ids.begin(), at->ids.end(), image.live.ids.begin())) {
            error = "chunked session: no checkpoint of " + std::to_string(opt.as_of_tokens) + " tokens leads the live state";
            return false;
        }
        for (auto& g : plan.L) {
            LayerGeom cut;
            std::string why;
            if (!prefix_layer(g, n, opt.as_of_tokens, cut, why)) { error = "chunked session: " + why; return false; }
            g = cut;
        }
        as_of = true;
        cps.push_back(at);
        for (const auto& c : image.checkpoints) if ((int64_t) c.ids.size() <= opt.as_of_tokens) cps.push_back(&c);
    } else {
        cps.push_back(&image.live);
        for (const auto& c : image.checkpoints) cps.push_back(&c);
    }
    if (!plan.build(error)) return false;
    for (uint64_t i = 0; i < plan.n_slabs; ++i)
        if (plan.slab_len(i) > kMaxChunkRaw) { error = "chunked session: a slab is over the chunk limit; use fewer chunk_tokens"; return false; }
    std::vector<CkptMeta> metas(cps.size());
    uint64_t raw_total = plan.kv_total, chunk_raw_total = plan.kv_total;
    for (size_t l = 0; l < plan.L.size(); ++l)
        if (plan.L[l].pooled_rows > 0) chunk_raw_total -= plan.row[l];   // the spare rows ride in the manifest
    for (size_t i = 0; i < cps.size(); ++i) {
        metas[i].gdn_len = cps[i]->gdn.size();
        metas[i].inline_bytes = inline_bytes_of(*cps[i]);
        metas[i].slice_len = kStateSlice;
        metas[i].n_slices = (uint32_t) ((metas[i].gdn_len + kStateSlice - 1) / kStateSlice);
        raw_total = sat_add(raw_total, sat_add(metas[i].gdn_len, metas[i].inline_bytes));
        chunk_raw_total = sat_add(chunk_raw_total, metas[i].gdn_len);
        if (metas[i].gdn_len > UINT32_MAX * uint64_t(kStateSlice)) { error = "chunked session: the state is too large"; return false; }
    }
    if (raw_total == UINT64_MAX || raw_total > (uint64_t) std::numeric_limits<size_t>::max()) {
        error = "chunked session: the state is too large for this build";
        return false;
    }
    st.error = SessionError::io;

    std::vector<ChunkRef> refs;
    bool ok = false;
    // every exit that is not a success gives the references back
    struct Release {
        ChunkPool& pool; std::vector<ChunkRef>& refs; bool& ok;
        ~Release() { if (!ok && !refs.empty()) pool.unref(refs.data(), refs.size()); }
    } release{pool, refs, ok};

    // The chunk pipeline.  The calling thread gathers each chunk's raw bytes (a K/V source reads the device, which stays
    // on this thread) into a staging buffer; with workers > 1 the hash, the codec and the disk write of a full buffer run
    // on worker threads, at most workers + 1 buffers in flight (the gathering waits for a free one).  Results land at
    // the chunk's index, so the manifest order does not depend on who finished first.
    size_t n_chunks = (size_t) plan.n_slabs;
    for (const auto& m : metas) n_chunks += m.n_slices;
    std::vector<ChunkRef> results(n_chunks);
    std::vector<char> have(n_chunks, 0);
    int n_workers = std::clamp(opt.workers, 1, 8);
    struct Job {
        ChunkBuffer buf;
        size_t index = 0, raw_len = 0;
        ChunkKind kind = ChunkKind::kv_slab;
        uint8_t width = 0;
        std::vector<CodecSegment> segs;
    };
    std::mutex mu;
    std::mutex cancel_mu;   // the caller's cancel hook is asked from the workers too: never two at once
    auto cancelled = [&]() -> bool {
        if (!opt.cancel) return false;
        std::lock_guard<std::mutex> g(cancel_mu);
        return opt.cancel();
    };
    std::condition_variable cv;
    std::deque<Job> queue;
    std::vector<ChunkBuffer> spare_bufs;
    int made_bufs = 0, busy = 0;
    std::atomic<bool> failed{false};
    bool stopping = false;
    SessionError fail_kind = SessionError::io;
    std::string fail_msg;
    uint64_t done = 0;
    if (opt.progress) opt.progress(0, chunk_raw_total);

    auto note_failure = [&](SessionError k, const std::string& m) {   // first failure wins
        std::lock_guard<std::mutex> g(mu);
        if (!failed) { failed = true; fail_kind = k; fail_msg = m; }
        cv.notify_all();
    };
    auto run_job = [&](Job& j) {
      try {
        // (inline, the gathering has just asked; a worker asks again before the hash and the write)
        if (n_workers > 1 && cancelled()) { note_failure(SessionError::io, "chunked session: cancelled"); return; }
        ChunkRef ref;
        ChunkPutInfo info;
        const auto t0 = Clock::now();
        std::string why;
        const bool put_ok = pool.put_buffer(j.kind, j.buf, j.raw_len, j.width, ref, info, why, opt.durable,
                                            j.segs.empty() ? nullptr : &j.segs, opt.lease);
        const double took = ms_since(t0);
        if (!put_ok) { note_failure(info.error_kind, "chunked session: " + why); return; }
        std::lock_guard<std::mutex> g(mu);
        results[j.index] = ref;
        have[j.index] = 1;
        ++stats.chunks;
        stats.raw_bytes += j.raw_len;
        stats.io_ms += took;
        stats.encode_ms += info.encode_ms;
        if (info.hit) { ++stats.hits; stats.hit_raw_bytes += j.raw_len; } else { ++stats.misses; stats.new_file_bytes += info.new_file_bytes; }
        if (info.direct_io) stats.direct_io = true;
        done += j.raw_len;
        if (opt.progress) opt.progress(done, chunk_raw_total);   // (under the lock: the callback is never concurrent)
      } catch (const std::bad_alloc&) {
        note_failure(SessionError::memory, "chunked session: out of memory in the chunk pipeline");
      } catch (const std::exception& e) {
        note_failure(SessionError::io, std::string("chunked session: ") + e.what());
      }
    };
    std::vector<std::thread> threads;
    // stops the workers and moves the references of the chunks done so far into `refs`, in manifest order (once); every
    // exit path runs it, so a failure's Release below gives all of them back
    bool collected = false;
    auto collect = [&] {
        {
            std::lock_guard<std::mutex> g(mu);
            stopping = true;
            cv.notify_all();
        }
        for (auto& t : threads) t.join();
        threads.clear();
        if (collected) return;
        collected = true;
        for (size_t i = 0; i < have.size(); ++i) if (have[i]) refs.push_back(results[i]);
    };
    struct Joiner { std::function<void()> f; ~Joiner() { f(); } } joiner{collect};
    if (n_workers > 1) {
        for (int w = 0; w < n_workers; ++w) {
          try {
            threads.emplace_back([&] {
                // off the host thread's core (an idle save runs on the host thread, whose one CPU a new thread inherits)
                strata::aux_cpus::spread_current_thread();
#if defined(__linux__)
                if (opt.low_priority) (void) ::setpriority(PRIO_PROCESS, (id_t) ::syscall(SYS_gettid), 5);   // a background write yields to the engine
#endif
                for (;;) {
                    Job j;
                    {
                        std::unique_lock<std::mutex> lk(mu);
                        cv.wait(lk, [&] { return stopping || !queue.empty(); });
                        if (queue.empty()) return;   // stopping and drained
                        j = std::move(queue.front());
                        queue.pop_front();
                        ++busy;
                    }
                    if (!failed) run_job(j);   // (a read of `failed` racing a failure only costs one more chunk)
                    {
                        std::lock_guard<std::mutex> g(mu);
                        spare_bufs.push_back(std::move(j.buf));
                        --busy;
                        cv.notify_all();
                    }
                }
            });
          } catch (const std::system_error&) { break; }   // no more threads: go on with the ones there are
        }
        n_workers = std::max<int>(1, (int) threads.size());
        if (threads.empty()) n_workers = 1;   // (inline: the calling thread does it all)
    }
    size_t next_index = 0;
    // gathers one chunk (fill on this thread), then hands it to the pipeline; false: the write is given up
    auto emit = [&](ChunkKind kind, size_t raw_len, uint8_t width, std::vector<CodecSegment>* segs, const auto& fill) -> bool {
        if (cancelled()) { st.error = SessionError::io; error = "chunked session: cancelled"; return false; }
        Job j;
        {
            std::unique_lock<std::mutex> lk(mu);
            if (n_workers > 1) {
                cv.wait(lk, [&] { return failed || !spare_bufs.empty() || made_bufs < n_workers + 1; });
            }
            if (failed) { st.error = fail_kind; error = fail_msg; return false; }
            if (!spare_bufs.empty()) { j.buf = std::move(spare_bufs.back()); spare_bufs.pop_back(); }
            else ++made_bufs;
        }
        if (!j.buf.reserve(raw_len)) { st.error = SessionError::memory; error = "chunked session: out of memory for the chunk buffer"; return false; }
        if (!fill(j.buf.payload())) return false;
        j.index = next_index++;
        j.raw_len = raw_len;
        j.kind = kind;
        j.width = width;
        if (segs) j.segs = std::move(*segs);
        if (n_workers > 1) {
            std::lock_guard<std::mutex> g(mu);
            queue.push_back(std::move(j));
            cv.notify_all();
            return true;
        }
        run_job(j);
        std::lock_guard<std::mutex> g(mu);
        if (failed) { st.error = fail_kind; error = fail_msg; return false; }
        return true;
    };
    auto finish = [&]() -> bool {
        collect();
        if (failed) { st.error = fail_kind; error = fail_msg; return false; }
        return true;
    };

    std::string read_error;
    for (uint64_t i = 0; i < plan.n_slabs; ++i) {
        const size_t len = (size_t) plan.slab_len(i);
        std::vector<CodecSegment> segs;
        const bool good = emit(ChunkKind::kv_slab, len, 0, &segs, [&](uint8_t* dst) {
            size_t at = 0;
            for (size_t part = 0; part < 5; ++part)
                for (size_t l = 0; l < plan.L.size(); ++l) {
                    uint64_t off, n;
                    plan.slice(l, part, i, off, n);
                    if (!n) continue;
                    bool got;
                    if (sources) {
                        const auto& src = (*sources)[l];
                        got = src.read && src.read(part, (size_t) off, dst + at, (size_t) n);
                        if (!got && src.error) read_error = src.error();
                    } else {
                        const auto& k = image.kv[l];
                        const ConversationBuffer* b[5] = {&k.k, &k.v, &k.k_scale, &k.v_scale, &k.pooled};
                        got = b[part]->read(dst + at, (size_t) off, (size_t) n);
                    }
                    if (!got) {
                        st.error = SessionError::io;
                        error = "chunked session: K/V source read" + (read_error.empty() ? std::string() : ": " + read_error);
                        return false;
                    }
                    // one codec segment per part (the layers' bytes of a part are adjacent and of one width)
                    const uint8_t w = plan.L[l].widths[part];
                    if (!segs.empty() && segs.back().kind == part && segs.back().width == w && segs.back().len + n <= UINT32_MAX)
                        segs.back().len += (uint32_t) n;
                    else
                        segs.push_back({(uint32_t) n, w, (uint8_t) part});
                    at += (size_t) n;
                }
            return true;
        });
        if (!good) { collect(); return false; }
    }
    for (size_t i = 0; i < cps.size(); ++i) {
        const auto& gdn = cps[i]->gdn;
        for (uint32_t j = 0; j < metas[i].n_slices; ++j) {
            const size_t off = (size_t) j * kStateSlice, n = std::min<size_t>(kStateSlice, gdn.size() - off);
            if (!emit(ChunkKind::state_slice, n, 4, nullptr, [&](uint8_t* dst) { std::memcpy(dst, gdn.data() + off, n); return true; })) {
                collect();
                return false;
            }
        }
    }
    if (!finish()) return false;
    if (refs.size() != n_chunks) { st.error = SessionError::io; error = "chunked session: a chunk is missing from the pipeline"; return false; }
    // slice refs back into the checkpoints' metas, in the order they were emitted
    size_t at = plan.n_slabs;
    std::vector<ChunkRef> slabs(refs.begin(), refs.begin() + (std::ptrdiff_t) plan.n_slabs);
    for (auto& m : metas) { m.refs.assign(refs.begin() + (std::ptrdiff_t) at, refs.begin() + (std::ptrdiff_t) (at + m.n_slices)); at += m.n_slices; }

    // spare pooled rows
    std::vector<std::vector<uint8_t>> spare(plan.L.size());
    size_t n_index_layers = 0, index_layer = 0;
    for (const auto& g : plan.L) if (g.pooled_rows > 0) ++n_index_layers;
    for (size_t l = 0; l < plan.L.size(); ++l) {
        if (plan.L[l].pooled_rows == 0) continue;
        spare[l].resize((size_t) plan.row[l]);
        const uint64_t off = plan.data_rows[l] * plan.row[l];
        bool got = true;
        if (as_of) {   // the checkpoint's own row (what its restore writes there)
            const auto& dead = cps[0]->dead;
            if (plan.row[l] && (n_index_layers == 0 || dead.size() != n_index_layers * plan.row[l])) {
                error = "chunked session: the checkpoint's indexer rows do not fit the layers";
                return false;
            }
            if (plan.row[l]) std::memcpy(spare[l].data(), dead.data() + index_layer * plan.row[l], (size_t) plan.row[l]);
            ++index_layer;
        } else if (plan.row[l]) {
            if (sources) got = (*sources)[l].read && (*sources)[l].read(4, (size_t) off, spare[l].data(), spare[l].size());
            else got = image.kv[l].pooled.read(spare[l].data(), (size_t) off, spare[l].size());
        }
        if (!got) { error = "chunked session: K/V source read of the spare row"; return false; }
    }

    // the body
    Wr w;
    for (int64_t g : image.geometry) w.i64(g);
    w.i64(image.layer_lo); w.i64(image.layer_hi);
    const uint64_t rope_code = image.rope == 1.0 ? 0 : (uint64_t) std::llround(image.rope * 1048576.0);
    w.u64((image.cvec ? 1 : 0) | (rope_code << 1));
    w.u32((uint32_t) plan.L.size());
    for (size_t l = 0; l < plan.L.size(); ++l) {
        const auto& g = plan.L[l];
        for (int64_t v : {g.format, g.cells, g.heads, g.head_dim, g.page_size, g.pooled_rows, g.idx_dim}) w.i64(v);
        for (uint64_t v : g.sizes) w.u64(v);
        for (uint8_t v : g.widths) w.u8(v);
        w.raw(spare[l].data(), spare[l].size());
    }
    w.u32((uint32_t) slabs.size());
    for (const auto& r : slabs) w.ref(r);
    put_checkpoint(w, *cps[0], metas[0], as_of);   // (as of a checkpoint: its copy as the live state has no use count, as in the prefix read)
    w.u64(cps.size() - 1);
    for (size_t i = 1; i < cps.size(); ++i) put_checkpoint(w, *cps[i], metas[i]);

    std::vector<uint8_t> file(kHeader);
    manifest_header(file.data(), id, w.b.size(), plan.T, raw_total);
    file.insert(file.end(), w.b.begin(), w.b.end());
    const uint64_t body_hash = session_hash64(w.b.data(), w.b.size(), 0);
    const size_t tail = file.size();
    file.resize(tail + kTrailer);
    std::memcpy(file.data() + tail, &body_hash, 8);
    std::memcpy(file.data() + tail + 8, kManEnd, 8);

    if (opt.durable) {
        if (opt.phase) opt.phase("flush", stats.new_file_bytes);
        std::string why;
        if (!pool.sync_touched_shards(why)) { st.error = SessionError::io; error = "chunked session: " + why; return false; }
    }
    if (opt.before_publish) {   // shared folders: the chunks hit may have been swept since they were looked at
        std::string why;
        if (!opt.before_publish(refs, why)) { st.error = SessionError::io; error = "chunked session: " + why; return false; }
    }
    if (opt.phase) opt.phase("publish", file.size());
    std::string why;
    if (!publish_manifest(path, file, opt.durable, opt.fault, why, st)) {
        error = why;
        return false;
    }
    st.error = SessionError::none;
    stats.manifest_bytes = file.size();
    refs_out = refs;
    ok = true;
    return true;
}
#endif
} // namespace

bool chunked_write(ChunkPool& pool, const std::string& path, const SavedConversation& image,
                   const std::vector<SessionKvSource>* sources, const SessionFileIdentity& id,
                   const ChunkedWriteOptions& opt, std::vector<ChunkRef>& refs_out, ChunkedWriteStats& stats,
                   std::string& error, SessionStatus* status) {
    SessionStatus st;
    bool ok = false;
#ifdef _WIN32
    error = "the chunked disk format is POSIX only";
    st.error = SessionError::invalid;
#else
    try {
        ok = write_impl(pool, path, image, sources, id, opt, refs_out, stats, error, st);
    } catch (const std::bad_alloc&) {
        error = "chunked session: out of memory while writing " + path;
        st.error = SessionError::memory;
        ok = false;
    }
#endif
    if (status) *status = st;
    return ok;
}

// ------------------------------------------------------------------------------------------------ read

uint64_t chunked_read_peak_bytes(uint64_t raw_total, int workers, uint32_t chunk_tokens, uint64_t max_chunk_raw) {
    // the largest chunk: the manifest's own when it is known, else a slab of 16 KiB a token (it covers 12.4 KB)
    const uint64_t slab = max_chunk_raw ? std::min<uint64_t>(max_chunk_raw, kMaxChunkRaw)
                                        : std::min<uint64_t>(sat_mul(chunk_tokens, 16384), kMaxChunkRaw);
    const uint64_t buffer = kChunkAlign + std::max<uint64_t>(slab, max_chunk_raw ? 1 : kStateSlice);
    // a worker holds three at once: the aligned file buffer (ChunkPool::get), the caller's scratch for the raw
    // bytes, and the temporary of a shuffled segment's decode (zstd_unframe)
    const uint64_t per_worker = sat_mul(3, buffer);
    // with more than one worker the calling thread drains tasks too (chunked_read), so one more set is live at once
    const uint64_t holders = workers > 1 ? (uint64_t) workers + 1 : 1;
    const uint64_t segments = raw_total / ConversationBuffer::segment_bytes + 1;
    return sat_add(sat_add(sat_add(raw_total, sat_mul(holders, per_worker)), sat_mul(segments, 4096)),
                   uint64_t(1) << 20);
}

bool is_chunked_manifest(const std::string& path) {
#ifdef _WIN32
    (void) path;
    return false;
#else
    Fd fd(::open(path.c_str(), O_RDONLY | O_NOFOLLOW | O_NONBLOCK | O_CLOEXEC));
    if (!fd) return false;
    char m[8];
    size_t got = 0;
    while (got < 8) {
        const ssize_t r = ::read(fd.get(), m + got, 8 - got);
        if (r < 0 && errno == EINTR) continue;
        if (r <= 0) return false;
        got += (size_t) r;
    }
    return std::memcmp(m, kManMagic, 8) == 0;
#endif
}

namespace {
#ifndef _WIN32
struct Loaded {
    uint64_t file_bytes = 0;
    uint32_t T = 0;
    Parsed p;
};

// the manifest: file checks, header, identity, trailer, then the bounded parse.  `admit` is asked (once) after the
// body is parsed (its largest chunk sizes the buffers) and before anything the image needs is allocated.
bool load_manifest(const std::string& path, const SessionFileIdentity& id, const SessionReadLimits& limits,
                   bool skip_inline, int workers, Loaded& out, std::string& error, SessionStatus& st,
                   bool defer_admit = false) {
    error.clear();
    st = {};
    st.error = SessionError::invalid;
    Fd fd;
    struct stat sb {};
    std::string why;
    bool refused = false;
    int eno = 0;
    if (!open_regular(path, fd, sb, why, refused, &eno)) {
        st.error = refused ? SessionError::invalid : errno_kind(eno);
        error = "chunked session: " + path + ": " + why;
        return false;
    }
    const uint64_t size = (uint64_t) sb.st_size;
    if (size < kHeader + kTrailer) { error = "chunked session: size " + std::to_string(size) + " is below the minimum"; return false; }
    if (size > kMaxManifest) { error = "chunked session: manifest size " + std::to_string(size) + " exceeds the limit"; return false; }
    std::vector<uint8_t> file((size_t) size);
    for (size_t off = 0; off < file.size();) {
        const ssize_t r = ::pread(fd.get(), file.data() + off, file.size() - off, (off_t) off);
        if (r < 0 && errno == EINTR) continue;
        if (r <= 0) {
            st.error = r < 0 ? errno_kind(errno) : SessionError::io;
            error = std::string("chunked session: read error: ") + (r < 0 ? std::strerror(errno) : "file ends early");
            return false;
        }
        off += (size_t) r;
    }
    const uint8_t* h = file.data();
    if (std::memcmp(h, kManMagic, 8) != 0) { error = "chunked session: not a chunked Strata session file (magic)"; return false; }
    if (get_le<uint64_t>(h + 56) != session_hash64(h, 56, 0)) { error = "chunked session: header checksum mismatch"; return false; }
    if (get_le<uint32_t>(h + 8) != kManVersion) { error = "chunked session: unsupported version " + std::to_string(get_le<uint32_t>(h + 8)); return false; }
    const uint32_t T = get_le<uint32_t>(h + 40), flags = get_le<uint32_t>(h + 44);
    if (get_le<uint32_t>(h + 12) != kHeader || flags != 0) { error = "chunked session: invalid header fields"; return false; }
    if (T < 64 || T > 4096 || T % 4) { error = "chunked session: invalid chunk_tokens"; return false; }
    if (get_le<uint64_t>(h + 16) != id.model) { error = "chunked session: saved with another model (model fingerprint differs)"; return false; }
    if (get_le<uint64_t>(h + 24) != id.config) { error = "chunked session: saved with another engine configuration (config fingerprint differs)"; return false; }
    const uint64_t body_len = get_le<uint64_t>(h + 32), raw_total = get_le<uint64_t>(h + 48);
    if (body_len > size || size - kHeader - kTrailer != body_len) {
        error = "chunked session: size " + std::to_string(size) + " does not match the header (" + std::to_string(sat_add(kHeader + kTrailer, body_len)) + ")";
        return false;
    }
    const uint8_t* body = h + kHeader;
    const uint8_t* tr = body + body_len;
    if (get_le<uint64_t>(tr) != session_hash64(body, (size_t) body_len, 0)) { error = "chunked session: body checksum mismatch"; return false; }
    if (std::memcmp(tr + 8, kManEnd, 8) != 0) { error = "chunked session: bad end marker"; return false; }
    if (!skip_inline && !defer_admit) {   // (defer: a prefix read asks for the size of the prefix it builds)
        const uint64_t cap = session_read_max_file_bytes(limits);
        if (raw_total > cap || raw_total > (uint64_t) std::numeric_limits<size_t>::max()) {
            error = "chunked session: image size " + std::to_string(raw_total) + " exceeds this engine's limit";
            return false;
        }
    }
    Parsed parsed;
    if (!parse_body(body, (size_t) body_len, T, limits, skip_inline, parsed, error)) return false;
    if (parsed.raw_total != raw_total) { error = "chunked session: the image size in the header does not match the body"; return false; }
    if (!skip_inline && !defer_admit && limits.admit) {
        // asked once, with the manifest's own largest chunk (the parse is bounded by the limits and kMaxManifest)
        uint64_t max_raw = 0;
        for (const auto& r : parsed.slabs) max_raw = std::max<uint64_t>(max_raw, r.raw_len);
        for (const auto& c : parsed.cps) for (const auto& r : c.refs) max_raw = std::max<uint64_t>(max_raw, r.raw_len);
        std::string why2;
        if (!limits.admit(chunked_read_peak_bytes(raw_total, workers, T, max_raw), why2)) {
            st.error = SessionError::memory;
            error = "chunked session: " + why2;
            return false;
        }
    }
    out.file_bytes = size;
    out.T = T;
    out.p = std::move(parsed);
    st.error = SessionError::none;
    return true;
}

bool read_impl(ChunkPool& pool, const std::string& path, const SessionFileIdentity& id, SavedConversation& out,
               size_t& manifest_bytes, uint64_t& raw_bytes_read, std::string& error, const SessionReadLimits& limits,
               const ChunkedReadOptions& opt, SessionStatus& st) {
    raw_bytes_read = 0;
    Loaded ld;
    // the admission is asked below, once the size of what is built is known (a prefix is smaller than the file's image)
    if (!load_manifest(path, id, limits, false, opt.workers, ld, error, st, true)) return false;
    st.error = SessionError::invalid;
    Parsed& p = ld.p;
    auto fail_invalid = [&](const std::string& m) { error = "chunked session: " + m; return false; };

    // the image's extent: the whole of it, or the prefix of `cut` tokens
    const int64_t n_live = (int64_t) p.s.live.ids.size();
    int64_t cut = 0;
    if (opt.upto < 0) return fail_invalid("negative prefix length");
    if (opt.upto > n_live) return fail_invalid("the prefix is longer than the image");
    if (opt.upto > 0 && opt.upto < n_live) cut = opt.upto;
    Plan pp = p.plan;                       // the plan of the image built (the manifest's for a whole read)
    std::vector<size_t> kept;               // the manifest's checkpoints (index in p.cps, >= 1) the image carries
    size_t chosen = 0;                      // prefix: its position in `kept`
    if (cut > 0) {
        bool found = false;
        for (size_t i = 1; i < p.cps.size(); ++i) {
            const auto& c = p.s.checkpoints[i - 1];
            if ((int64_t) c.ids.size() > cut) continue;
            if (!std::equal(c.ids.begin(), c.ids.end(), p.s.live.ids.begin())) return fail_invalid("a checkpoint is not a prefix of the live state");
            if ((int64_t) c.ids.size() == cut) { chosen = kept.size(); found = true; }
            kept.push_back(i);
        }
        if (!found) return fail_invalid("no checkpoint of " + std::to_string(cut) + " tokens to read");
        std::string why;
        for (auto& g : pp.L) {
            LayerGeom c;
            if (!prefix_layer(g, n_live, cut, c, why)) return fail_invalid(why);
            g = c;
        }
        if (!pp.build(error)) return false;
    } else {
        for (size_t i = 1; i < p.cps.size(); ++i) kept.push_back(i);
    }
    // the bytes that image holds (K/V, every kept checkpoint's state, and for a prefix the chosen one's twice: as the
    // live state too), checked against the engine's limit and offered to the RAM gate
    uint64_t image_raw = pp.kv_total;
    for (size_t i : kept) image_raw = sat_add(image_raw, sat_add(p.cps[i].gdn_len, p.cps[i].inline_bytes));
    if (cut > 0) {
        const auto& m = p.cps[kept[chosen]];
        image_raw = sat_add(image_raw, sat_add(m.gdn_len, m.inline_bytes));
    } else {
        image_raw = sat_add(image_raw, sat_add(p.cps[0].gdn_len, p.cps[0].inline_bytes));
    }
    if (image_raw > session_read_max_file_bytes(limits) || image_raw > (uint64_t) std::numeric_limits<size_t>::max())
        return fail_invalid("image size " + std::to_string(image_raw) + " exceeds this engine's limit");
    // the chunks to read: the slabs [0, pp.n_slabs) and the state slices of the kept checkpoints (and, whole, the live one)
    std::vector<ChunkRef> refs(p.slabs.begin(), p.slabs.begin() + (std::ptrdiff_t) pp.n_slabs);
    if (cut == 0) refs.insert(refs.end(), p.cps[0].refs.begin(), p.cps[0].refs.end());
    for (size_t i : kept) refs.insert(refs.end(), p.cps[i].refs.begin(), p.cps[i].refs.end());
    uint64_t max_raw = 0;
    for (const auto& r : refs) max_raw = std::max<uint64_t>(max_raw, r.raw_len);
    if (limits.admit) {
        std::string why;
        if (!limits.admit(chunked_read_peak_bytes(image_raw, opt.workers, ld.T, max_raw), why)) {
            st.error = SessionError::memory;
            error = "chunked session: " + why;
            return false;
        }
    }
    // every chunk must exist before the image is allocated: the bytes the manifest claims are really on the disk
    for (const auto& r : refs)
        if (!pool.exists(r)) { error = "chunked session: chunk " + r.key.hex() + " is missing"; return false; }

    SavedConversation img = std::move(p.s);
    uint64_t done = 0;
    uint64_t total = 0;
    for (const auto& r : refs) total += r.raw_len;
    // the checkpoints of the image: whole, the manifest's (live apart); a prefix, the kept ones, and the chosen one is the live state
    struct Target { ConversationCheckpoint* cp; size_t meta; };
    std::vector<Target> targets;
    {
        std::vector<ConversationCheckpoint> all = std::move(img.checkpoints);
        ConversationCheckpoint live0 = std::move(img.live);
        img.checkpoints.clear();
        for (size_t i : kept) img.checkpoints.push_back(std::move(all[i - 1]));
        if (cut > 0) {
            img.live = img.checkpoints[chosen];   // (its state is copied below, once the slices are in)
            img.live.used = 0;
        } else {
            img.live = std::move(live0);
        }
        if (cut == 0) targets.push_back({&img.live, 0});
        for (size_t k = 0; k < kept.size(); ++k) targets.push_back({&img.checkpoints[k], kept[k]});
    }
    // K/V buffers: not zeroed (resize_for_overwrite); the coverage count below is the proof that every byte was written
    const size_t n_layers = img.kv.size();
    for (size_t l = 0; l < n_layers; ++l) {
        auto& k = img.kv[l];
        const auto& g = pp.L[l];
        k.cells = g.cells;
        k.pooled_rows = g.pooled_rows;
        k.k.resize_for_overwrite((size_t) g.sizes[0]); k.v.resize_for_overwrite((size_t) g.sizes[1]);
        k.k_scale.resize_for_overwrite((size_t) g.sizes[2]); k.v_scale.resize_for_overwrite((size_t) g.sizes[3]);
        k.pooled.resize_for_overwrite((size_t) g.sizes[4]);
    }
    std::unique_ptr<std::atomic<uint64_t>[]> stored(new std::atomic<uint64_t>[n_layers * 5 + 1]);
    for (size_t i = 0; i < n_layers * 5 + 1; ++i) stored[i] = 0;
    // The chunks are independent: workers fetch, verify and decode them (each into its own scratch, then into its own
    // disjoint range of the image), the first failure stops the rest.  The spare rows come from the manifest.
    std::mutex mu;
    std::string first_error;
    SessionError first_kind = SessionError::invalid;
    std::atomic<bool> failed{false};
    auto fail = [&](SessionError k, const std::string& m) {
        std::lock_guard<std::mutex> g(mu);
        if (!failed) { failed = true; first_kind = k; first_error = m; }
    };
    auto get = [&](ChunkKind kind, const ChunkRef& r, void* dst, size_t cap) {
        SessionError kind_err = SessionError::invalid;
        std::string why;
        if (!pool.get(kind, r, dst, cap, why, &kind_err)) { fail(kind_err, why); return false; }
        std::lock_guard<std::mutex> g(mu);
        done += r.raw_len;
        raw_bytes_read += r.raw_len;
        if (limits.progress) limits.progress(done, total);
        return true;
    };
    struct Task { size_t target = 0; uint32_t j = 0; };   // target = SIZE_MAX: a K/V slab, j = its index
    std::vector<Task> tasks;
    for (uint64_t i = 0; i < pp.n_slabs; ++i) tasks.push_back({SIZE_MAX, (uint32_t) i});
    for (size_t t = 0; t < targets.size(); ++t) {
        targets[t].cp->gdn.resize((size_t) p.cps[targets[t].meta].gdn_len);
        for (uint32_t j = 0; j < p.cps[targets[t].meta].n_slices; ++j) tasks.push_back({t, j});
    }
    auto run = [&](const Task& t, std::vector<uint8_t>& scratch) {
        try {
            if (t.target != SIZE_MAX) {
                ConversationCheckpoint& cp = *targets[t.target].cp;
                const CkptMeta& m = p.cps[targets[t.target].meta];
                const size_t off = (size_t) t.j * m.slice_len;
                get(ChunkKind::state_slice, m.refs[t.j], cp.gdn.data() + off, cp.gdn.size() - off);
                return;
            }
            const uint64_t i = t.j;
            const ChunkRef& r = p.slabs[(size_t) i];
            scratch.resize(r.raw_len);
            if (!get(ChunkKind::kv_slab, r, scratch.data(), scratch.size())) return;
            size_t at = 0;
            for (size_t part = 0; part < 5; ++part)
                for (size_t l = 0; l < n_layers; ++l) {
                    uint64_t off, full, n;
                    p.plan.slice(l, part, i, off, full);   // what the chunk carries
                    if (!full) continue;
                    pp.slice(l, part, i, off, n);          // what the image wants of it (a prefix: the head)
                    if (n) {
                        auto& k = img.kv[l];
                        ConversationBuffer* b[5] = {&k.k, &k.v, &k.k_scale, &k.v_scale, &k.pooled};
                        if (!write_buffer_at(*b[part], off, scratch.data() + at, n)) { fail(SessionError::invalid, "chunked session: slab does not fit its part"); return; }
                        stored[l * 5 + part] += n;
                    }
                    at += (size_t) full;
                }
        } catch (const std::bad_alloc&) {
            fail(SessionError::memory, "chunked session: out of memory while reading a chunk");
        } catch (const std::exception& e) {
            fail(SessionError::io, std::string("chunked session: ") + e.what());
        }
    };
    const size_t n_workers = std::min<size_t>((size_t) std::clamp(opt.workers, 1, 8), tasks.size());
    if (n_workers <= 1) {
        std::vector<uint8_t> scratch;
        for (const Task& t : tasks) { if (failed) break; run(t, scratch); }
    } else {
        std::atomic<size_t> next{0};
        std::vector<std::thread> threads;
        for (size_t w = 0; w < n_workers; ++w) {
            try {
                threads.emplace_back([&] {
                    // off the host thread's core: a read back runs on the host thread, whose one CPU a new thread inherits
                    strata::aux_cpus::spread_current_thread();
                    std::vector<uint8_t> scratch;
                    for (size_t i; !failed && (i = next.fetch_add(1)) < tasks.size();) run(tasks[i], scratch);
                });
            } catch (const std::system_error&) { break; }
        }
        {   // the calling thread works too: with no thread made it is the only one
            std::vector<uint8_t> scratch;
            for (size_t i; !failed && (i = next.fetch_add(1)) < tasks.size();) run(tasks[i], scratch);
        }
        for (auto& t : threads) t.join();
    }
    if (failed) { st.error = first_kind; error = first_error; return false; }
    // the spare rows: the manifest's (whole), or for a prefix the chosen checkpoint's `dead` (what its restore writes there)
    {
        size_t index_layer = 0, n_index = 0;
        for (const auto& g : pp.L) if (g.pooled_rows > 0) ++n_index;
        const auto& dead = img.live.dead;
        for (size_t l = 0; l < n_layers; ++l) {
            if (pp.L[l].pooled_rows == 0) continue;
            const uint64_t row = pp.row[l], off = pp.data_rows[l] * row;
            const uint8_t* src = p.spare[l].data();
            size_t len = p.spare[l].size();
            if (cut > 0) {
                if (row && dead.size() != n_index * row) return fail_invalid("the checkpoint's indexer rows do not fit the layers");
                src = dead.data() + index_layer * row;
                len = (size_t) row;
            }
            ++index_layer;
            if (len && !write_buffer_at(img.kv[l].pooled, off, src, len)) return fail_invalid("the spare row does not fit");
            stored[l * 5 + 4] += len;
        }
    }
    // The coverage proof for resize_for_overwrite: the slices are disjoint by construction, so a part whose stored bytes add up
    // to its size has every byte written; any other count means the manifest's slabs did not cover the geometry.
    for (size_t l = 0; l < n_layers; ++l)
        for (size_t part = 0; part < 5; ++part)
            if (stored[l * 5 + part] != pp.L[l].sizes[part])
                return fail_invalid("the slabs do not cover K/V layer " + std::to_string(l) + " part " + std::to_string(part));
    if (cut > 0) {   // the chosen checkpoint is the live state too
        img.live.gdn = img.checkpoints[chosen].gdn;
    }
    out = std::move(img);
    manifest_bytes = (size_t) ld.file_bytes;
    st.error = SessionError::none;
    return true;
}
#endif
} // namespace

bool chunked_read(ChunkPool& pool, const std::string& path, const SessionFileIdentity& id, SavedConversation& out,
                  size_t& manifest_bytes, uint64_t& raw_bytes_read, std::string& error,
                  const SessionReadLimits& limits, const ChunkedReadOptions& opt, SessionStatus* status) {
    SessionStatus st;
    bool ok = false;
#ifdef _WIN32
    (void) pool; (void) path; (void) id; (void) out; (void) manifest_bytes; (void) raw_bytes_read; (void) limits; (void) opt;
    error = "the chunked disk format is POSIX only";
    st.error = SessionError::invalid;
#else
    try {
        ok = read_impl(pool, path, id, out, manifest_bytes, raw_bytes_read, error, limits, opt, st);
    } catch (const std::bad_alloc&) {
        error = "chunked session: out of memory while reading " + path;
        st.error = SessionError::memory;
        ok = false;
    }
#endif
    if (status) *status = st;
    return ok;
}

bool chunked_manifest_refs(const std::string& path, const SessionFileIdentity& id, std::vector<ChunkRef>& refs,
                           uint64_t& manifest_bytes, std::string& error, bool* transient) {
    if (transient) *transient = false;
#ifdef _WIN32
    (void) path; (void) id; (void) refs; (void) manifest_bytes;
    error = "the chunked disk format is POSIX only";
    return false;
#else
    Loaded ld;
    SessionStatus st;
    SessionReadLimits none;
    try {
        if (!load_manifest(path, id, none, true, 1, ld, error, st)) {
            if (transient) *transient = st.error != SessionError::invalid;
            return false;
        }
    } catch (const std::bad_alloc&) {
        error = "chunked session: out of memory while reading " + path;
        if (transient) *transient = true;
        return false;
    }
    refs = all_refs(ld.p);
    manifest_bytes = ld.file_bytes;
    return true;
#endif
}

} // namespace strata::core
