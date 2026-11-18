// src/core/conversation_disk.cpp - the disk tier of the parked conversations.  See the header.
// (xrip, upstream PR #1937 + b0c4d361 / 65f3fd00; the strata-nvfp4 additions are marked in the header.)
#include "strata/core/conversation_disk.hpp"
#include "strata/platform/aux_cpus.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <map>
#include <random>
#include <set>
#include <system_error>
#include <unordered_set>
#include <utility>
#ifndef _WIN32
#include <fcntl.h>
#include <sys/stat.h>
#endif
#if defined(__linux__)
#include <sys/resource.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif
#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace strata::core {
namespace {

namespace fs = std::filesystem;

// Paths are UTF-8 everywhere (session_file_write / _read take UTF-8 and use the wide APIs on Windows); a std::string
// given to std::filesystem::path is read in the ANSI code page on Windows, so these two convert explicitly.
fs::path upath(const std::string& s) { return fs::path(std::u8string(s.begin(), s.end())); }
std::string ustr(const fs::path& p) {
    const std::u8string u = p.u8string();
    return std::string(u.begin(), u.end());
}

// The second half of every file's identity: with this run's random number as the first, a file another run (or
// another tier) wrote is refused on read.
constexpr uint64_t kTierConfig = 0x5354524154414449ull;   // "STRATADI"

// ids and images only: the part of a checkpoint best() matches against
ConversationCheckpoint key_of(const ConversationCheckpoint& c) {
    ConversationCheckpoint k;
    k.ids = c.ids;
    k.imgs = c.imgs;
    return k;
}

bool same_key(const ConversationCheckpoint& x, const ConversationCheckpoint& y) {
    return x.ids == y.ids && x.imgs == y.imgs;
}

// The payload is the conversation its index names: the same keys and control-vector state, in every stage.
bool holds(const SavedConversation& s, const ConversationCheckpoint& live,
           const std::vector<ConversationCheckpoint>& checkpoints, bool cvec) {
    if (s.cvec != cvec || !same_key(s.live, live) || s.checkpoints.size() != checkpoints.size()) return false;
    for (size_t i = 0; i < checkpoints.size(); ++i)
        if (!same_key(s.checkpoints[i], checkpoints[i])) return false;
    for (const auto& stage : s.stage_images)
        if (!holds(stage, live, checkpoints, cvec)) return false;
    return true;
}

// a file of this tier (also a temporary one session_file_write left behind: ".<name>.<hex>.tmp")
bool tier_file(const std::string& name) {
    const std::string prefix = ConversationDiskTier::file_prefix();
    return name.rfind(prefix, 0) == 0 || name.rfind("." + prefix, 0) == 0;
}

void remove_file(const std::string& path) {
    std::error_code ec;
    fs::remove(upath(path), ec);
}

// The index file of a persistent entry: native-endian fields (the session files refuse a big-endian build too),
// counts before arrays, and (v2, this fork) a hash of everything before it at the end: an index a power loss tore is
// refused at start like a damaged one, never trusted.
constexpr uint64_t kIndexMagic = 0x3258444943525453ull;   // "STRCIDX2"

struct IndexWriter {
    std::string b;
    template<class T> void put(T v) { b.append(reinterpret_cast<const char*>(&v), sizeof v); }
    void str(const std::string& s) { put<uint32_t>((uint32_t) s.size()); b += s; }
    void key(const ConversationCheckpoint& c) {
        put<uint32_t>((uint32_t) c.ids.size());
        b.append(reinterpret_cast<const char*>(c.ids.data()), c.ids.size() * sizeof(int32_t));
        put<uint32_t>((uint32_t) c.imgs.size());
        for (const auto& i : c.imgs) { put<int64_t>(i.start); put<uint64_t>(i.hash); }
    }
};

struct IndexReader {
    const std::string& b;
    size_t at = 0;
    bool ok = true;
    size_t left() const { return b.size() - at; }
    template<class T> T get() {
        T v{};
        if (left() < sizeof v) { ok = false; return v; }
        std::memcpy(&v, b.data() + at, sizeof v);
        at += sizeof v;
        return v;
    }
    std::string str() {
        const uint32_t n = get<uint32_t>();
        if (!ok || left() < n) { ok = false; return {}; }
        at += n;
        return b.substr(at - n, n);
    }
    bool key(ConversationCheckpoint& c) {
        const uint32_t n = get<uint32_t>();
        if (!ok || left() / sizeof(int32_t) < n) return ok = false;
        c.ids.resize(n);
        std::memcpy(c.ids.data(), b.data() + at, n * sizeof(int32_t));
        at += n * sizeof(int32_t);
        const uint32_t m = get<uint32_t>();
        if (!ok || left() / 16 < m) return ok = false;
        for (uint32_t i = 0; i < m; ++i) c.imgs.push_back({get<int64_t>(), get<uint64_t>()});
        return ok;
    }
};

double mib(uint64_t bytes) { return double(bytes) / (1024.0 * 1024.0); }
double mbps(uint64_t bytes, double ms) { return ms > 0 ? double(bytes) / 1e6 / (ms / 1000.0) : 0.0; }
double ms_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

#if defined(__GNUC__)
std::string fmt(const char* f, ...) __attribute__((format(printf, 1, 2)));
#endif
std::string fmt(const char* f, ...) {
    char buf[2048];
    va_list ap;
    va_start(ap, f);
    std::vsnprintf(buf, sizeof buf, f, ap);
    va_end(ap);
    return buf;
}

// the refs of one entry, each key once (a stat per chunk, not per mention)
std::vector<ChunkRef> unique_refs(const std::vector<ChunkRef>& refs) {
    std::set<ChunkKey> seen;
    std::vector<ChunkRef> out;
    for (const auto& r : refs)
        if (seen.insert(r.key).second) out.push_back(r);
    return out;
}

// the file's mtime in nanoseconds (0 when it cannot be read)
int64_t file_mtime_ns(const std::string& path) {
#ifndef _WIN32
    struct stat st {};
    if (::stat(path.c_str(), &st) != 0) return 0;
    return (int64_t) st.st_mtim.tv_sec * 1000000000ll + (int64_t) st.st_mtim.tv_nsec;
#else
    std::error_code ec;
    const auto t = fs::last_write_time(upath(path), ec);
    return ec ? 0 : (int64_t) std::chrono::duration_cast<std::chrono::nanoseconds>(t.time_since_epoch()).count();
#endif
}

uint64_t sources_bytes(const std::vector<SessionKvSource>* sources) {
    uint64_t n = 0;
    if (sources)
        for (const auto& s : *sources)
            for (size_t b : s.sizes) n += b;
    return n;
}

}  // namespace

ConversationDiskTier::ConversationDiskTier(ConversationDiskOptions options, std::function<Clock::time_point()> now)
    : options_(std::move(options)), now_(std::move(now)) {
    std::random_device rd;
    nonce_ = ((uint64_t) rd() << 32) ^ rd() ^ (uint64_t) std::chrono::system_clock::now().time_since_epoch().count();
}

ConversationDiskTier::~ConversationDiskTier() {
    if (open_ && options_.persist) wait_idle();   // what is queued belongs to the next start
    size_t dropped = 0;
    {
        std::lock_guard<std::mutex> lock(m_);
        stop_ = true;
        dropped = queue_.size();
        queue_.clear();   // not written: dropped, as the RAM cache would have
    }
    cv_.notify_all();
    if (thread_.joinable()) thread_.join();
    std::lock_guard<std::mutex> lock(m_);
    if (!open_) return;
    if (options_.persist) {
        log(fmt("end: %zu conversations kept on the disk for the next start (%.0f MiB)", entries_.size(),
                mib(bytes_ + pool_bytes_all())));
        return;
    }
    if (dropped || !entries_.empty())
        log(fmt("end: %zu queued dropped, %zu conversations removed from the disk", dropped, entries_.size()));
    while (!entries_.empty()) remove_locked(0, true);
    if (pool_) pool_->remove_all();   // not persistent: the chunks go with the tier
}

SessionFileIdentity ConversationDiskTier::file_identity() const {
    return options_.persist ? options_.identity : SessionFileIdentity{nonce_, kTierConfig};
}

std::string ConversationDiskTier::pool_dir() const {
    char name[64];
    if (options_.persist) {
        const uint64_t id[2] = {options_.identity.model, options_.identity.config};
        std::snprintf(name, sizeof name, "strata-pool-%016llx", (unsigned long long) session_hash64(id, sizeof id, 0));
    } else {
        std::snprintf(name, sizeof name, "strata-pool-n%016llx", (unsigned long long) nonce_);
    }
    return ustr(upath(options_.dir) / name);
}

void ConversationDiskTier::remove_foreign_pools() const {
    if (options_.shared) return;   // other engines' pools are theirs
    std::error_code ec;
    const std::string ours = ustr(upath(pool_dir()).filename());
    for (fs::directory_iterator it(upath(options_.dir), ec), end; !ec && it != end; it.increment(ec)) {
        const std::string name = ustr(it->path().filename());
        if (name.rfind("strata-pool-", 0) != 0 || name == ours) continue;
        std::error_code dec;
        if (!it->is_directory(dec) || it->is_symlink(dec)) continue;
        // not persistent: the leftovers of earlier runs (random names); persistent: the other identities' pools
        if (options_.persist || name.rfind("strata-pool-n", 0) == 0) fs::remove_all(it->path(), dec);
    }
}

uint64_t ConversationDiskTier::pool_bytes_all() const { return pool_ ? pool_->bytes() : 0; }

size_t ConversationDiskTier::pool_chunks() const {
    std::lock_guard<std::mutex> lock(m_);
    return pool_ ? pool_->count() : 0;
}

void ConversationDiskTier::release_refs(Entry& e) const {
    if (pool_ && !e.refs.empty()) pool_->unref(e.refs.data(), e.refs.size());
    e.refs.clear();
}

// The manifests of a scanned entry: parsed for their chunk references, every chunk checked on the disk (a missing one
// refuses the entry - a power loss tore it, or another engine swept it), then the references are taken.  A v1 entry
// passes untouched.
bool ConversationDiskTier::bind_chunks(Entry& e) const {
    size_t chunked = 0;
    for (const auto& f : e.files) chunked += is_chunked_manifest(f) ? 1 : 0;
    if (chunked == 0) return true;
    if (chunked != e.files.size() || !pool_) return false;
    const SessionFileIdentity fid = file_identity();
    std::vector<ChunkRef> refs;
    uint64_t mbytes = 0;
    for (const auto& f : e.files) {
        std::vector<ChunkRef> r;
        uint64_t mb = 0;
        std::string why;
        if (!chunked_manifest_refs(f, fid, r, mb, why)) return false;
        refs.insert(refs.end(), r.begin(), r.end());
        mbytes += mb;
    }
    for (const auto& r : unique_refs(refs))
        if (!pool_->exists(r)) return false;
    pool_->ref(refs.data(), refs.size());
    e.refs = std::move(refs);
    e.chunked = true;
    e.manifest_bytes = mbytes;
    e.bytes = mbytes;
    return true;
}

bool ConversationDiskTier::write_index(Entry& e, std::string& error) const {
    char name[96];
    std::snprintf(name, sizeof name, "%s%016llx-%llu.idx", file_prefix(), (unsigned long long) nonce_,
                  (unsigned long long) e.id);
    const fs::path path = upath(options_.dir) / name;
    const fs::path temp = upath(options_.dir) / ("." + std::string(name) + ".tmp");
    e.written_s = std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    const SessionFileIdentity id = file_identity();
    IndexWriter out;
    out.put(kIndexMagic);
    out.put(id.model);
    out.put(id.config);
    out.put<int64_t>(e.written_s);
    out.put<uint8_t>(e.cvec ? 1 : 0);
    out.put<uint64_t>(e.bytes);
    out.put<uint32_t>((uint32_t) e.files.size());
    for (const auto& f : e.files) out.str(ustr(upath(f).filename()));
    out.key(e.live);
    out.put<uint32_t>((uint32_t) e.checkpoints.size());
    for (const auto& c : e.checkpoints) out.key(c);
    out.put<uint64_t>(session_hash64(out.b.data(), out.b.size(), kIndexMagic));
    {
        std::ofstream f(temp, std::ios::binary | std::ios::trunc);
        f.write(out.b.data(), (std::streamsize) out.b.size());
        f.flush();
        if (!f) {
            error = "cannot write the index " + ustr(temp);
            f.close();
            remove_file(ustr(temp));
            return false;
        }
    }
    std::error_code ec;
    fs::rename(temp, path, ec);   // a reader sees the whole index or none
    if (ec) {
        error = "cannot publish the index: " + ec.message();
        remove_file(ustr(temp));
        return false;
    }
    e.index = ustr(path);
    return true;
}

bool ConversationDiskTier::read_index(const std::string& path, Entry& e, bool* transient) const {
    if (transient) *transient = false;
    std::ifstream f(upath(path), std::ios::binary);
    std::string b;
    try {
        b.assign((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    } catch (const std::bad_alloc&) {
        if (transient) *transient = true;
        return false;
    }
    if (transient) {   // an open or read error says nothing about the index (a missing file is a removed entry: not transient)
        std::error_code xec;
        if ((!f.is_open() || f.bad()) && fs::exists(upath(path), xec)) *transient = true;
    }
    if (b.size() < 8) return false;
    uint64_t stored = 0;
    std::memcpy(&stored, b.data() + b.size() - 8, 8);
    if (stored != session_hash64(b.data(), b.size() - 8, kIndexMagic)) return false;   // torn or damaged
    const std::string body = b.substr(0, b.size() - 8);
    IndexReader in{body};
    const SessionFileIdentity id = file_identity();
    if (in.get<uint64_t>() != kIndexMagic || in.get<uint64_t>() != id.model || in.get<uint64_t>() != id.config)
        return false;
    e.written_s = in.get<int64_t>();
    e.cvec = in.get<uint8_t>() != 0;
    e.bytes = in.get<uint64_t>();
    const uint32_t nfiles = in.get<uint32_t>();
    for (uint32_t i = 0; in.ok && i < nfiles; ++i) {
        const std::string name = in.str();
        // a bare name of this tier in this folder, nothing else (no separators, no "..")
        if (name.find_first_of("/\\:") != std::string::npos || name.rfind(file_prefix(), 0) != 0) return false;
        e.files.push_back(ustr(upath(options_.dir) / upath(name)));
    }
    if (!in.key(e.live)) return false;
    const uint32_t nck = in.get<uint32_t>();
    for (uint32_t i = 0; in.ok && i < nck; ++i) {
        ConversationCheckpoint c;
        if (in.key(c)) e.checkpoints.push_back(std::move(c));
    }
    if (!in.ok || in.left() != 0 || e.files.empty()) return false;
    std::error_code ec;
    for (const auto& file : e.files)
        if (!fs::is_regular_file(upath(file), ec)) return false;
    e.index = path;
    return true;
}

std::string ConversationDiskTier::scan_folder(size_t& stale, std::error_code& ec) {
    std::vector<Entry> found;
    std::vector<std::string> others;
    size_t refused = 0;
    for (fs::directory_iterator it(upath(options_.dir), ec), end; !ec && it != end; it.increment(ec)) {
        if (!it->is_regular_file(ec) || !tier_file(ustr(it->path().filename()))) continue;
        Entry e;
        const bool index = it->path().extension() == ".idx";
        const bool taken = index && options_.persist && read_index(ustr(it->path()), e) && bind_chunks(e);
        if (taken) e.idx_ns = file_mtime_ns(ustr(it->path()));
        if (index && options_.shared) seen_index_.insert(ustr(it->path()));
        if (taken) found.push_back(std::move(e));
        else others.push_back(ustr(it->path()));
        refused += index && !taken;
    }
    // the files of the entries taken in stay; the rest goes (refused indexes and their files, interrupted writes).
    // Shared: another engine's files (another identity) and young leftovers (a write still running) stay - only a
    // leftover untouched for kSharedGrace goes.
    std::set<std::string> kept;
    for (const auto& e : found) kept.insert(e.files.begin(), e.files.end());
    const auto now = fs::file_time_type::clock::now();
    for (const auto& p : others) {
        if (kept.count(p)) continue;
        if (options_.shared) {
            const fs::path fp = upath(p);
            if (fp.extension() == ".idx") continue;   // another identity's (or a torn one): not this engine's to judge
            // an image whose entry's index exists (whatever its identity) belongs to that entry
            const std::string name = ustr(fp.filename());
            const size_t dash = name.rfind('-');
            std::error_code xec;
            if (fp.extension() == ".bin" && dash != std::string::npos &&
                fs::exists(fp.parent_path() / upath(name.substr(0, dash) + ".idx"), xec))
                continue;
            std::error_code tec;
            const auto t = fs::last_write_time(fp, tec);
            if (tec || now - t < std::chrono::minutes(10)) continue;   // perhaps still being written by another engine
        }
        remove_file(p);
        ++stale;
    }
    // the LRU order a restart rebuilds: the index files' mtimes (a keep-take touches one), then the time it was written
    std::sort(found.begin(), found.end(), [](const Entry& a, const Entry& b) {
        if (a.idx_ns != b.idx_ns) return a.idx_ns < b.idx_ns;
        if (a.written_s != b.written_s) return a.written_s < b.written_s;
        return a.index < b.index;
    });
    for (auto& e : found) {
        e.id = ++next_id_;
        e.at = now_();
        bytes_ += e.bytes;
        entries_.push_back(std::move(e));
    }
    // (chunks no manifest names - an interrupted write, a removed entry - stay as ghosts, oldest first out)
    size_t over = 0;
    make_room_locked(0, 0, over);
    if (!options_.persist) return {};
    return fmt("persistent (storage limit only, oldest out first): %zu conversations from earlier runs (%.0f MiB), %zu refused "
               "(another model, engine version or settings), %zu over the limit removed", entries_.size(),
               mib(bytes_ + pool_bytes_all()), refused, over);
}

void ConversationDiskTier::log(const std::string& line) const {
    if (options_.log) options_.log(line);
    else std::fprintf(stderr, "strata serve: conversation disk: %s\n", line.c_str());
}

bool ConversationDiskTier::open(std::string& error) {
    open_ = false;
    if (options_.dir.empty() || options_.budget_bytes == 0) return true;   // off: nothing to open
    std::error_code ec;
    const fs::path dir = upath(options_.dir);
    fs::create_directories(dir, ec);
    if (ec || !fs::is_directory(dir, ec)) {
        error = "cannot create the folder " + options_.dir + (ec ? ": " + ec.message() : std::string());
        return false;
    }
#ifndef _WIN32
    fs::permissions(dir, fs::perms::owner_all, fs::perm_options::replace, ec);   // the files hold conversations
    ec.clear();
#endif
    pool_.reset();
    remove_foreign_pools();
#ifndef _WIN32
    const std::string pdir = pool_dir();
    std::error_code pec;
    if (options_.chunked || (options_.persist && fs::is_directory(upath(pdir), pec))) {
        // (a persistent folder an earlier chunked run left is read and swept even when this run writes v1)
        ChunkPoolOptions po;
        po.dir = pdir;
        po.codec = options_.codec;
        po.zstd_level = options_.codec_level;
        po.min_free_bytes = options_.min_free_bytes;
        po.shared = options_.shared;
        auto pool = std::make_unique<ChunkPool>(std::move(po));
        if (!pool->open(error)) return false;
        pool_ = std::move(pool);
    }
#else
    if (options_.chunked) log("chunked format: POSIX only, writing the one-file format");
#endif
    size_t stale = 0;
    std::string taken_in;
    {
        std::lock_guard<std::mutex> lock(m_);
        taken_in = scan_folder(stale, ec);
    }
    if (ec) {
        error = "cannot list the folder " + options_.dir + ": " + ec.message();
        return false;
    }
    open_ = true;
    thread_ = std::thread([this] { writer(); });
    const std::string kept = options_.persist ? taken_in
        : options_.chunked && pool_ ? std::string("no time limit, oldest out first")
        : fmt("%lld min per conversation", (long long) (options_.ttl_s / 60));
    std::string codec = "raw";
    if (options_.codec == ChunkCodec::zstd)
        codec = codec_available(ChunkCodec::zstd) ? fmt("zstd:%d", options_.codec_level) : "raw (zstd is not built in)";
    flush_warnings();
    log(fmt("on: %s, storage limit %.0f MiB, %s; %zu stale files removed%s", options_.dir.c_str(), mib(options_.budget_bytes),
            kept.c_str(), stale,
            options_.chunked && pool_ ? fmt("; chunked format v2, %u tokens a slab, codec %s, %d I/O workers, live state %s, pool %s",
                                            options_.chunk_tokens, codec.c_str(), options_.io_workers,
                                            options_.live_policy == LivePolicy::always ? "always written"
                                            : fmt("%s when a checkpoint is within %lld tokens",
                                                  options_.live_policy == LivePolicy::skip ? "skipped" : "skipped in background writes",
                                                  (long long) options_.live_gap).c_str(),
                                            pool_dir().c_str()).c_str() : ""));
    return true;
}

void ConversationDiskTier::remove_locked(size_t index, bool delete_files) {
    Entry& e = entries_.at(index);
    if (!e.index.empty()) remove_file(e.index);   // first, always: an entry gone from the list is not taken in again
    if (delete_files)
        for (const auto& f : e.files) remove_file(f);
    release_refs(e);   // the chunks become the newest ghosts (deleted, oldest first, only when the limit needs the room)
    bytes_ -= e.bytes;
    entries_.erase(entries_.begin() + (std::ptrdiff_t) index);
}

size_t ConversationDiskTier::find_locked(uint64_t id) const {
    for (size_t i = 0; i < entries_.size(); ++i)
        if (entries_[i].id == id) return i;
    return entries_.size();
}

size_t ConversationDiskTier::expire_locked() {
    // the chunked format has no time limit: the storage limit decides (v1 keeps the upstream time limit)
    if (!open_ || options_.ttl_s <= 0 || (options_.chunked && pool_)) return 0;
    const auto now = now_();
    const auto ttl = std::chrono::seconds(options_.ttl_s);
    size_t n = 0;
    // oldest first: the rest are younger (an entry a request is reading is left for the next pass)
    while (!entries_.empty() && entries_.front().id != reading_ && now - entries_.front().at >= ttl) {
        remove_locked(0, true);
        ++n;
    }
    stats_.expired += n;
    return n;
}

size_t ConversationDiskTier::expire() {
    size_t n = 0;
    {
        std::lock_guard<std::mutex> lock(m_);
        n = expire_locked();
    }
    if (n) log(fmt("removed %zu conversation%s older than %lld min", n, n == 1 ? "" : "s",
                   (long long) (options_.ttl_s / 60)));
    return n;
}

bool ConversationDiskTier::put(SavedConversation&& image, bool flush) {
    if (!open_) return false;
    Job job;
    job.image = std::move(image);
    job.flush = flush;
    job.entry.live = key_of(job.image.live);
    for (const auto& c : job.image.checkpoints) job.entry.checkpoints.push_back(key_of(c));
    job.entry.cvec = job.image.cvec;
    job.entry.bytes = job.image.bytes();
    const size_t tokens = job.entry.live.ids.size();
    // chunked: the raw size is not the footprint (dedup, zstd): the real bytes are known after the write, and the limit
    // loop (make_room_locked, LimitEvent::remaining) judges those.  Only the v1 file's size is exact up front.
    if (!(options_.chunked && pool_) && job.entry.bytes > options_.budget_bytes) {
        log(fmt("dropped %zu tokens: %.0f MiB is over the storage limit", tokens, mib(job.entry.bytes)));
        {
            std::lock_guard<std::mutex> lock(m_);
            ++stats_.write_failed;
            ++stats_.limit_rejected;
            warn_limit_locked(LimitEvent::rejected, (int64_t) tokens, job.entry.bytes, job.entry.bytes);
        }
        flush_warnings();
        return false;
    }
    {
        std::lock_guard<std::mutex> lock(m_);
        job.entry.id = ++next_id_;
        ++stats_.queued;
    }
    size_t depth = 0;
    uint64_t queued_bytes = 0;
    const uint64_t estimate = job.entry.bytes;
    {
        std::lock_guard<std::mutex> lock(m_);
        queue_.push_back(std::move(job));
        depth = queue_.size();
        for (const auto& j : queue_) queued_bytes += j.entry.bytes;
    }
    cv_.notify_all();
    log(fmt("queued %zu tokens (%.0f MiB in RAM until written; queue %zu, %.0f MiB)", tokens, mib(estimate), depth,
            mib(queued_bytes)));
    return true;
}

void ConversationDiskTier::writer() {
    // off the host thread's core: the tier is opened by the host thread, already pinned to its one CPU, which this
    // thread inherits (strata/platform/aux_cpus.hpp; a no-op before the host is placed or off Linux)
    strata::aux_cpus::spread_current_thread();
#if defined(__linux__)
    // below the serving threads: the write is background work, decode and prompt reading come first
    setpriority(PRIO_PROCESS, (id_t) syscall(SYS_gettid), 10);
#elif defined(_WIN32)
    SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_BELOW_NORMAL);
#endif
    for (;;) {
        Job job;
        {
            std::unique_lock<std::mutex> lock(m_);
            // a job starts while the engine is idle, or when someone waits for it (its RAM, or its entry); shared: at
            // once, the other replicas may want it while this engine stays busy
            cv_.wait(lock, [&] { return stop_ || (!queue_.empty() && (!busy_ || urgent_ > 0 || options_.shared)); });
            if (stop_) return;
            job = std::move(queue_.front());
            queue_.pop_front();
            writing_ = true;
            writing_entry_ = job.entry;   // best() still sees it; take() waits for its file
        }
        write(job);
        {
            std::lock_guard<std::mutex> lock(m_);
            writing_ = false;
            writing_entry_ = Entry{};
        }
        cv_.notify_all();
    }
}

namespace {
std::string skip_note(size_t live, int64_t as_of) {
    return fmt("; live state not written: as of the checkpoint at %lld tokens (%zu tokens left out)", (long long) as_of,
               live - (size_t) as_of);
}
}  // namespace

int64_t ConversationDiskTier::live_skip_tokens(const SavedConversation& img, bool background) const {
    // only the chunked writer can cut an image (the v1 file holds the K/V as it is), and a layer split's stage images
    // are not cut here (each stage would need its own checkpoint set): they are written whole
    if (!options_.chunked || !pool_ || options_.live_policy == LivePolicy::always || !img.stage_images.empty()) return 0;
    if (options_.live_policy == LivePolicy::automatic && !background) return 0;
    const int64_t live = (int64_t) img.live.ids.size();
    int64_t deepest = 0;
    for (const auto& c : img.checkpoints) {
        const int64_t n = (int64_t) c.ids.size();
        if (n > deepest && n < live && c.stage_parts.empty() && std::equal(c.ids.begin(), c.ids.end(), img.live.ids.begin()))
            deepest = n;
    }
    return conversation_disk_skip_live(options_.live_policy, live, deepest, options_.live_gap) ? deepest : 0;
}

bool ConversationDiskTier::write_files(Entry& e, const SavedConversation& img,
                                       const std::vector<SessionKvSource>* sources,
                                       const std::function<bool()>& cancel, std::string& error, bool flush, int64_t as_of) {
    const SessionFileIdentity id = file_identity();
    SessionWriteOptions wo;
    wo.min_free_bytes = options_.min_free_bytes;
    // a cache: a file a power loss damages fails its hash check and is read again from the prompt.  Flushed only for
    // RAM pressure: its bytes must leave the modified file cache before the next RAM reading
    wo.durable = flush;
    wo.cancel = cancel;
    e.bytes = 0;
    const bool use_chunks = options_.chunked && pool_;
    // shared: the chunks this write hits or writes are announced in a lease until its index is published, so another
    // engine's eviction cannot take one a manifest of ours is about to name
    std::unique_ptr<ChunkLease> lease;
    if (use_chunks && options_.shared) lease = pool_->begin_lease();
    ChunkedWriteStats totals;
    bool storage_failed = false;   // the disk is full (or its reserve is): the chunks this write made are ghosts to free now
    auto write_one = [&](const SavedConversation& part, const std::vector<SessionKvSource>* src, size_t k) -> bool {
        char name[96];
        std::snprintf(name, sizeof name, "%s%016llx-%llu-%zu.bin", file_prefix(), (unsigned long long) nonce_,
                      (unsigned long long) e.id, k);
        const std::string path = ustr(upath(options_.dir) / name);
        size_t written = 0;
        if (use_chunks) {
            // one manifest per stage, the chunks in the one pool; the manifest is the entry's file
            ChunkedWriteOptions co;
            co.chunk_tokens = options_.chunk_tokens;
            co.durable = flush;
            co.min_free_bytes = options_.min_free_bytes;
            co.workers = options_.io_workers;
            co.cancel = cancel;
            co.as_of_tokens = k == 0 ? as_of : 0;
            co.lease = lease.get();
            if (options_.shared || options_.before_manifest)   // another engine's eviction may have taken a chunk this write hit: look again
                co.before_publish = [this](const std::vector<ChunkRef>& refs, std::string& why) {
                    if (options_.before_manifest) options_.before_manifest();
                    if (!options_.shared) return true;
                    for (const auto& r : unique_refs(refs))
                        if (!pool_->exists(r)) { why = "chunk " + r.key.hex() + " vanished before the manifest"; return false; }
                    return true;
                };
            std::vector<ChunkRef> refs;
            ChunkedWriteStats ws;
            SessionStatus cst;
            if (!chunked_write(*pool_, path, part, src, id, co, refs, ws, error, &cst)) {
                storage_failed = storage_failed || cst.error == SessionError::storage;
                remove_file(path);   // (a manifest published before a failed folder flush too)
                return false;
            }
            e.files.push_back(path);
            e.refs.insert(e.refs.end(), refs.begin(), refs.end());
            e.bytes += ws.manifest_bytes;
            e.manifest_bytes += ws.manifest_bytes;
            e.chunked = true;
            totals.raw_bytes += ws.raw_bytes;
            totals.hit_raw_bytes += ws.hit_raw_bytes;
            totals.new_file_bytes += ws.new_file_bytes;
            totals.hits += ws.hits;
            totals.misses += ws.misses;
            totals.direct_io = totals.direct_io || ws.direct_io;
            return true;
        }
        // one file per stage: session_file_write writes an image's own state and K/V, never its stage_images
        const bool ok = src ? session_file_write(path, part, *src, id, written, error, wo)
                            : session_file_write(path, part, id, written, error, wo);
        if (!ok) {
            remove_file(path);
            return false;
        }
        e.files.push_back(path);
        e.bytes += written;
        return true;
    };
    bool ok = write_one(img, sources, 0);
    for (size_t k = 0; ok && k < img.stage_images.size(); ++k) ok = write_one(img.stage_images[k], nullptr, k + 1);
    if (ok && as_of > 0) {
        // what is on the disk is the image of `as_of` tokens: its keys are the entry's (best() and the index say so)
        e.checkpoints.clear();
        for (const auto& c : img.checkpoints) {
            if ((int64_t) c.ids.size() < as_of) e.checkpoints.push_back(key_of(c));
            else if ((int64_t) c.ids.size() == as_of) { e.live = key_of(c); e.checkpoints.push_back(key_of(c)); }
        }
    }
    if (ok && options_.persist) {
        if (options_.shared) {   // ours: rescan() must not take it in as another engine's (it is seen before it exists)
            char name[96];
            std::snprintf(name, sizeof name, "%s%016llx-%llu.idx", file_prefix(), (unsigned long long) nonce_,
                          (unsigned long long) e.id);
            std::lock_guard<std::mutex> lock(m_);
            seen_index_.insert(ustr(upath(options_.dir) / name));
        }
        ok = write_index(e, error);   // after the images: an index names complete files only
    }
    if (!ok) {
        for (const auto& f : e.files) remove_file(f);
        e.files.clear();
        release_refs(e);   // the chunks only this write made stay as ghosts
        e.bytes = e.manifest_bytes = 0;
        if (storage_failed && pool_ && !options_.shared) {
            // the disk itself is full (not our limit): the ghosts are of no use if the free-space preflight then refuses
            // every retry - free them all now, oldest first (never a referenced or pinned chunk)
            uint64_t freed = 0;
            const size_t n = pool_->evict_ghosts(UINT64_MAX, {}, &freed);
            {
                std::lock_guard<std::mutex> lock(m_);
                stats_.chunks_swept += n;
                stats_.swept_bytes += freed;
            }
            log(fmt("WARNING [disk-full]: the disk is out of space (the free-space reserve is %.0f MiB): freed %zu unreferenced "
                    "chunk%s (%.0f MiB); the storage limit is %.0f MiB, so free space elsewhere or lower --conversation-disk-mib",
                    mib(options_.min_free_bytes), n, n == 1 ? "" : "s", mib(freed), mib(options_.budget_bytes)));
        }
        return false;
    }
    if (use_chunks) {
        std::lock_guard<std::mutex> lock(m_);
        stats_.chunks_hit += totals.hits;
        stats_.chunks_new += totals.misses;
        stats_.raw_bytes += totals.raw_bytes;
        stats_.dedup_bytes += totals.hit_raw_bytes;
        stats_.new_file_bytes += totals.new_file_bytes;
        stats_.direct_io = totals.direct_io;
        e.written_bytes = totals.new_file_bytes + e.manifest_bytes;
        e.chunk_note = fmt("; chunks %u new, %u hit, %.0f MiB raw, %.0f MiB deduplicated, %.0f MiB new, %s I/O",
                           totals.misses, totals.hits, mib(totals.raw_bytes), mib(totals.hit_raw_bytes),
                           mib(totals.new_file_bytes), totals.direct_io ? "direct" : "buffered");
    }
    return true;
}

namespace {
// `c` is where `live` starts: its tokens lead `live`'s, and so do the images below its end
bool leads(const ConversationCheckpoint& c, const ConversationCheckpoint& live) {
    if (c.ids.size() >= live.ids.size() || !std::equal(c.ids.begin(), c.ids.end(), live.ids.begin())) return false;
    size_t j = 0;
    for (const auto& im : live.imgs) {
        if (im.start >= (int64_t) c.ids.size()) continue;
        if (j == c.imgs.size() || !(c.imgs[j++] == im)) return false;
    }
    return j == c.imgs.size();
}

// `old` adds nothing to `n` (both keys only): the same conversation a turn or more back, whose deepest checkpoint the
// newer chain still holds (the RAM cache's #342 rule: all `old` adds is the tail its client rewrote), or the very
// same live state with no checkpoint `n` lacks.  On the disk also (strata-nvfp4): the same conversation carried on
// past `old`'s deepest checkpoint although `n`'s chain has rotated that point out (6 checkpoints: a long session's
// older copies piled up) - `n`'s live tokens start with it.  Not when that checkpoint is `old`'s only or shallowest
// one (the shared system prompt's root: a sibling conversation starts there too).
template<class E>
bool superseded_by(const E& old, const E& n) {
    if (old.cvec != n.cvec) return false;
    auto held = [&](const ConversationCheckpoint& c) {
        if (same_key(c, n.live)) return true;
        for (const auto& k : n.checkpoints)
            if (same_key(k, c)) return true;
        return false;
    };
    const ConversationCheckpoint* deepest = nullptr;
    const ConversationCheckpoint* shallowest = nullptr;
    for (const auto& c : old.checkpoints) {
        if (!deepest || c.ids.size() > deepest->ids.size()) deepest = &c;
        if (!shallowest || c.ids.size() < shallowest->ids.size()) shallowest = &c;
    }
    if (deepest && !deepest->ids.empty() && held(*deepest)) return true;
    if (deepest && deepest != shallowest && deepest->ids.size() > shallowest->ids.size() && leads(*deepest, n.live))
        return true;
    if (!same_key(old.live, n.live)) return false;
    for (const auto& c : old.checkpoints)
        if (!held(c)) return false;
    return true;
}
}  // namespace

std::string ConversationDiskTier::publish_locked(Entry&& e, size_t& made_room, size_t& dropped) {
    made_room = dropped = 0;
    const uint64_t id = e.id;
    bytes_ += e.bytes;
    entries_.push_back(std::move(e));
    // the older copies of this conversation it supersedes go now that it is complete (never the other way round)
    for (size_t i = 0; i + 1 < entries_.size();) {
        if (entries_[i].id != reading_ && superseded_by(entries_[i], entries_.back())) {
            remove_locked(i, true);
            ++dropped;
            continue;
        }
        ++i;
    }
    // the limit: the oldest go (not the new one, not one being read)
    make_room_locked(id, 0, made_room);
    stats_.superseded += dropped;
    return fmt("on disk %zu, %.0f MiB; %zu removed over the limit, %zu superseded", entries_.size(),
               mib(bytes_ + pool_bytes_all()), made_room, dropped);
}

// The limit loop.  total = the files' bytes (v1) + the manifests + the whole chunk pool, unreferenced chunks included;
// there is no time filter.  Over the limit:
//   A. the OLDEST unreferenced chunks go (ChunkPool::evict_ghosts: least recently released / hit / read first), only as
//      many as are needed - the rest stay as a dedup cache;
//   B. still over: the OLDEST entry goes (a WARNING [disk-limit]: it was still valid, the limit is too small); its
//      chunks no other entry names become ghosts, and A takes them, oldest first, on the next turn.
// An entry that shares all its chunks frees about nothing and the loop goes on; the last entry (the protected one)
// owns whatever is left and is kept, with a WARNING.  Shared folders: other engines' references are invisible, so
// A consults their manifests and leases (ChunkPool::evict_ghosts) and a census adds their chunks to the total.
void ConversationDiskTier::make_room_locked(uint64_t protect_a, uint64_t protect_b, size_t& evicted) {
    evicted = 0;
    const uint64_t limit = options_.budget_bytes;
    auto total = [&]() { return bytes_ + (pool_ ? pool_->bytes() : 0); };
    const bool shared = options_.shared && pool_;
    if (shared) {
        pool_->census();
        sort_by_use_locked();
    }
    const uint64_t demand = total();   // before any eviction
    for (;;) {
        if (shared && evicted) pool_->census();
        if (total() <= limit) break;
        if (pool_) {
            uint64_t freed = 0;
            bool undecided = false;
            std::function<bool(std::set<ChunkKey>&)> named;
            if (shared) named = [this](std::set<ChunkKey>& out) { return named_chunks_locked(out); };
            const size_t n = pool_->evict_ghosts(total() - limit, named, &freed, &undecided);
            stats_.chunks_swept += n;
            stats_.swept_bytes += freed;
            if (total() <= limit) break;
            if (undecided) {
                // the manifests of the other engines cannot be read just now: nothing may be deleted for that reason, and
                // evicting entries would only turn more chunks into ghosts that cannot go either - stop, say so
                warn_limit_locked(LimitEvent::undecided, 0, total(), std::max(demand, total()));
                break;
            }
        }
        size_t victim = 0;
        while (victim < entries_.size() && (entries_[victim].id == protect_a || entries_[victim].id == protect_b ||
                                            entries_[victim].id == reading_)) ++victim;
        if (victim == entries_.size()) {
            // what is left is protected (the entry just written, the one being read) or other engines' ghosts
            warn_limit_locked(LimitEvent::remaining, 0, total(), demand);
            break;
        }
        const Entry& v = entries_[victim];
        const uint64_t held = footprint(v);
        warn_limit_locked(LimitEvent::evicted, (int64_t) v.live.ids.size(), held, std::max(demand, total()));
        ++stats_.forced_evictions;
        ++stats_.evicted;
        stats_.forced_evicted_bytes += held;
        remove_locked(victim, true);
        ++evicted;
    }
}

uint64_t ConversationDiskTier::footprint(const Entry& e) const {
    uint64_t n = e.bytes;
    if (e.chunked)
        for (const auto& r : unique_refs(e.refs)) n += chunk_file_size(r.stored_len);
    return n;
}

void ConversationDiskTier::touch_index(const Entry& e) const {
#ifndef _WIN32
    if (e.index.empty()) return;
    struct timespec ts[2];
    ts[0].tv_sec = 0; ts[0].tv_nsec = UTIME_OMIT;
    ts[1].tv_sec = 0; ts[1].tv_nsec = UTIME_NOW;
    ::utimensat(AT_FDCWD, e.index.c_str(), ts, 0);
#else
    (void) e;
#endif
}

// shared: an entry another engine used (a keep-take touches its index) or published is newer than this list says
void ConversationDiskTier::sort_by_use_locked() {
    if (!options_.shared || entries_.size() < 2) return;
    std::map<uint64_t, int64_t> ns;
    for (const auto& e : entries_) ns[e.id] = e.index.empty() ? INT64_MAX : file_mtime_ns(e.index);
    std::stable_sort(entries_.begin(), entries_.end(), [&](const Entry& a, const Entry& b) { return ns[a.id] < ns[b.id]; });
}

// One line a minute at most (now_(), the tier's own steady clock): the first is immediate, the events in between are
// counted and named in the next line.  The counters (stats_) always count; this only limits the log.
void ConversationDiskTier::warn_limit_locked(LimitEvent what, int64_t tokens, uint64_t entry_bytes, uint64_t demand_bytes) {
    const auto now = now_();
    if (limit_warned_ && now - limit_warn_at_ < std::chrono::seconds(60)) {
        ++limit_warn_pending_;
        return;
    }
    const size_t more = limit_warn_pending_;
    limit_warn_pending_ = 0;
    limit_warned_ = true;
    limit_warn_at_ = now;
    ++stats_.limit_warnings;
    const unsigned long long suggest = (unsigned long long) std::ceil(mib(demand_bytes) * 1.25);
    const std::string since = more ? fmt(" (%zu more since the last warning)", more) : std::string();
    std::string line;
    switch (what) {
    case LimitEvent::evicted:
        line = fmt("WARNING [disk-limit]: the conversation disk limit is too small: evicted a parked conversation that was "
                   "still referenced (%lld tokens, %.0f MiB) to stay within %.0f MiB; %.0f MiB were in use or needed%s. "
                   "Raise --conversation-disk-mib (e.g. to %llu).",
                   (long long) tokens, mib(entry_bytes), mib(options_.budget_bytes), mib(demand_bytes), since.c_str(), suggest);
        break;
    case LimitEvent::rejected:
        line = fmt("WARNING [disk-limit]: the conversation disk limit is too small: rejected a conversation of %lld tokens "
                   "(%.0f MiB) that does not fit the %.0f MiB limit alone%s. Raise --conversation-disk-mib (e.g. to %llu).",
                   (long long) tokens, mib(entry_bytes), mib(options_.budget_bytes), since.c_str(), suggest);
        break;
    case LimitEvent::undecided:
        line = fmt("WARNING [disk-limit]: the conversation disk is %.0f MiB over its %.0f MiB limit, but a manifest of another "
                   "engine in the shared folder could not be read, so no chunk may be deleted and nothing was evicted%s. It is "
                   "retried at the next write (a manifest that stays unreadable is given up on after 3 tries).",
                   mib(entry_bytes > options_.budget_bytes ? entry_bytes - options_.budget_bytes : 0), mib(options_.budget_bytes),
                   since.c_str());
        break;
    case LimitEvent::remaining:
        line = fmt("WARNING [disk-limit]: the conversation disk limit is too small: the conversation just written or being "
                   "read (%.0f MiB in use) alone exceeds the %.0f MiB limit%s. Raise --conversation-disk-mib (e.g. to %llu).",
                   mib(entry_bytes), mib(options_.budget_bytes), since.c_str(), suggest);
        break;
    }
    warn_lines_.push_back(std::move(line));
}

void ConversationDiskTier::flush_warnings() {
    std::vector<std::string> lines;
    {
        std::lock_guard<std::mutex> lock(m_);
        lines.swap(warn_lines_);
    }
    for (const auto& l : lines) log(l);
}

// Shared: every chunk any manifest of this identity in the folder names - the entries this engine holds from their
// references, the others' parsed from their manifests (cached per index file: an index never changes after it is
// published).  False when it cannot be told (a manifest of an index that still exists does not parse): the caller then
// deletes nothing.  A manifest that is gone with its index (another engine removed the entry meanwhile) is skipped.
bool ConversationDiskTier::named_chunks_locked(std::set<ChunkKey>& out) {
    std::set<std::string> held;
    for (const auto& e : entries_) {
        if (!e.index.empty()) held.insert(e.index);
        for (const auto& r : e.refs) out.insert(r.key);
    }
    std::error_code ec;
    const SessionFileIdentity fid = file_identity();
    std::set<std::string> present;
    bool ok = true;
    for (fs::directory_iterator it(upath(options_.dir), ec), end; !ec && it != end; it.increment(ec)) {
        const std::string name = ustr(it->path().filename());
        if (it->path().extension() != ".idx" || name.rfind(file_prefix(), 0) != 0) continue;
        const std::string path = ustr(it->path());
        present.insert(path);
        if (held.count(path)) continue;
        auto cached = named_cache_.find(path);
        if (cached == named_cache_.end()) {
            std::vector<ChunkKey> keys;
            Entry t;
            bool transient = false;
            // (a definite failed read: another identity's, its own pool, or torn: names nothing here)
            if (read_index(path, t, &transient)) {
                for (const auto& f : t.files) {
                    std::vector<ChunkRef> r;
                    uint64_t mb = 0;
                    std::string why;
                    bool tr = false;
                    if (!is_chunked_manifest(f)) continue;
                    if (chunked_manifest_refs(f, fid, r, mb, why, &tr)) {
                        for (const auto& x : r) keys.push_back(x.key);
                    } else {
                        std::error_code xec;
                        if (!fs::exists(upath(path), xec)) { keys.clear(); break; }   // the entry was removed while it was read
                        if (tr) { transient = true; break; }   // an I/O or resource error: cannot tell, and never cached
                        // a manifest that was read and is invalid under a live index (nobody can use that entry) must not
                        // block every eviction for good: after 3 passes in a row it is taken as naming nothing - the keys
                        // of the entry's other files stay named
                        if (++named_fail_[path] < 3) { ok = false; break; }
                    }
                }
            }
            if (transient) ok = false;   // cannot tell now: no chunk is deleted this time, nothing is cached
            if (!ok) continue;
            named_fail_.erase(path);
            cached = named_cache_.emplace(path, std::move(keys)).first;
        }
        out.insert(cached->second.begin(), cached->second.end());
    }
    if (ec) return false;
    for (auto it = named_cache_.begin(); it != named_cache_.end();)
        if (present.count(it->first)) ++it;
        else it = named_cache_.erase(it);
    for (auto it = named_fail_.begin(); it != named_fail_.end();)
        if (present.count(it->first)) ++it;
        else it = named_fail_.erase(it);
    return ok;
}

void ConversationDiskTier::write(Job& job) {
    if (options_.before_write) options_.before_write();
    const size_t tokens = job.entry.live.ids.size();
    size_t expired = 0;
    {
        std::lock_guard<std::mutex> lock(m_);
        expired = expire_locked();
    }
    const auto t0 = std::chrono::steady_clock::now();
    Entry e = std::move(job.entry);
    std::string error;
    const int64_t as_of = live_skip_tokens(job.image, !job.flush);   // RAM pressure (flush) keeps every token
    const bool ok = write_files(e, job.image, nullptr, {}, error, job.flush, as_of);
    job.image = SavedConversation{};   // the RAM goes back now, written or not
    const double ms = ms_since(t0);
    if (!ok) {
        {
            std::lock_guard<std::mutex> lock(m_);
            ++stats_.write_failed;
        }
        log(fmt("dropped %zu tokens: the write failed after %.0f ms (%s)", tokens, ms, error.c_str()));
        return;
    }
    e.at = now_();
    const uint64_t b = e.chunked ? e.written_bytes : e.bytes;
    const size_t nfiles = e.files.size();
    const std::string note = e.chunk_note + (as_of > 0 ? skip_note(tokens, as_of) : std::string());
    std::string tail;
    {
        std::lock_guard<std::mutex> lock(m_);
        if (as_of > 0) { ++stats_.live_skipped; stats_.live_skipped_tokens += tokens - (size_t) as_of; }
        ++stats_.written;
        stats_.bytes_written += b;
        stats_.write_ms += ms;
        size_t made_room = 0, dropped = 0;
        tail = publish_locked(std::move(e), made_room, dropped);
    }
    flush_warnings();
    log(fmt("wrote %zu tokens: %.0f MiB in %zu file%s, %.0f ms (%.0f MB/s); %s; %zu expired%s", tokens, mib(b), nfiles,
            nfiles == 1 ? "" : "s", ms, mbps(b, ms), tail.c_str(), expired, note.c_str()));
}

bool ConversationDiskTier::write_now(const SavedConversation& image, const std::vector<SessionKvSource>* sources,
                                     const std::function<bool()>& cancel, uint64_t& id, std::string& error,
                                     bool* cancelled, bool background, int64_t* as_of_out) {
    id = 0;
    if (cancelled) *cancelled = false;
    if (as_of_out) *as_of_out = 0;
    if (!open_) { error = "the disk tier is off"; return false; }
    Entry e;
    const int64_t as_of = live_skip_tokens(image, background);   // the keys are those of the image that is written
    if (as_of_out) *as_of_out = as_of;
    e.live = key_of(image.live);
    for (const auto& c : image.checkpoints) e.checkpoints.push_back(key_of(c));
    if (as_of > 0) {
        e.checkpoints.clear();
        for (const auto& c : image.checkpoints) {
            if ((int64_t) c.ids.size() < as_of) e.checkpoints.push_back(key_of(c));
            else if ((int64_t) c.ids.size() == as_of) { e.live = key_of(c); e.checkpoints.push_back(key_of(c)); }
        }
    }
    e.cvec = image.cvec;
    const uint64_t estimate = image.bytes() + sources_bytes(sources);
    const size_t tokens = image.live.ids.size();
    {
        std::unique_lock<std::mutex> lock(m_);
        // already there with exactly these keys: nothing to write
        for (const auto& x : entries_)
            if (x.cvec == e.cvec && same_key(x.live, e.live) && x.checkpoints.size() == e.checkpoints.size() &&
                std::equal(x.checkpoints.begin(), x.checkpoints.end(), e.checkpoints.begin(), same_key)) {
                id = x.id;
                // a write hit is a use: the entry is the newest in the LRU order (and in its index file's mtime)
                const size_t at = find_locked(x.id);
                if (at != entries_.size()) {
                    Entry moved = std::move(entries_[at]);
                    entries_.erase(entries_.begin() + (std::ptrdiff_t) at);
                    moved.at = now_();
                    touch_index(moved);
                    entries_.push_back(std::move(moved));
                }
                return true;
            }
        if (!(options_.chunked && pool_) && estimate > options_.budget_bytes) {   // chunked: judged on the real bytes after the write
            ++stats_.write_failed;
            ++stats_.limit_rejected;
            error = fmt("%.0f MiB is over the storage limit (%.0f MiB; raise --conversation-disk-mib)", mib(estimate),
                        mib(options_.budget_bytes));
            warn_limit_locked(LimitEvent::rejected, (int64_t) tokens, estimate, estimate);
            lock.unlock();
            flush_warnings();
            return false;
        }
        e.id = ++next_id_;
    }
    const auto t0 = std::chrono::steady_clock::now();
    bool was_cancelled = false;
    auto ask = [&]() -> bool {
        if (cancel && cancel()) { was_cancelled = true; return true; }
        return false;
    };
    const bool ok = write_files(e, image, sources, ask, error, false, as_of);
    const double ms = ms_since(t0);
    if (!ok) {
        std::lock_guard<std::mutex> lock(m_);
        if (was_cancelled) ++stats_.sync_cancelled;
        else ++stats_.write_failed;
        if (cancelled) *cancelled = was_cancelled;
        if (was_cancelled) error = "cancelled";
        return false;
    }
    e.at = now_();
    const uint64_t b = e.chunked ? e.written_bytes : e.bytes;
    const size_t nfiles = e.files.size();
    const std::string note = e.chunk_note + (as_of > 0 ? skip_note(tokens, as_of) : std::string());
    std::string tail;
    {
        std::lock_guard<std::mutex> lock(m_);
        if (as_of > 0) { ++stats_.live_skipped; stats_.live_skipped_tokens += tokens - (size_t) as_of; }
        ++stats_.synced;
        stats_.bytes_written += b;
        stats_.write_ms += ms;
        id = e.id;
        size_t made_room = 0, dropped = 0;
        tail = publish_locked(std::move(e), made_room, dropped);
    }
    flush_warnings();
    log(fmt("saved %zu tokens while idle: %.0f MiB in %zu file%s, %.0f ms (%.0f MB/s); %s%s", tokens, mib(b), nfiles,
            nfiles == 1 ? "" : "s", ms, mbps(b, ms), tail.c_str(), note.c_str()));
    return true;
}

size_t ConversationDiskTier::rescan() {
    if (!open_ || !options_.shared) return 0;
    std::lock_guard<std::mutex> lock(m_);
    std::error_code ec;
    std::set<std::string> present;
    std::vector<std::string> fresh;
    for (fs::directory_iterator it(upath(options_.dir), ec), end; !ec && it != end; it.increment(ec)) {
        const std::string name = ustr(it->path().filename());
        if (it->path().extension() != ".idx" || name.rfind(file_prefix(), 0) != 0) continue;
        const std::string path = ustr(it->path());
        present.insert(path);
        if (!seen_index_.count(path)) fresh.push_back(path);
    }
    if (ec) return 0;
    // the entries whose index another engine removed (the budget, a newer copy, a #879 clear) leave this list; their
    // files are that engine's to delete
    size_t gone = 0;
    for (size_t i = 0; i < entries_.size();) {
        if (!entries_[i].index.empty() && !present.count(entries_[i].index) && entries_[i].id != reading_) {
            bytes_ -= entries_[i].bytes;
            release_refs(entries_[i]);
            entries_.erase(entries_.begin() + (std::ptrdiff_t) i);
            ++gone;
            continue;
        }
        ++i;
    }
    size_t in = 0;
    for (const auto& path : fresh) {
        seen_index_.insert(path);
        Entry e;
        if (!read_index(path, e) || !bind_chunks(e)) continue;   // another identity, torn, or a chunk is missing
        e.id = ++next_id_;
        e.at = now_();
        e.idx_ns = file_mtime_ns(path);
        bytes_ += e.bytes;
        entries_.push_back(std::move(e));
        ++in;
    }
    if (in) sort_by_use_locked();   // (an index another engine wrote earlier than the last one looked at is not the newest)
    if (pool_) pool_->census();
    if (in || gone)
        log(fmt("shared folder: %zu conversation%s from other engines, %zu removed by them; on disk %zu (%.0f MiB)", in,
                in == 1 ? "" : "s", gone, entries_.size(), mib(bytes_ + pool_bytes_all())));
    return in;
}

bool ConversationDiskTier::has(uint64_t id) const {
    if (id == 0) return false;
    std::lock_guard<std::mutex> lock(m_);
    if (find_locked(id) != entries_.size()) return true;
    for (const auto& j : queue_)
        if (j.entry.id == id) return true;
    return writing_ && writing_entry_.id == id;
}

void ConversationDiskTier::set_busy(bool busy) {
    {
        std::lock_guard<std::mutex> lock(m_);
        busy_ = busy;
    }
    if (!busy) cv_.notify_all();
}

bool ConversationDiskTier::take(uint64_t id, int64_t tokens, SavedConversation& out, std::string& error,
                                const std::function<bool(uint64_t, std::string&)>& admit, bool keep, int64_t upto) {
    out = SavedConversation{};
    const auto t0 = std::chrono::steady_clock::now();
    std::unique_lock<std::mutex> lock(m_);
    if (!open_) { error = "the disk tier is off"; return false; }
    // still queued: back from RAM, no disk at all
    for (size_t i = 0; i < queue_.size(); ++i)
        if (queue_[i].entry.id == id) {
            out = std::move(queue_[i].image);
            const uint64_t b = queue_[i].entry.bytes;
            queue_.erase(queue_.begin() + (std::ptrdiff_t) i);
            ++stats_.ram_hits;
            stats_.tokens_back += (uint64_t) std::max<int64_t>(tokens, 0);
            lock.unlock();
            log(fmt("RAM hit: %lld tokens back from the write queue (%.0f MiB, no disk) in %.1f ms",
                    (long long) tokens, mib(b), ms_since(t0)));
            log(summary());
            return true;
        }
    // being written: its file is complete when the write ends
    double waited = 0;
    if (writing_ && writing_entry_.id == id) {
        const auto w0 = std::chrono::steady_clock::now();
        ++urgent_;
        cv_.wait(lock, [&] { return !(writing_ && writing_entry_.id == id); });
        --urgent_;
        waited = ms_since(w0);
    }
    const size_t index = find_locked(id);
    if (index == entries_.size()) {
        ++stats_.read_failed;
        error = "the entry is gone (its write failed, or it expired)";
        return false;
    }
    // A write cut at a checkpoint (live-skip) rewrites the keys of an entry that best() reported with its full live
    // state: the image on the disk then holds fewer tokens than the match said, and a request that resumes at the
    // match would run past the restored K/V.  A miss - the entry stays, as the checkpoint it is.
    if (tokens > (int64_t) entries_[index].live.ids.size()) {
        error = fmt("the entry holds %zu tokens, fewer than the %lld matched (written as of a checkpoint)",
                    entries_[index].live.ids.size(), (long long) tokens);
        lock.unlock();
        log(fmt("disk miss: %s", error.c_str()));
        return false;
    }
    const Entry want = entries_[index];
    const std::vector<std::string>& files = want.files;
    // a prefix read: the request reuses a checkpoint shorter than the entry's live state (chunked entries, kept)
    int64_t cut = 0;
    if (keep && upto > 0 && want.chunked && pool_ && files.size() == 1 && upto < (int64_t) want.live.ids.size())
        for (const auto& c : want.checkpoints)
            if ((int64_t) c.ids.size() == upto) cut = upto;
    // the chunks stay pinned while they are read: the entry leaves the list now and drops its references, but a pinned
    // chunk is never swept; after the read the pins go and the chunks stay as ghosts until the limit needs the room (a
    // continuation of this conversation is written next and hits them)
    std::vector<ChunkRef> pins;
    std::unique_ptr<ChunkLease> lease;   // shared: the entry's chunks are announced before its index goes (see below)
    if (want.chunked && pool_ && !keep) {
        pins = want.refs;
        pool_->ref(pins.data(), pins.size());
        if (options_.shared) {
            lease = pool_->begin_lease();
            if (lease) lease->announce(pins.data(), pins.size());
        }
    }
    if (keep) {
        // stays, as the newest: the limit takes the least recently used first (also after a restart: the index's mtime)
        Entry moved = std::move(entries_[index]);
        entries_.erase(entries_.begin() + (std::ptrdiff_t) index);
        moved.at = now_();
        touch_index(moved);
        entries_.push_back(std::move(moved));
        reading_ = id;
    } else {
        remove_locked(index, false);   // the entry goes now; its files are deleted after the read, whatever it gives
    }
    lock.unlock();
    const SessionFileIdentity fid = file_identity();
    SessionReadLimits limits;
    limits.admit = admit;
    uint64_t bytes = 0;
    bool ok = true;
    for (size_t k = 0; ok && k < files.size(); ++k) {
        SavedConversation part;
        size_t n = 0;
        if (want.chunked && pool_) {
            uint64_t raw = 0;
            ok = chunked_read(*pool_, files[k], fid, part, n, raw, error, limits, ChunkedReadOptions{cut, options_.io_workers});
            bytes += raw;
        } else {
            ok = session_file_read(files[k], fid, part, n, error, limits);
        }
        bytes += n;
        if (!ok) break;
        if (k == 0) out = std::move(part);
        else out.stage_images.push_back(std::move(part));
    }
    // a prefix image holds the checkpoint it was cut at as its live state, and the checkpoints up to it
    ConversationCheckpoint want_live = want.live;
    std::vector<ConversationCheckpoint> want_checks = want.checkpoints;
    if (cut > 0) {
        want_checks.clear();
        for (const auto& c : want.checkpoints)
            if ((int64_t) c.ids.size() < cut) want_checks.push_back(c);
            else if ((int64_t) c.ids.size() == cut) { want_live = c; want_checks.push_back(c); }
    }
    if (ok && !holds(out, want_live, want_checks, want.cvec)) {
        ok = false;
        error = "the files hold another conversation than their index";
    }
    const double ms = ms_since(t0);
    lock.lock();
    reading_ = 0;
    if (!pins.empty()) pool_->unref(pins.data(), pins.size());
    lease.reset();   // (the pins are the ghosts' protection from here on, and the files are gone)
    if (!keep) {
        for (const auto& f : files) remove_file(f);
    } else if (!ok) {
        // a kept entry whose files do not read back is of no use to anyone: it goes
        const size_t at = find_locked(id);
        if (at != entries_.size()) remove_locked(at, true);
    }
    if (!ok) {
        ++stats_.read_failed;
        lock.unlock();
        out = SavedConversation{};
        log(fmt("read of %lld tokens failed after %.0f ms (%s)", (long long) tokens, ms, error.c_str()));
        return false;
    }
    ++stats_.disk_hits;
    if (keep) ++stats_.kept;
    stats_.bytes_read += bytes;
    if (cut > 0) { ++stats_.prefix_reads; stats_.prefix_raw_bytes += bytes; }
    stats_.read_ms += ms - waited;
    stats_.tokens_back += (uint64_t) std::max<int64_t>(tokens, 0);
    lock.unlock();
    log(fmt("disk hit: %lld tokens, %.0f MiB in %zu file%s, %.0f ms (%.0f MB/s)%s%s%s", (long long) tokens, mib(bytes),
            files.size(), files.size() == 1 ? "" : "s", ms - waited, mbps(bytes, ms - waited),
            waited > 0 ? fmt(", after %.0f ms waiting for its write", waited).c_str() : "",
            cut > 0 ? fmt("; prefix read of %lld of %zu tokens", (long long) cut, want.live.ids.size()).c_str() : "",
            keep ? "; kept on the disk" : ""));
    log(summary());
    return true;
}

bool ConversationDiskTier::pending() const {
    std::lock_guard<std::mutex> lock(m_);
    return !queue_.empty() || writing_;
}

double ConversationDiskTier::wait_idle() {
    std::unique_lock<std::mutex> lock(m_);
    if (queue_.empty() && !writing_) return 0;
    const auto t0 = std::chrono::steady_clock::now();
    const size_t depth = queue_.size() + (writing_ ? 1 : 0);
    ++urgent_;   // the writer starts its jobs even while the engine is busy
    cv_.notify_all();
    cv_.wait(lock, [&] { return queue_.empty() && !writing_; });
    --urgent_;
    const double ms = ms_since(t0);
    ++stats_.waits;
    stats_.wait_ms += ms;
    lock.unlock();
    log(fmt("waited %.0f ms for %zu queued write%s (their RAM was needed)", ms, depth, depth == 1 ? "" : "s"));
    return ms;
}

size_t ConversationDiskTier::size() const {
    std::lock_guard<std::mutex> lock(m_);
    return entries_.size();
}

uint64_t ConversationDiskTier::bytes() const {
    std::lock_guard<std::mutex> lock(m_);
    return bytes_ + pool_bytes_all();
}

ConversationDiskStats ConversationDiskTier::stats() const {
    std::lock_guard<std::mutex> lock(m_);
    return stats_;
}

size_t ConversationDiskTier::clear() {
    std::unique_lock<std::mutex> lock(m_);
    const size_t queued = queue_.size();
    queue_.clear();
    cv_.wait(lock, [&] { return !writing_; });
    const size_t n = queued + entries_.size();
    while (!entries_.empty()) remove_locked(0, true);
    if (pool_ && !options_.shared) {   // #879: the chunks too, at once (a shared pool's are other engines' as well)
        std::string why;
        pool_->remove_all();
        pool_->open(why);
    }
    lock.unlock();
    if (n) log(fmt("removed all %zu conversations (#879)", n));
    return n;
}

std::string ConversationDiskTier::summary() const {
    std::lock_guard<std::mutex> lock(m_);
    const auto& s = stats_;
    const std::string chunks = !pool_ ? std::string() :
        fmt("; chunks: %llu new, %llu hit, %.0f MiB written, %.0f MiB deduplicated of %.0f MiB, %llu evicted for the limit (%.0f MiB), "
            "pool %zu (%.0f MiB, %zu unreferenced = %.0f MiB), %s I/O",
            (unsigned long long) s.chunks_new, (unsigned long long) s.chunks_hit, mib(s.new_file_bytes),
            mib(s.dedup_bytes), mib(s.raw_bytes), (unsigned long long) s.chunks_swept, mib(s.swept_bytes), pool_->count(),
            mib(pool_->bytes()), pool_->ghost_count(), mib(pool_->ghost_bytes()), s.direct_io ? "direct" : "buffered") +
        fmt("; %llu prefix reads (%.0f MiB), %llu writes as of a checkpoint (%llu tokens left out)",
            (unsigned long long) s.prefix_reads, mib(s.prefix_raw_bytes), (unsigned long long) s.live_skipped,
            (unsigned long long) s.live_skipped_tokens);
    return fmt("totals: %zu queued, %zu written + %zu saved while idle (%.0f MiB, %.0f MB/s; %zu given way to a "
               "request), %zu RAM hits, %zu disk hits (%zu kept; %.0f MiB, %.0f MB/s), %llu prompt tokens given back, "
               "%zu waits for RAM (%.0f ms), %zu expired, %zu removed over the limit (%zu forced%s, %llu rejected), %zu superseded, "
               "%zu/%zu writes/reads failed; on disk %zu (%.0f MiB of the %.0f MiB limit), queue %zu%s",
               s.queued, s.written, s.synced, mib(s.bytes_written), mbps(s.bytes_written, s.write_ms),
               s.sync_cancelled, s.ram_hits, s.disk_hits, s.kept, mib(s.bytes_read), mbps(s.bytes_read, s.read_ms),
               (unsigned long long) s.tokens_back, s.waits, s.wait_ms, s.expired, s.evicted,
               (size_t) s.forced_evictions, s.forced_evictions ? ": raise --conversation-disk-mib" : "",
               (unsigned long long) s.limit_rejected, s.superseded,
               s.write_failed, s.read_failed, entries_.size(), mib(bytes_ + pool_bytes_all()), mib(options_.budget_bytes),
               queue_.size() + (writing_ ? 1 : 0), chunks.c_str());
}

bool conversation_disk_skip_live(LivePolicy policy, int64_t live_tokens, int64_t deepest, int64_t gap) {
    if (policy == LivePolicy::always) return false;
    return deepest > 0 && deepest < live_tokens && live_tokens - deepest <= std::max<int64_t>(gap, 0);
}

bool conversation_disk_parse_live(const std::string& text, LivePolicy& policy, std::string& error) {
    if (text == "always") { policy = LivePolicy::always; return true; }
    if (text == "skip") { policy = LivePolicy::skip; return true; }
    if (text == "auto") { policy = LivePolicy::automatic; return true; }
    error = "the live policy is always, skip or auto";
    return false;
}

bool conversation_disk_parse_codec(const std::string& text, ChunkCodec& codec, int& level, std::string& error) {
    if (text == "raw") { codec = ChunkCodec::raw; level = 1; return true; }
    if (text == "zstd" || text.rfind("zstd:", 0) == 0) {
        int lv = 1;
        if (text.size() > 4) {
            const std::string num = text.substr(5);
            if (num.empty() || num.size() > 2 || num.find_first_not_of("0123456789") != std::string::npos) {
                error = "the zstd level is a number from 1 to 19";
                return false;
            }
            lv = std::atoi(num.c_str());
            if (lv < 1 || lv > 19) { error = "the zstd level is a number from 1 to 19"; return false; }
        }
        codec = ChunkCodec::zstd;
        level = lv;
        return true;
    }
    error = "the codec is raw, zstd or zstd:LEVEL";
    return false;
}

}  // namespace strata::core
