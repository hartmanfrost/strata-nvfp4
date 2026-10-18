// CPU-only tests for the chunked disk format (include/strata/core/conversation_chunks.hpp): chunk objects, the pool,
// slab geometry, the manifest and its reader's limits.  Built with -DSTRATA_BUILD_CONVERSATION_TESTS=ON; no CUDA.
// Run it twice for the two I/O modes: as is (O_DIRECT where the filesystem takes it) and STRATA_SESSION_BUFFERED=1.
#include "strata/core/conversation_chunks.hpp"

#include <algorithm>
#include <atomic>
#include <memory>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <utility>
#include <vector>
#ifndef _WIN32
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

using namespace strata::core;
namespace fs = std::filesystem;

namespace {
int checks = 0;
void check(bool ok, const char* label) {
    ++checks;
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", label); std::exit(1); }
}

bool g_low_entropy = false;   // make_image / fill: bytes of 3 bits of entropy, so a codec has something to find
uint8_t hb(uint64_t seed, uint64_t i) {
    uint64_t x = (i + 1) * 0x9E3779B97F4A7C15ull ^ (seed * 0xD6E8FEB86659FD93ull);
    x ^= x >> 32; x *= 0xD6E8FEB86659FD93ull; x ^= x >> 32;
    return g_low_entropy ? (uint8_t) (x & 7) : (uint8_t) x;
}
ConversationBuffer fill_buf(size_t n, uint64_t seed) {
    ConversationBuffer b;
    b.resize(n);
    b.visit(0, n, [&](uint8_t* p, size_t c, size_t at) {
        for (size_t i = 0; i < c; ++i) p[i] = hb(seed, at + i);
        return true;
    });
    return b;
}
std::vector<uint8_t> fill_vec(size_t n, uint64_t seed) {
    std::vector<uint8_t> v(n);
    for (size_t i = 0; i < n; ++i) v[i] = hb(seed, i);
    return v;
}

// A small image with the real structure: two main layers (K, V, k_scale, pooled rows) and a draft layer (no pooled
// part), cells = ceil(n/4)*4, pooled rows = n/4 + 1 (the last is the spare row), all bytes a function of (seed, index)
// so an image that continues another one shares its prefix bytes.
struct Spec {
    int64_t n = 100;
    uint64_t seed = 1;
    size_t live_gdn = (4u << 20) + 4097;       // two slices
    std::vector<int64_t> cps = {25, 50};       // checkpoint token counts (gdn: one slice each)
    size_t cp_gdn = 1000;
    bool draft = true, cvec = false;
    double rope = 1.0;
    uint64_t pad_seed = 0;                     // garbage in the padding cells and the spare row
};
constexpr size_t kMainUnit[4] = {8, 4, 2, 0}, kDraftUnit[4] = {8, 8, 2, 2};
constexpr size_t kRowBytes = 32;

ConversationCheckpoint make_cp(int64_t tokens, size_t gdn, uint64_t gseed) {
    ConversationCheckpoint c;
    for (int64_t i = 0; i < tokens; ++i) c.ids.push_back(int32_t(1000 + i));
    c.imgs = {{3, 0x1234567890abcdefull ^ uint64_t(tokens)}};
    c.gdn = fill_vec(gdn, gseed + (uint64_t) tokens);
    c.ple = fill_vec(77, 5 + tokens);
    c.tails = fill_vec(301, 6 + tokens);
    c.dead = fill_vec(64, 7 + tokens);
    c.block_pos = fill_vec(8, 8 + tokens);
    c.used = 42 + (uint64_t) tokens;
    return c;
}

SavedConversation make_image(const Spec& sp) {
    SavedConversation s;
    for (size_t i = 0; i < s.geometry.size(); ++i) s.geometry[i] = int64_t(100 + i);
    s.layer_lo = 0; s.layer_hi = 48;
    s.cvec = sp.cvec;
    s.rope = sp.rope;
    s.live = make_cp(sp.n, sp.live_gdn, sp.seed * 77);
    for (int64_t t : sp.cps) s.checkpoints.push_back(make_cp(t, sp.cp_gdn, sp.seed * 77));
    const int64_t cells = (sp.n + 3) / 4 * 4;
    for (int l = 0; l < (sp.draft ? 3 : 2); ++l) {
        const bool draft = l == 2;
        const size_t* u = draft ? kDraftUnit : kMainUnit;
        ConversationKv kv;
        kv.format = draft ? 1 : 2; kv.cells = cells; kv.heads = 2; kv.head_dim = 256; kv.page_size = 4; kv.idx_dim = 8;
        kv.pooled_rows = draft ? 0 : sp.n / 4 + 1;
        const uint64_t ls = sp.seed * 1000 + (uint64_t) l * 10;
        kv.k = fill_buf((size_t) cells * u[0], ls + 0);
        kv.v = fill_buf((size_t) cells * u[1], ls + 1);
        kv.k_scale = fill_buf((size_t) cells * u[2], ls + 2);
        kv.v_scale = fill_buf((size_t) cells * u[3], ls + 3);
        kv.pooled = fill_buf((size_t) kv.pooled_rows * kRowBytes, ls + 4);
        if (sp.pad_seed) {
            // garbage where a real image has stale bytes: the padding cells [n, cells) and the spare row
            auto scribble = [&](ConversationBuffer& b, size_t from, size_t to) {
                b.visit(from, to - from, [&](uint8_t* p, size_t c, size_t at) {
                    for (size_t i = 0; i < c; ++i) p[i] = hb(sp.pad_seed, at + i);
                    return true;
                });
            };
            scribble(kv.k, (size_t) sp.n * u[0], kv.k.size());
            scribble(kv.v, (size_t) sp.n * u[1], kv.v.size());
            if (kv.pooled_rows) scribble(kv.pooled, (size_t) (kv.pooled_rows - 1) * kRowBytes, kv.pooled.size());
        }
        s.kv.push_back(std::move(kv));
    }
    return s;
}

bool same_cp(const ConversationCheckpoint& a, const ConversationCheckpoint& b) {
    return a.ids == b.ids && a.imgs == b.imgs && a.gdn == b.gdn && a.ple == b.ple && a.tails == b.tails &&
           a.dead == b.dead && a.block_pos == b.block_pos && a.used == b.used && b.stage_parts.empty();
}
bool same(const SavedConversation& a, const SavedConversation& b) {
    if (a.geometry != b.geometry || a.layer_lo != b.layer_lo || a.layer_hi != b.layer_hi || a.cvec != b.cvec ||
        a.rope != b.rope || !same_cp(a.live, b.live) || a.checkpoints.size() != b.checkpoints.size() ||
        a.kv.size() != b.kv.size()) return false;
    for (size_t i = 0; i < a.checkpoints.size(); ++i)
        if (!same_cp(a.checkpoints[i], b.checkpoints[i])) return false;
    for (size_t i = 0; i < a.kv.size(); ++i) {
        const auto& x = a.kv[i]; const auto& y = b.kv[i];
        if (x.format != y.format || x.cells != y.cells || x.heads != y.heads || x.head_dim != y.head_dim ||
            x.page_size != y.page_size || x.pooled_rows != y.pooled_rows || x.idx_dim != y.idx_dim ||
            !(x.k == y.k) || !(x.v == y.v) || !(x.k_scale == y.k_scale) || !(x.v_scale == y.v_scale) ||
            !(x.pooled == y.pooled)) return false;
    }
    return true;
}

std::vector<SessionKvSource> sources_of(const SavedConversation& img) {
    std::vector<SessionKvSource> out;
    for (const auto& k : img.kv) {
        SessionKvSource s;
        s.format = k.format; s.cells = k.cells; s.heads = k.heads; s.head_dim = k.head_dim;
        s.page_size = k.page_size; s.pooled_rows = k.pooled_rows; s.idx_dim = k.idx_dim;
        s.sizes = {k.k.size(), k.v.size(), k.k_scale.size(), k.v_scale.size(), k.pooled.size()};
        const ConversationKv* kv = &k;
        s.read = [kv](size_t part, size_t off, void* dst, size_t n) {
            const ConversationBuffer* b[5] = {&kv->k, &kv->v, &kv->k_scale, &kv->v_scale, &kv->pooled};
            return part < 5 && b[part]->read(dst, off, n);
        };
        out.push_back(std::move(s));
    }
    return out;
}

std::vector<uint8_t> slurp(const fs::path& p) {
    std::ifstream f(p, std::ios::binary);
    return std::vector<uint8_t>((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}
void spit(const fs::path& p, const std::vector<uint8_t>& d) {
    std::ofstream f(p, std::ios::binary | std::ios::trunc);
    f.write((const char*) d.data(), (std::streamsize) d.size());
}
template<class T> T rd(const std::vector<uint8_t>& v, size_t o) { T x; std::memcpy(&x, v.data() + o, sizeof x); return x; }
template<class T> void wr(std::vector<uint8_t>& v, size_t o, T x) { std::memcpy(v.data() + o, &x, sizeof x); }

size_t count_files(const fs::path& dir, const std::string& suffix) {
    size_t n = 0;
    std::error_code ec;
    for (const auto& e : fs::recursive_directory_iterator(dir, ec))
        if (e.is_regular_file() && e.path().filename().string().size() >= suffix.size() &&
            e.path().filename().string().compare(e.path().filename().string().size() - suffix.size(), suffix.size(), suffix) == 0) ++n;
    return n;
}
size_t count_prefixed(const fs::path& dir, const std::string& prefix) {
    size_t n = 0;
    std::error_code ec;
    for (const auto& e : fs::recursive_directory_iterator(dir, ec))
        if (e.is_regular_file() && e.path().filename().string().rfind(prefix, 0) == 0) ++n;
    return n;
}

const SessionFileIdentity kId{0x1111222233334444ull, 0x5555666677778888ull};

// this process's folders only: a second run sharing the temp directory must not lose its own to this one's cleanup
std::string own_prefix() { return "strata-chunks-test-" + std::to_string((long long) ::getpid()) + "-"; }

struct Fx {
    fs::path root, pool_dir, man;
    std::unique_ptr<ChunkPool> pool;
    int64_t clock = 1000000;
    std::vector<std::string> steps;
    std::mutex steps_mu;   // the pool calls the hook from every writer thread
    std::function<int(const char*)> fault;
    ChunkCodec codec = ChunkCodec::raw;
    int level = 1;
    explicit Fx(const std::string& name, bool direct = true, int64_t grace = 600) {
        root = fs::temp_directory_path() / (own_prefix() + name + "-" +
               std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        fs::create_directories(root);
        pool_dir = root / "strata-pool-test";
        man = root / "manifest.bin";
        make_pool(direct, grace);
    }
    void make_pool(bool direct, int64_t grace = 600, uint64_t min_free = 0) {
        ChunkPoolOptions o;
        o.dir = pool_dir.string();
        o.direct = direct;
        o.grace_s = grace;
        o.min_free_bytes = min_free;
        o.codec = codec;
        o.zstd_level = level;
        o.clock_s = [this] { return clock; };
        o.fault = [this](const char* s) { { std::lock_guard<std::mutex> g(steps_mu); steps.push_back(s); } return fault ? fault(s) : 0; };
        pool = std::make_unique<ChunkPool>(o);
        std::string e;
        check(pool->open(e), "pool opens");
    }
    ~Fx() { std::error_code ec; fs::remove_all(root, ec); }
    std::string path(const std::string& leaf) const { return (root / leaf).string(); }
};

struct W {
    std::vector<ChunkRef> refs;
    ChunkedWriteStats stats;
    std::string error;
    SessionStatus st;
};
bool write_to(Fx& fx, const std::string& path, const SavedConversation& img, W& w, uint32_t T = 512,
              ChunkedWriteOptions o = {}, const std::vector<SessionKvSource>* sources = nullptr) {
    o.chunk_tokens = T;
    return chunked_write(*fx.pool, path, img, sources, kId, o, w.refs, w.stats, w.error, &w.st);
}
bool read_from(Fx& fx, const std::string& path, SavedConversation& out, std::string& error,
               const SessionReadLimits& lim = {}, SessionStatus* st = nullptr) {
    size_t mb = 0;
    uint64_t rb = 0;
    return chunked_read(*fx.pool, path, kId, out, mb, rb, error, lim, {}, st);
}

// expected raw length of slab i for the Spec's geometry (an independent formula)
uint64_t expect_slab(const Spec& sp, uint64_t i, uint32_t T) {
    const uint64_t cells = (uint64_t) ((sp.n + 3) / 4 * 4);
    const uint64_t c = std::min<uint64_t>((i + 1) * T, cells) - std::min<uint64_t>(i * T, cells);
    uint64_t total = 0;
    for (int l = 0; l < (sp.draft ? 3 : 2); ++l) {
        const size_t* u = l == 2 ? kDraftUnit : kMainUnit;
        total += c * (u[0] + u[1] + u[2] + u[3]);
        if (l < 2) {
            const uint64_t data_rows = (uint64_t) (sp.n / 4);   // pooled_rows - 1
            const uint64_t r0 = std::min<uint64_t>(i * (T / 4), data_rows), r1 = std::min<uint64_t>((i + 1) * (T / 4), data_rows);
            total += (r1 - r0) * kRowBytes;
        }
    }
    return total;
}
size_t expect_slices(const Spec& sp) {
    auto sl = [](size_t n) { return (n + kStateSlice - 1) / kStateSlice; };
    return sl(sp.live_gdn) + sp.cps.size() * sl(sp.cp_gdn);
}

// rebuild a manifest file with a new body (header length and hashes recomputed) or with header edits resealed
std::vector<uint8_t> reseal_body(const std::vector<uint8_t>& file, const std::vector<uint8_t>& body) {
    std::vector<uint8_t> out(file.begin(), file.begin() + 64);
    wr<uint64_t>(out, 32, body.size());
    wr<uint64_t>(out, 56, session_hash64(out.data(), 56, 0));
    out.insert(out.end(), body.begin(), body.end());
    const uint64_t h = session_hash64(body.data(), body.size(), 0);
    out.resize(out.size() + 16);
    std::memcpy(out.data() + out.size() - 16, &h, 8);
    std::memcpy(out.data() + out.size() - 8, "STRSEND\x02", 8);
    return out;
}
std::vector<uint8_t> body_of(const std::vector<uint8_t>& file) { return std::vector<uint8_t>(file.begin() + 64, file.end() - 16); }
void reseal_header(std::vector<uint8_t>& file) { wr<uint64_t>(file, 56, session_hash64(file.data(), 56, 0)); }

bool refused(Fx& fx, const std::string& path, const char* expect = nullptr, const SessionReadLimits& lim = {}) {
    SavedConversation out;
    std::string error;
    const bool ok = read_from(fx, path, out, error, lim);
    if (!ok && expect && error.find(expect) == std::string::npos) {
        std::fprintf(stderr, "  unexpected error text: %s (wanted %s)\n", error.c_str(), expect);
        return false;
    }
    return !ok && !error.empty();
}

void test_hash_and_names() {
    // digest128: deterministic, the low half is digest(), streaming == one-shot, sensitive to every byte and the length
    std::vector<uint8_t> d = fill_vec(100000, 9);
    SessionHasher a(0), b(0);
    a.update(d.data(), d.size());
    for (size_t at = 0; at < d.size();) { const size_t c = std::min<size_t>(d.size() - at, 1 + (at * 7) % 4099); b.update(d.data() + at, c); at += c; }
    check(a.digest128() == b.digest128(), "digest128: streaming equals one-shot");
    const auto h = a.digest128();
    uint64_t lo = 0;
    std::memcpy(&lo, h.data(), 8);
    check(lo == a.digest(), "digest128: the low half is digest()");
    uint64_t hi = 0;
    std::memcpy(&hi, h.data() + 8, 8);
    check(hi != lo && hi != 0, "digest128: the halves differ");
    std::vector<uint8_t> e = d;
    e[54321] ^= 1;
    check(!(chunk_key(e.data(), e.size()) == chunk_key(d.data(), d.size())), "digest128: a flipped bit changes the key");
    check(!(chunk_key(d.data(), 99999) == chunk_key(d.data(), 100000)), "digest128: the length matters");
    std::vector<uint8_t> z(4096, 0), z2(4095, 0);
    check(!(chunk_key(z.data(), z.size()) == chunk_key(z2.data(), z2.size())), "digest128: zeros of two lengths differ");
    check(!(chunk_key(nullptr, 0) == chunk_key(z.data(), 1)), "digest128: empty differs from a zero byte");
    bool halves_independent = true;
    for (int i = 0; i < 64; ++i) {   // both halves change on a one-bit change (no half is a copy of the other's state)
        std::vector<uint8_t> x(1000, 0), y(1000, 0);
        y[i * 13] = 1;
        const auto kx = chunk_key(x.data(), x.size()), ky = chunk_key(y.data(), y.size());
        halves_independent = halves_independent && std::memcmp(kx.b.data(), ky.b.data(), 8) != 0 && std::memcmp(kx.b.data() + 8, ky.b.data() + 8, 8) != 0;
    }
    check(halves_independent, "digest128: both halves move with every input bit");
    // golden key of 1000 zero bytes (the formula is part of format v2)
    std::vector<uint8_t> g(1000, 0);
    const char* gp = std::getenv("STRATA_CHUNKS_GOLDEN_PRINT");
    if (gp) std::printf("golden key zeros1000 %s\n", chunk_key(g.data(), g.size()).hex().c_str());
    if (!gp) check(chunk_key(g.data(), g.size()).hex() == "5a074479f21e2b0e632c570939293878", "digest128: golden key of 1000 zero bytes");

    check(chunk_file_size(0) == 4096 && chunk_file_size(1) == 8192 && chunk_file_size(4095) == 8192 &&
          chunk_file_size(4096) == 8192 && chunk_file_size(4097) == 12288, "chunk_file_size formula");
    ChunkKey k;
    for (int i = 0; i < 16; ++i) k.b[i] = (uint8_t) (0xa0 + i);
    check(k.hex() == "a0a1a2a3a4a5a6a7a8a9aaabacadaeaf", "key hex");
}

void test_chunk_files(bool direct) {
    Fx fx("chunk", direct);
    ChunkPool& pool = *fx.pool;
    const size_t lens[] = {0, 1, 4095, 4096, 4097, 6u << 20};
    for (size_t n : lens) {
        std::vector<uint8_t> raw = fill_vec(n, n + 3);
        ChunkRef r;
        ChunkPutInfo info;
        std::string err;
        check(pool.put(ChunkKind::kv_slab, raw.data(), raw.size(), 0, r, info, err), "put");
        check(!info.hit && info.stored_len == n && r.raw_len == n && r.stored_len == n, "put reports lengths");
        check(info.new_file_bytes == chunk_file_size((uint32_t) n), "put reports the file bytes");
        const std::string path = pool.path_of(r.key);
        check(path.find("/" + r.key.hex().substr(0, 2) + "/" + r.key.hex() + ".chk") != std::string::npos, "chunk path layout");
        const auto f = slurp(path);
        check(f.size() == chunk_file_size((uint32_t) n), "file size is 4096 + round_up(stored_len, 4096)");
        check(std::memcmp(f.data(), "STRCHNK1", 8) == 0, "magic");
        check(rd<uint32_t>(f, 8) == n && rd<uint32_t>(f, 12) == n, "raw_len and stored_len at 8 and 12");
        check(f[16] == 0 && f[17] == 0, "codec raw, no flags");
        check(rd<uint16_t>(f, 18) == 0 && rd<uint32_t>(f, 20) == 0, "kind 0, reserved 0");
        check(std::memcmp(f.data() + 24, r.key.b.data(), 16) == 0, "key at 24");
        check(rd<uint64_t>(f, 40) == session_hash64(f.data(), 40, 0), "header hash at 40");
        bool zero_hdr = true;
        for (size_t i = 48; i < 4096; ++i) zero_hdr = zero_hdr && f[i] == 0;
        check(zero_hdr, "header block is zero past 48");
        check(n == 0 || std::memcmp(f.data() + 4096, raw.data(), n) == 0, "payload starts at 4096");
        bool zero_tail = true;
        for (size_t i = 4096 + n; i < f.size(); ++i) zero_tail = zero_tail && f[i] == 0;
        check(zero_tail, "the tail is zero");
        struct stat st {};
        check(::stat(path.c_str(), &st) == 0 && (st.st_mode & 0777) == 0600, "chunk file mode 0600");
        check(::stat((fx.pool_dir / r.key.hex().substr(0, 2)).c_str(), &st) == 0 && (st.st_mode & 0777) == 0700, "shard folder mode 0700");
        check(count_prefixed(fx.pool_dir, ".tmp-") == 0, "no temporary file left");
        std::vector<uint8_t> back(n + 1, 0xee);
        check(pool.get(ChunkKind::kv_slab, r, back.data(), n, err), "get");
        check(std::equal(raw.begin(), raw.end(), back.begin()) && back[n] == 0xee, "get returns the bytes and nothing more");
        check(pool.exists(r), "exists");
        // a second put is a hit: nothing written, one more ref
        ChunkRef r2;
        check(pool.put(ChunkKind::kv_slab, raw.data(), raw.size(), 0, r2, info, err), "second put");
        check(info.hit && info.new_file_bytes == 0 && r2 == r, "a second put is a hit");
        check(pool.refs_of(r.key) == 2, "two refs");
        ChunkRef r3;
        check(pool.put(ChunkKind::state_slice, raw.data(), raw.size(), 4, r3, info, err) && info.hit && r3 == r, "the same bytes under the other kind are the same chunk");
        // the kind is advisory: the same bytes under the other kind read fine (they would be one file anyway)
        check(pool.get(ChunkKind::state_slice, r, back.data(), n, err), "another kind reads the same bytes");
    }
    // buffered and direct write the same file bytes
    {
        std::vector<uint8_t> raw = fill_vec(5000, 77);
        ChunkRef a, b;
        ChunkPutInfo ia, ib;
        std::string err;
        Fx d("dir", true), bf("buf", false);
        check(d.pool->put(ChunkKind::state_slice, raw.data(), raw.size(), 4, a, ia, err), "direct put");
        check(bf.pool->put(ChunkKind::state_slice, raw.data(), raw.size(), 4, b, ib, err), "buffered put");
        check(!ib.direct_io, "a pool with direct=false writes buffered");
        check(slurp(d.pool->path_of(a.key)) == slurp(bf.pool->path_of(b.key)), "direct and buffered files are byte-identical");
        // EINVAL on the transfer: the buffered fallback writes the identical file (a direct write only: not when
        // STRATA_SESSION_BUFFERED=1 already made every write buffered)
        const char* forced = std::getenv("STRATA_SESSION_BUFFERED");
        if (forced && forced[0] == '1') return;
        Fx e("einval", true);
        bool injected = false;
        e.fault = [&](const char* s) { if (!std::strcmp(s, "chunk_write")) { injected = true; return EINVAL; } return 0; };
        ChunkRef c;
        ChunkPutInfo ic;
        check(e.pool->put(ChunkKind::state_slice, raw.data(), raw.size(), 4, c, ic, err), "put with an injected EINVAL");
        check(injected && !ic.direct_io, "the fallback ran buffered");
        check(slurp(e.pool->path_of(c.key)) == slurp(d.pool->path_of(a.key)), "the fallback file is byte-identical");
        check(std::count(e.steps.begin(), e.steps.end(), std::string("chunk_sync")) == 1, "the buffered path ran the writeback step");
        check(std::count(d.steps.begin(), d.steps.end(), std::string("chunk_sync")) == (ia.direct_io ? 0 : 1), "direct mode skips it (unless the filesystem refused O_DIRECT)");
    }
}

void test_chunk_corruption() {
    Fx fx("corrupt");
    ChunkPool& pool = *fx.pool;
    std::vector<uint8_t> raw = fill_vec(10000, 5);
    ChunkRef r;
    ChunkPutInfo info;
    std::string err;
    check(pool.put(ChunkKind::kv_slab, raw.data(), raw.size(), 0, r, info, err), "put");
    const std::string path = pool.path_of(r.key);
    const std::vector<uint8_t> good = slurp(path);
    std::vector<uint8_t> dst(raw.size());
    auto bad = [&](const char* label, const std::vector<uint8_t>& f, const ChunkRef& ref, ChunkKind kind = ChunkKind::kv_slab) {
        spit(path, f);
        std::string e;
        const bool ok = pool.get(kind, ref, dst.data(), dst.size(), e);
        spit(path, good);
        check(!ok && !e.empty(), label);
    };
    auto ok_with = [&](const char* label, const std::vector<uint8_t>& f) {
        spit(path, f);
        std::string e;
        const bool ok = pool.get(ChunkKind::kv_slab, r, dst.data(), dst.size(), e);
        spit(path, good);
        check(ok && dst == raw, label);
    };
    auto fix_hash = [](std::vector<uint8_t> f) { wr<uint64_t>(f, 40, session_hash64(f.data(), 40, 0)); return f; };
    // each header field, flipped (the hash catches it) and flipped with the hash recomputed (the field's own check)
    const size_t offs[] = {0, 8, 12, 16, 17, 18, 20, 24, 39};
    for (size_t o : offs) {
        auto f = good;
        f[o] ^= 1;
        bad("flipped header byte", f, r);
        if (o != 18) bad("flipped header byte, hash recomputed", fix_hash(f), r);   // the kind is advisory
    }
    { auto f = good; wr<uint32_t>(f, 8, 20000); bad("raw_len raised", fix_hash(f), r); }
    { auto f = good; wr<uint32_t>(f, 12, 9999); bad("stored_len lowered", fix_hash(f), r); }
    { auto f = good; f[16] = 2; bad("codec zstd on a raw payload", fix_hash(f), r); }
    { auto f = good; f[16] = 1; bad("codec lz4 is reserved", fix_hash(f), r); }
    { auto f = good; f[16] = 9; bad("unknown codec", fix_hash(f), r); }
    { auto f = good; f[17] = 1; bad("shuffled raw chunk", fix_hash(f), r); }
    { auto f = good; f[17] = 2; bad("unknown flag", fix_hash(f), r); }
    { auto f = good; f[18] = 1; ok_with("the kind is advisory, not a check", fix_hash(f)); }
    { auto f = good; f[18] = 7; bad("unknown kind", fix_hash(f), r); }
    { auto f = good; f[20] = 1; bad("reserved field set", fix_hash(f), r); }
    { auto f = good; f[24] ^= 0x80; bad("key changed in the file", fix_hash(f), r); }
    // payload and size
    { auto f = good; f[4096 + 5000] ^= 0x10; bad("payload bit flip", f, r); }
    { auto f = good; f[4096] ^= 0x01; bad("first payload byte flip", f, r); }
    { auto f = good; f[4096 + 9999] ^= 0x01; bad("last payload byte flip", f, r); }
    { auto f = good; f.pop_back(); bad("truncated by one byte", f, r); }
    { auto f = good; f.resize(f.size() - 4096); bad("truncated by a block", f, r); }
    { auto f = good; f.resize(4096); bad("header only", f, r); }
    { auto f = good; f.resize(100); bad("shorter than a header", f, r); }
    { auto f = good; f.resize(f.size() + 4096); bad("extended by a block", f, r); }
    { bad("empty file", {}, r); }
    // what the reader ignores
    { auto f = good; f[100] = 0xab; f[4095] = 0xcd; ok_with("garbage past byte 48 of the header block is ignored", f); }
    { auto f = good; f[f.size() - 1] = 0x7f; ok_with("the zero tail is not verified (the key covers the raw bytes)", f); }
    // the reference's expectations
    { ChunkRef x = r; x.raw_len += 1; std::string e; check(!pool.get(ChunkKind::kv_slab, x, dst.data(), dst.size(), e), "ref raw_len off by one"); }
    { ChunkRef x = r; x.stored_len -= 1; std::string e; check(!pool.get(ChunkKind::kv_slab, x, dst.data(), dst.size(), e), "ref stored_len off by one"); }
    { ChunkRef x = r; x.key.b[3] ^= 1; std::string e; check(!pool.get(ChunkKind::kv_slab, x, dst.data(), dst.size(), e), "ref key off (file not found)"); }
    { std::string e; check(!pool.get(ChunkKind::kv_slab, r, dst.data(), dst.size() - 1, e), "destination one byte short"); }
    { ChunkRef x = r; x.raw_len = (uint32_t) kMaxChunkRaw + 1; x.stored_len = x.raw_len; std::string e; check(!pool.get(ChunkKind::kv_slab, x, dst.data(), dst.size(), e), "ref over the chunk limit"); }
    // a chunk moved under another key's name (the wrong shard or file)
    {
        std::vector<uint8_t> raw2 = fill_vec(10000, 6);
        ChunkRef r2;
        check(pool.put(ChunkKind::kv_slab, raw2.data(), raw2.size(), 0, r2, info, err), "second chunk");
        const std::string p2 = pool.path_of(r2.key);
        const auto keep = slurp(p2);
        spit(p2, good);   // chunk 1's bytes under chunk 2's name
        std::string e;
        check(!pool.get(ChunkKind::kv_slab, r2, dst.data(), dst.size(), e), "a chunk under another key's name is refused");
        spit(p2, keep);
        check(pool.get(ChunkKind::kv_slab, r2, dst.data(), dst.size(), e), "restored");
    }
    // links, directories, a huge sparse file
    {
        const std::string alias = path + ".alias";
        fs::rename(path, alias);
        fs::create_symlink(alias, path);
        std::string e;
        check(!pool.get(ChunkKind::kv_slab, r, dst.data(), dst.size(), e), "a symbolic link is refused");
        fs::remove(path);
        fs::create_hard_link(alias, path);
        check(!pool.get(ChunkKind::kv_slab, r, dst.data(), dst.size(), e), "a hard link (two names) is refused");
        fs::remove(path);
        fs::rename(alias, path);
        fs::remove(path);
        fs::create_directory(path);
        check(!pool.get(ChunkKind::kv_slab, r, dst.data(), dst.size(), e), "a directory is refused");
        fs::remove(path);
        spit(path, good);
        // a sparse file of 1 TiB: refused by its size before anything is allocated or read
        std::ofstream(path, std::ios::binary | std::ios::trunc).close();
        check(::truncate(path.c_str(), (off_t) 1 << 40) == 0, "sparse file");
        check(!pool.get(ChunkKind::kv_slab, r, dst.data(), dst.size(), e), "a 1 TiB file is refused by its size");
        spit(path, good);
        check(pool.get(ChunkKind::kv_slab, r, dst.data(), dst.size(), e), "good again");
    }
    // a header that claims a bomb: raw_len and stored_len over the chunk limit, hash valid
    {
        auto f = good;
        wr<uint32_t>(f, 8, (uint32_t) kMaxChunkRaw + 1);
        wr<uint32_t>(f, 12, (uint32_t) kMaxChunkRaw + 1);
        f = fix_hash(f);
        ChunkRef x = r;
        x.raw_len = x.stored_len = (uint32_t) kMaxChunkRaw + 1;
        bad("raw_len over the limit in header and ref", f, x);
    }
    // a zero-length chunk and a missing file
    {
        ChunkRef z;
        check(pool.put(ChunkKind::kv_slab, nullptr, 0, 0, z, info, err), "empty put");
        check(pool.get(ChunkKind::kv_slab, z, dst.data(), 0, err), "empty get");
        check(fs::file_size(pool.path_of(z.key)) == 4096, "an empty chunk is one header block");
        fs::remove(path);
        check(!pool.exists(r) && !pool.get(ChunkKind::kv_slab, r, dst.data(), dst.size(), err), "a missing chunk");
    }
}

void test_pool_lifecycle() {
    Fx fx("life", true, 600);
    ChunkPool& pool = *fx.pool;
    std::string err;
    ChunkPutInfo info;
    std::vector<ChunkRef> refs(3);
    for (int i = 0; i < 3; ++i) {
        std::vector<uint8_t> raw = fill_vec(5000 + (size_t) i * 100, 40 + i);
        check(pool.put(ChunkKind::kv_slab, raw.data(), raw.size(), 0, refs[i], info, err), "put");
    }
    check(pool.count() == 3, "pool count");
    const uint64_t total = pool.bytes();
    check(total == chunk_file_size(5000) + chunk_file_size(5100) + chunk_file_size(5200), "pool bytes are the file sizes");
    // referenced: never swept
    check(pool.sweep(false, true, nullptr) == 0 && pool.count() == 3, "a pressure sweep leaves referenced chunks");
    pool.unref(&refs[0], 1);
    check(pool.refs_of(refs[0].key) == 0, "ghost");
    fx.clock += 599;
    check(pool.sweep(false, false, nullptr) == 0, "a ghost survives inside the grace");
    fx.clock += 2;
    size_t freed = 0;
    check(pool.sweep(false, false, nullptr, &freed) == 1 && freed == chunk_file_size(5000), "a ghost goes after the grace");
    check(!fs::exists(pool.path_of(refs[0].key)) && pool.bytes() == total - chunk_file_size(5000), "file removed, bytes updated");
    // a pressure sweep takes a fresh ghost at once, never a referenced chunk
    pool.unref(&refs[1], 1);
    check(pool.sweep(false, true, nullptr) == 1 && pool.count() == 1, "pressure ignores the grace");
    // a pin is one more ref
    pool.ref(&refs[2], 1);
    pool.unref(&refs[2], 1);
    check(pool.sweep(false, true, nullptr) == 0 && pool.refs_of(refs[2].key) == 1, "a pinned chunk stays");
    // old file, then unref: the grace starts at the release (a chunk written long ago survives take + continue)
    fx.clock += 100000;
    pool.unref(&refs[2], 1);
    check(pool.sweep(false, false, nullptr) == 0, "grace counts from the release, not from the write");
    // shared: only unnamed old chunks, re-checked on the disk
    {
        std::set<ChunkKey> named{refs[2].key};
        fx.clock += 1000;
        // the file's real mtime is the injected clock (stamped), so the re-check agrees
        check(pool.sweep(true, true, &named) == 0 && pool.count() == 1, "shared: a chunk another manifest names stays");
        named.clear();
        check(pool.sweep(true, false, &named) == 1 && pool.count() == 0, "shared: unnamed and old: removed");
    }
    // a shared sweep keeps a chunk younger than the grace even under pressure (a replica's chunk, not yet named)
    {
        std::vector<uint8_t> raw = fill_vec(777, 90);
        ChunkRef r;
        check(pool.put(ChunkKind::kv_slab, raw.data(), raw.size(), 0, r, info, err), "put");
        pool.unref(&r, 1);
        std::set<ChunkKey> none;
        check(pool.sweep(true, true, &none) == 0, "shared: young unnamed chunk survives pressure");
        fx.clock += 601;
        check(pool.sweep(true, false, &none) == 1, "shared: gone after the grace");
    }
    // open() indexes what is there, removes old temp files, and remove_all clears the folder
    {
        std::vector<uint8_t> raw = fill_vec(1234, 91);
        ChunkRef r;
        check(pool.put(ChunkKind::kv_slab, raw.data(), raw.size(), 0, r, info, err), "put");
        const fs::path shard = fx.pool_dir / r.key.hex().substr(0, 2);
        const fs::path old_tmp = shard / ".tmp-0000000000000001", young_tmp = shard / ".tmp-0000000000000002";
        spit(old_tmp, {1}); spit(young_tmp, {2});
        struct timespec ts[2] = {{0, UTIME_OMIT}, {(time_t) (fx.clock - 700), 0}};
        ::utimensat(AT_FDCWD, old_tmp.c_str(), ts, 0);
        ts[1].tv_sec = (time_t) fx.clock;
        ::utimensat(AT_FDCWD, young_tmp.c_str(), ts, 0);
        fs::create_directories(fx.pool_dir / "zz");           // not a shard
        spit(shard / "README", {1});                           // not a chunk
        fx.make_pool(true);
        check(!fs::exists(old_tmp) && fs::exists(young_tmp), "open removes only the temporary files older than 10 minutes");
        check(fx.pool->count() == 1 && fx.pool->refs_of(r.key) == 0 && fx.pool->bytes() == chunk_file_size(1234), "open indexes the chunks as ghosts");
        // a put of the same bytes in the new pool is a hit
        ChunkRef again;
        check(fx.pool->put(ChunkKind::kv_slab, raw.data(), raw.size(), 0, again, info, err) && info.hit && again == r, "hit on a chunk found at open");
        fx.pool->remove_all();
        check(!fs::exists(fx.pool_dir) && fx.pool->count() == 0 && fx.pool->bytes() == 0, "remove_all");
    }
    // a pool dir that cannot be a folder
    {
        spit(fx.root / "afile", {1});
        ChunkPoolOptions o;
        o.dir = (fx.root / "afile" / "pool").string();
        ChunkPool bad(o);
        std::string e;
        check(!bad.open(e) && !e.empty(), "open fails under a file");
    }
}

void test_failures() {
    Fx fx("fail");
    ChunkRef r;
    ChunkPutInfo info;
    std::string err;
    std::vector<uint8_t> raw = fill_vec(9000, 3);
    // a failed write leaves no temporary file and no index entry
    fx.fault = [](const char* s) { return !std::strcmp(s, "chunk_write") ? ENOSPC : 0; };
    check(!fx.pool->put(ChunkKind::kv_slab, raw.data(), raw.size(), 0, r, info, err) && info.error_kind == SessionError::storage, "ENOSPC is a storage failure");
    check(count_prefixed(fx.pool_dir, ".tmp-") == 0 && fx.pool->count() == 0 && fx.pool->bytes() == 0, "nothing left by a failed write");
    fx.fault = [](const char* s) { return !std::strcmp(s, "chunk_rename") ? EIO : 0; };
    check(!fx.pool->put(ChunkKind::kv_slab, raw.data(), raw.size(), 0, r, info, err) && info.error_kind == SessionError::io, "a rename failure");
    check(count_prefixed(fx.pool_dir, ".tmp-") == 0 && count_files(fx.pool_dir, ".chk") == 0, "nothing left by a failed rename");
    fx.fault = [](const char* s) { return !std::strcmp(s, "chunk_sync") ? EIO : 0; };
    check(!fx.pool->put(ChunkKind::kv_slab, raw.data(), raw.size(), 0, r, info, err) || info.direct_io, "a failed writeback (buffered mode only)");
    fx.fault = nullptr;
    // the free-space preflight
    fx.make_pool(true, 600, UINT64_MAX / 2);
    const size_t before_files = count_files(fx.pool_dir, ".chk");
    std::vector<uint8_t> fresh = fill_vec(9001, 4);
    check(!fx.pool->put(ChunkKind::kv_slab, fresh.data(), fresh.size(), 0, r, info, err) && info.error_kind == SessionError::storage, "the free-space preflight refuses");
    check(count_files(fx.pool_dir, ".chk") == before_files, "nothing written when refused");
    fx.make_pool(true);
    // an oversized payload
    check(!fx.pool->put(ChunkKind::kv_slab, raw.data(), (size_t) kMaxChunkRaw + 1, 0, r, info, err), "a payload over the limit is refused");
    // a stale file under the name (right size, wrong content) is replaced by a put
    check(fx.pool->put(ChunkKind::kv_slab, raw.data(), raw.size(), 0, r, info, err), "put");
    {
        auto f = slurp(fx.pool->path_of(r.key));
        f[0] ^= 1;   // a torn header
        spit(fx.pool->path_of(r.key), f);
        ChunkRef r2;
        check(fx.pool->put(ChunkKind::kv_slab, raw.data(), raw.size(), 0, r2, info, err) && !info.hit && r2 == r, "a torn existing chunk is replaced, not hit");
        std::vector<uint8_t> back(raw.size());
        check(fx.pool->get(ChunkKind::kv_slab, r, back.data(), back.size(), err) && back == raw, "and reads back");
    }
}

void test_concurrent_put() {
    Fx fx("conc");
    std::vector<uint8_t> raw = fill_vec(300000, 12);
    std::atomic<int> misses{0}, hits{0}, failures{0};
    std::vector<std::thread> ts;
    ChunkRef refs[8];
    for (int t = 0; t < 8; ++t)
        ts.emplace_back([&, t] {
            ChunkPutInfo info;
            std::string err;
            if (!fx.pool->put(ChunkKind::kv_slab, raw.data(), raw.size(), 0, refs[t], info, err)) { ++failures; return; }
            (info.hit ? hits : misses)++;
        });
    for (auto& t : ts) t.join();
    check(failures == 0, "concurrent puts of one chunk all succeed");
    check(misses == 1 && hits == 7, "exactly one writer won the no-replace rename");
    check(fx.pool->refs_of(refs[0].key) == 8 && fx.pool->count() == 1 && count_files(fx.pool_dir, ".chk") == 1, "8 refs, one file");
    check(count_prefixed(fx.pool_dir, ".tmp-") == 0, "no temporary left");
    check(fx.pool->bytes() == chunk_file_size(300000), "bytes counted once");
    // two independent pools on one folder (two engines) racing on one chunk
    ChunkPoolOptions o = fx.pool->options();
    ChunkPool other(o);
    std::string e;
    check(other.open(e), "second pool opens");
    std::vector<uint8_t> raw2 = fill_vec(200000, 13);
    ChunkRef a, b;
    ChunkPutInfo ia, ib;
    std::thread t1([&] { std::string er; fx.pool->put(ChunkKind::kv_slab, raw2.data(), raw2.size(), 0, a, ia, er); });
    std::thread t2([&] { std::string er; other.put(ChunkKind::kv_slab, raw2.data(), raw2.size(), 0, b, ib, er); });
    t1.join(); t2.join();
    check(a == b && (ia.hit != ib.hit || (ia.hit && ib.hit)), "two pools: one chunk");
    check(count_files(fx.pool_dir, ".chk") == 2, "two pools: the same file");
}

void test_round_trip_geometry() {
    const int64_t ns[] = {0, 1, 3, 4, 511, 512, 513, 1024, 65537};
    for (uint32_t T : {64u, 512u}) {
        for (int64_t n : ns) {
            Fx fx("geo");
            Spec sp;
            sp.n = n;
            sp.live_gdn = n % 2 ? (4u << 20) + 4097 : 5000;
            sp.cps = {n / 4, n / 2};
            if (n == 0) sp.cps = {};
            sp.cvec = n % 3 == 0;
            sp.rope = n % 5 == 0 ? 1.5 : 1.0;
            const SavedConversation img = make_image(sp);
            W w;
            const std::string label = "T=" + std::to_string(T) + " n=" + std::to_string(n);
            const bool ok = write_to(fx, fx.man.string(), img, w, T);
            if (!ok) std::fprintf(stderr, "  %s: %s\n", label.c_str(), w.error.c_str());
            check(ok, "geometry: write");
            const uint64_t cells = (uint64_t) ((n + 3) / 4 * 4), n_slabs = (cells + T - 1) / T;
            check(w.refs.size() == n_slabs + expect_slices(sp), "geometry: chunk count = slabs + state slices");
            for (uint64_t i = 0; i < n_slabs; ++i)
                if (w.refs[i].raw_len != expect_slab(sp, i, T)) check(false, "geometry: a slab's raw length is not what the geometry says");
            std::vector<ChunkRef> listed;
            uint64_t mbytes = 0;
            std::string err;
            check(chunked_manifest_refs(fx.man.string(), kId, listed, mbytes, err), "geometry: manifest refs");
            check(listed == w.refs && mbytes == fs::file_size(fx.man) && mbytes == w.stats.manifest_bytes, "geometry: refs and size match the write");
            for (const auto& r : w.refs) check(fx.pool->refs_of(r.key) >= 1, "geometry: every chunk is referenced");
            check(w.stats.chunks == w.refs.size() && w.stats.hits + w.stats.misses == w.stats.chunks, "geometry: stats add up");
            SavedConversation back;
            SessionStatus st;
            const bool rok = read_from(fx, fx.man.string(), back, err, {}, &st);
            if (!rok) std::fprintf(stderr, "  %s: %s\n", label.c_str(), err.c_str());
            check(rok && st.error == SessionError::none, "geometry: read");
            check(same(img, back), "geometry: the round trip is byte-identical");
            check(count_prefixed(fx.root, ".manifest") == 0 && count_prefixed(fx.root, ".manifest.bin") == 0, "geometry: no manifest temp");
        }
    }
    // an image with no K/V layers, and with a layer whose parts are empty
    {
        Fx fx("nolayers");
        Spec sp;
        sp.n = 50;
        SavedConversation img = make_image(sp);
        img.kv.clear();
        W w;
        std::string err;
        check(write_to(fx, fx.man.string(), img, w, 64), "no layers: write");
        SavedConversation back;
        check(read_from(fx, fx.man.string(), back, err) && same(img, back), "no layers: round trip");
        ConversationKv empty;
        empty.format = 3; empty.cells = 8; empty.pooled_rows = 0;
        img.kv.push_back(empty);   // cells > 0 but every part empty: slabs of zero bytes
        check(write_to(fx, fx.man.string(), img, w, 64), "empty parts: write");
        check(read_from(fx, fx.man.string(), back, err) && same(img, back), "empty parts: round trip");
    }
    // unsupported geometries are refused on write
    {
        Fx fx("badgeo");
        Spec sp;
        sp.n = 100;
        SavedConversation img = make_image(sp);
        W w;
        check(!write_to(fx, fx.man.string(), img, w, 102) && w.st.error == SessionError::invalid, "chunk_tokens not a multiple of 4");
        check(!write_to(fx, fx.man.string(), img, w, 32) && !write_to(fx, fx.man.string(), img, w, 8192), "chunk_tokens out of range");
        auto odd = img;
        odd.kv[0].k.resize(odd.kv[0].k.size() + 1);
        check(!write_to(fx, fx.man.string(), odd, w, 64) && w.st.error == SessionError::invalid, "a part that is not whole cells");
        odd = img;
        odd.kv[0].pooled.resize(odd.kv[0].pooled.size() + 1);
        check(!write_to(fx, fx.man.string(), odd, w, 64), "a pooled part that is not whole rows");
        odd = img;
        odd.kv[0].pooled_rows = 5;   // spare row of a much shorter image: rows not covered by the cells' slabs is fine, bytes are not
        check(!write_to(fx, fx.man.string(), odd, w, 64), "pooled rows that do not match the pooled bytes");
        odd = img;
        odd.live.stage_parts.resize(1);
        check(!write_to(fx, fx.man.string(), odd, w, 64) && w.error.find("layer-split") != std::string::npos, "layer-split state is refused");
        odd = img;
        auto src = sources_of(img);
        check(!write_to(fx, fx.man.string(), odd, w, 64, {}, &src), "K/V in the image and as sources is refused");
        check(count_files(fx.root, ".bin") == 0, "nothing published by refusals");
        // a slab over the chunk limit
        odd = make_image(Spec{.n = 8});
        odd.kv[0].cells = 4;
        odd.kv[0].k = fill_buf(4u << 20, 1);   // 1 MiB per cell: a 64-cell slab is 64 MiB+
        odd.kv[0].v = {};
        odd.kv[0].k_scale = {}; odd.kv[0].v_scale = {}; odd.kv[0].pooled = {}; odd.kv[0].pooled_rows = 0;
        odd.kv.resize(1);
        odd.kv[0].cells = 4;
        odd.kv[0].k = fill_buf(4u * (20u << 20), 1);   // 20 MiB per cell, 4 cells in one slab: 80 MiB
        check(!write_to(fx, fx.man.string(), odd, w, 64) && w.error.find("chunk limit") != std::string::npos, "a slab over the chunk limit is refused");
    }
}

void test_padding_and_spare() {
    // garbage in the padding cells and the spare row changes only the slabs they live in
    Fx fx("pad");
    Spec a, b;
    a.n = b.n = 1001;      // cells 1004, slabs [0,512) sealed and [512,1004) open at T=512
    b.pad_seed = 99;
    W wa, wb;
    check(write_to(fx, fx.path("a.bin"), make_image(a), wa, 512) && write_to(fx, fx.path("b.bin"), make_image(b), wb, 512), "write both");
    check(wa.refs.size() == wb.refs.size(), "same shape");
    check(wa.refs[0] == wb.refs[0], "the sealed slab's key ignores the garbage past n");
    check(!(wa.refs[1] == wb.refs[1]), "the open slab carries the padding");
    SavedConversation back;
    std::string err;
    check(read_from(fx, fx.path("b.bin"), back, err) && same(make_image(b), back), "the garbage round-trips (the file keeps what it was given)");
    // n a multiple of T: the spare row lives in the manifest, no slab holds it
    Spec c, d;
    c.n = d.n = 1024;      // cells 1024: 2 slabs; pooled rows 257: 256 in slabs + the spare
    d.pad_seed = 5;        // changes the spare row only (cells == n: no padding cells)
    W wc, wd;
    check(write_to(fx, fx.path("c.bin"), make_image(c), wc, 512) && write_to(fx, fx.path("d.bin"), make_image(d), wd, 512), "write both, n = 2T");
    check(wc.refs.size() == wd.refs.size() && wc.refs == wd.refs, "the spare row changes no chunk key");
    check(wc.refs[0].raw_len == expect_slab(c, 0, 512) && wc.refs[1].raw_len == expect_slab(c, 1, 512), "slab lengths exclude the spare row");
    check(slurp(fx.path("c.bin")) != slurp(fx.path("d.bin")), "but the manifests differ");
    check(read_from(fx, fx.path("d.bin"), back, err) && same(make_image(d), back), "the spare row round-trips through the manifest");
}

void test_source_equals_image() {
    Fx fx("src");
    for (int64_t n : {0, 5, 700, 2049}) {
        Spec sp;
        sp.n = n;
        sp.cps = {n / 3};
        const SavedConversation img = make_image(sp);
        SavedConversation meta = img;
        meta.kv.clear();
        auto src = sources_of(img);
        W wi, ws;
        check(write_to(fx, fx.path("i.bin"), img, wi, 64), "image write");
        check(write_to(fx, fx.path("s.bin"), meta, ws, 64, {}, &src), "source write");
        check(wi.refs == ws.refs && slurp(fx.path("i.bin")) == slurp(fx.path("s.bin")), "source write == image write, byte for byte");
        // a failing source
        auto bad = src;
        std::atomic<int> calls{0};
        for (auto& s : bad) {
            auto inner = s.read;
            s.read = [inner, &calls](size_t p, size_t o, void* d, size_t c) { return ++calls == 3 ? false : inner(p, o, d, c); };
            s.error = [] { return std::string("device lost"); };
        }
        W wf;
        if (n >= 64) {
            check(!write_to(fx, fx.path("f.bin"), meta, wf, 64, {}, &bad) && wf.error.find("device lost") != std::string::npos, "a failing source fails the write");
            check(!fs::exists(fx.path("f.bin")) && wf.refs.empty(), "and publishes nothing");
        }
    }
}

void test_dedup() {
    Fx fx("dedup");
    Spec a;
    a.n = 3000;
    a.cps = {800, 1700};
    W wa;
    check(write_to(fx, fx.path("a.bin"), make_image(a), wa, 512), "write a");
    check(wa.stats.hits == 0 && wa.stats.misses == wa.stats.chunks, "a first write is all new chunks");
    check(wa.stats.new_file_bytes > 0 && wa.stats.raw_bytes > 0, "stats count bytes");
    // the same image again: every chunk is a hit, no file bytes
    W wa2;
    check(write_to(fx, fx.path("a2.bin"), make_image(a), wa2, 512), "write a again");
    check(wa2.stats.hits == wa2.stats.chunks && wa2.stats.new_file_bytes == 0 && wa2.stats.hit_raw_bytes == wa2.stats.raw_bytes, "an identical image writes nothing");
    check(slurp(fx.path("a.bin")) == slurp(fx.path("a2.bin")), "and gives the identical manifest");
    for (const auto& r : wa.refs) check(fx.pool->refs_of(r.key) == 2, "every chunk has two refs");
    // a continuation by 300 tokens: the sealed slabs and the old checkpoints' states are shared
    Spec b = a;
    b.n = 3300;
    W wb;
    check(write_to(fx, fx.path("b.bin"), make_image(b), wb, 512), "write the continuation");
    const uint64_t sealed = 3000 / 512;   // slabs fully inside the old image
    const size_t cp_slices = 2;            // the two old checkpoints: same bytes, one slice each
    check(wb.stats.hits == sealed + cp_slices, "the continuation hits the sealed slabs and the old checkpoint states");
    check(wb.stats.misses == wb.refs.size() - wb.stats.hits, "and writes the rest");
    check(wb.stats.new_file_bytes < wa.stats.new_file_bytes / 3 + 3 * (4u << 20), "and few bytes");
    // two images sharing a prefix share the slabs of the prefix
    Spec c = a;
    c.n = 2000;
    c.cps = {800};
    W wc;
    check(write_to(fx, fx.path("c.bin"), make_image(c), wc, 512), "write the shorter image");
    check(wc.stats.hits >= 2000 / 512 + 1, "a shorter image of the same conversation shares slabs and a checkpoint");
    // the chunks outlive their manifest by reference count, not by the file
    SavedConversation back;
    std::string err;
    check(read_from(fx, fx.path("b.bin"), back, err) && same(make_image(b), back), "continuation reads back");
    fx.pool->unref(wb.refs.data(), wb.refs.size());
    check(read_from(fx, fx.path("b.bin"), back, err) && same(make_image(b), back), "ghosts (refs 0) still read");
    // another chunk size shares nothing for the K/V but still the states
    W wd;
    check(write_to(fx, fx.path("d.bin"), make_image(a), wd, 256), "write with T=256");
    check(wd.stats.hits == expect_slices(a), "a different chunk size shares only the state slices");
}

void test_manifest_format() {
    Fx fx("man");
    Spec sp;
    sp.n = 100;
    sp.cps = {25, 50};
    const SavedConversation img = make_image(sp);
    W w;
    check(write_to(fx, fx.man.string(), img, w, 64), "write");
    const auto f = slurp(fx.man);
    check(std::memcmp(f.data(), "STRSESS\x02", 8) == 0, "header: magic STRSESS\\x02");
    check(rd<uint32_t>(f, 8) == 2 && rd<uint32_t>(f, 12) == 64, "header: version 2, header size 64");
    check(rd<uint64_t>(f, 16) == kId.model && rd<uint64_t>(f, 24) == kId.config, "header: fingerprints at 16 and 24");
    check(rd<uint64_t>(f, 32) == f.size() - 64 - 16, "header: body length at 32");
    check(rd<uint32_t>(f, 40) == 64 && rd<uint32_t>(f, 44) == 0, "header: chunk_tokens at 40, flags at 44");
    check(rd<uint64_t>(f, 56) == session_hash64(f.data(), 56, 0), "header: hash at 56");
    check(std::memcmp(f.data() + f.size() - 8, "STRSEND\x02", 8) == 0, "trailer: end marker");
    check(rd<uint64_t>(f, f.size() - 16) == session_hash64(f.data() + 64, f.size() - 80, 0), "trailer: body hash");
    check(is_chunked_manifest(fx.man.string()) && !is_chunked_manifest(fx.path("nothing")), "is_chunked_manifest");
    // a raw total the reader can verify
    uint64_t raw_total = 0;
    for (const auto& k : img.kv) raw_total += k.k.size() + k.v.size() + k.k_scale.size() + k.v_scale.size() + k.pooled.size();
    auto cp = [](const ConversationCheckpoint& c) { return c.gdn.size() + c.ids.size() * 4 + c.imgs.size() * 16 + c.ple.size() + c.tails.size() + c.dead.size() + c.block_pos.size(); };
    raw_total += cp(img.live);
    for (const auto& c : img.checkpoints) raw_total += cp(c);
    check(rd<uint64_t>(f, 48) == raw_total, "header: raw_total_bytes at 48 is the parsed image's size");
    // deterministic
    W w2;
    check(write_to(fx, fx.path("again.bin"), img, w2, 64) && slurp(fx.path("again.bin")) == f, "the manifest is deterministic");
    // golden: the fixed sample's manifest, with the raw codec
    const uint64_t h = session_hash64(f.data(), f.size(), 0);
    if (std::getenv("STRATA_CHUNKS_GOLDEN_PRINT"))
        std::printf("golden manifest size %zu hash %016llx first %s (direct_io %d)\n", f.size(), (unsigned long long) h, w.refs[0].key.hex().c_str(), (int) w.stats.direct_io);
    if (!std::getenv("STRATA_CHUNKS_GOLDEN_PRINT")) check(f.size() == 3089 && h == 0xc0d401c284129aedull && w.refs[0].key.hex() == "d97408fbf29a3747294515400c239255", "golden: the fixed sample's manifest bytes are unchanged");
    // v1 and v2 side by side
    {
        const std::string v1 = fx.path("v1.bin");
        size_t bytes = 0;
        std::string err;
        check(session_file_write(v1, img, kId, bytes, err), "v1 write");
        check(!is_chunked_manifest(v1), "a v1 file is not a chunked manifest");
        SavedConversation back;
        check(!session_file_read(fx.man.string(), kId, back, bytes, err) && err.find("magic") != std::string::npos, "the v1 reader refuses a v2 manifest with a clean magic error");
        check(refused(fx, v1, "magic"), "the v2 reader refuses a v1 file with a clean magic error");
        check(session_file_read(v1, kId, back, bytes, err) && same(img, back), "and v1 still reads");
    }
}

void test_manifest_corruption() {
    Fx fx("mcorrupt");
    Spec sp;
    sp.n = 700;
    sp.cps = {100, 300};
    sp.live_gdn = 5000;
    const SavedConversation img = make_image(sp);
    W w;
    check(write_to(fx, fx.man.string(), img, w, 64), "write");
    const auto good = slurp(fx.man);
    const std::string p = fx.path("bad.bin");
    auto bad = [&](const char* label, const std::vector<uint8_t>& f, const char* expect = nullptr) {
        spit(p, f);
        check(refused(fx, p, expect), label);
        std::vector<ChunkRef> refs;
        uint64_t mb = 0;
        std::string e;
        check(!chunked_manifest_refs(p, kId, refs, mb, e), label);
    };
    // every byte flip in the header and trailer, and a stride through the body
    for (size_t i = 0; i < 64; ++i) { auto f = good; f[i] ^= 1; bad("header byte flip", f); }
    for (size_t i = good.size() - 16; i < good.size(); ++i) { auto f = good; f[i] ^= 1; bad("trailer byte flip", f); }
    for (size_t i = 64; i < good.size() - 16; i += 13) { auto f = good; f[i] ^= 0x40; bad("body byte flip", f); }
    { auto f = good; f.push_back(0); bad("extra byte after the trailer", f, "does not match"); }
    { auto f = good; f.pop_back(); bad("truncated trailer", f); }
    { bad("empty", {}); }
    { bad("short", std::vector<uint8_t>(good.begin(), good.begin() + 40)); }
    // identity
    {
        spit(p, good);
        SavedConversation out;
        size_t mb = 0;
        uint64_t rb = 0;
        std::string e;
        SessionFileIdentity other{kId.model + 1, kId.config};
        check(!chunked_read(*fx.pool, p, other, out, mb, rb, e, {}) && e.find("model") != std::string::npos, "another model");
        other = {kId.model, kId.config + 1};
        check(!chunked_read(*fx.pool, p, other, out, mb, rb, e, {}) && e.find("configuration") != std::string::npos, "another configuration");
    }
    // header edits with the header hash recomputed: their own checks
    { auto f = good; wr<uint32_t>(f, 8, 3); reseal_header(f); bad("unsupported version", f, "version"); }
    { auto f = good; wr<uint32_t>(f, 12, 80); reseal_header(f); bad("another header size", f); }
    { auto f = good; wr<uint32_t>(f, 44, 1); reseal_header(f); bad("flags set", f); }
    for (uint32_t t : {0u, 4u, 63u, 66u, 4100u, 0xFFFFFFFFu}) { auto f = good; wr<uint32_t>(f, 40, t); reseal_header(f); bad("invalid chunk_tokens", f); }
    { auto f = good; wr<uint32_t>(f, 40, 256); reseal_header(f); bad("another chunk_tokens than the body was cut with", f); }
    { auto f = good; wr<uint64_t>(f, 48, rd<uint64_t>(f, 48) + 1); reseal_header(f); bad("raw_total off by one", f, "does not match the body"); }
    { auto f = good; wr<uint64_t>(f, 32, rd<uint64_t>(f, 32) + 1); reseal_header(f); bad("body_len off by one", f); }
    // body edits, resealed: the invariants
    const auto body = body_of(good);
    const size_t layers_at = 172;
    const size_t main_layer = 56 + 40 + 5 + kRowBytes, draft_layer = 56 + 40 + 5;
    const size_t slabs_at = layers_at + 2 * main_layer + draft_layer;
    const uint32_t n_slabs = rd<uint32_t>(body, slabs_at);
    check(n_slabs == (700 + 63) / 64, "test layout: the slab count is where the test expects it");
    const size_t cp_at = slabs_at + 4 + (size_t) n_slabs * 24;
    auto edit = [&](const char* label, auto&& fn, const char* expect = nullptr) {
        auto b = body;
        fn(b);
        bad(label, reseal_body(good, b), expect);
    };
    edit("n_kv_layers huge", [&](auto& b) { wr<uint32_t>(b, 168, 0xFFFFFFFFu); });
    edit("n_kv_layers one more", [&](auto& b) { wr<uint32_t>(b, 168, 4); });
    edit("n_kv_layers one fewer", [&](auto& b) { wr<uint32_t>(b, 168, 2); });
    edit("cells huge", [&](auto& b) { wr<int64_t>(b, layers_at + 8, int64_t(1) << 60); });
    edit("cells negative", [&](auto& b) { wr<int64_t>(b, layers_at + 8, -4); });
    edit("cells that do not divide the K part", [&](auto& b) { wr<int64_t>(b, layers_at + 8, 701); });
    edit("pooled_rows negative", [&](auto& b) { wr<int64_t>(b, layers_at + 40, -1); });
    edit("K size not whole cells", [&](auto& b) { wr<uint64_t>(b, layers_at + 56, rd<uint64_t>(b, layers_at + 56) + 1); });
    edit("K size huge", [&](auto& b) { wr<uint64_t>(b, layers_at + 56, uint64_t(1) << 62); });
    edit("pooled size not whole rows", [&](auto& b) { wr<uint64_t>(b, layers_at + 56 + 32, rd<uint64_t>(b, layers_at + 56 + 32) + 1); });
    edit("shuffle width 3", [&](auto& b) { b[layers_at + 56 + 40] = 3; });
    edit("n_slabs zero", [&](auto& b) { wr<uint32_t>(b, slabs_at, 0); });
    edit("n_slabs one more", [&](auto& b) { wr<uint32_t>(b, slabs_at, n_slabs + 1); });
    edit("n_slabs one fewer", [&](auto& b) { wr<uint32_t>(b, slabs_at, n_slabs - 1); });
    edit("n_slabs huge", [&](auto& b) { wr<uint32_t>(b, slabs_at, 0xFFFFFFFFu); });
    edit("slab raw_len +1", [&](auto& b) { wr<uint32_t>(b, slabs_at + 4 + 16, rd<uint32_t>(b, slabs_at + 4 + 16) + 1); });
    edit("slab raw_len bomb", [&](auto& b) { wr<uint32_t>(b, slabs_at + 4 + 16, 0x80000000u); });
    edit("slab stored_len over raw_len", [&](auto& b) { wr<uint32_t>(b, slabs_at + 4 + 20, rd<uint32_t>(b, slabs_at + 4 + 20) + 1); });
    edit("ids count huge", [&](auto& b) { wr<uint64_t>(b, cp_at, ~uint64_t(0)); });
    edit("ids count over the body", [&](auto& b) { wr<uint64_t>(b, cp_at, b.size()); });
    edit("ids count overflows the multiplication", [&](auto& b) { wr<uint64_t>(b, cp_at, uint64_t(1) << 62); });
    const size_t ids_end = cp_at + 8 + 700 * 4;
    edit("imgs count huge", [&](auto& b) { wr<uint64_t>(b, ids_end, uint64_t(1) << 60); });
    const size_t gdn_len_at = ids_end + 8 + 16 + 4 * 8 + (77 + 301 + 64 + 8) + 8;
    check(rd<uint64_t>(body, gdn_len_at) == 5000, "test layout: gdn_len is where the test expects it");
    edit("gdn_len huge", [&](auto& b) { wr<uint64_t>(b, gdn_len_at, uint64_t(1) << 60); });
    edit("gdn_len +1 (slice length mismatch)", [&](auto& b) { wr<uint64_t>(b, gdn_len_at, 5001); wr<uint64_t>(b, gdn_len_at, 5001); });
    edit("slice_len zero", [&](auto& b) { wr<uint32_t>(b, gdn_len_at + 8, 0); });
    edit("slice_len not a multiple of 4", [&](auto& b) { wr<uint32_t>(b, gdn_len_at + 8, 4098); });
    edit("slice_len over the chunk limit", [&](auto& b) { wr<uint32_t>(b, gdn_len_at + 8, 0x10000000u); });
    edit("n_slices wrong", [&](auto& b) { wr<uint32_t>(b, gdn_len_at + 12, 2); });
    edit("n_slices huge", [&](auto& b) { wr<uint32_t>(b, gdn_len_at + 12, 0xFFFFFFFFu); });
    edit("state slice raw_len wrong", [&](auto& b) { wr<uint32_t>(b, gdn_len_at + 16 + 16, 4999); });
    edit("trailing bytes in the body", [&](auto& b) { b.push_back(0); }, "unparsed");
    edit("n_checkpoints huge", [&](auto& b) {
        // the checkpoint count sits after the live checkpoint: find it by its value (2)
        for (size_t i = gdn_len_at + 16 + 24; i + 8 <= b.size(); ++i) if (rd<uint64_t>(b, i) == 2) { wr<uint64_t>(b, i, uint64_t(1) << 40); return; }
    });
    // every truncation of the body, resealed: bounded reads, never a crash
    size_t cuts = 0;
    for (size_t cut = 0; cut < body.size(); cut += cut < 700 ? 1 : 29) {
        std::vector<uint8_t> b(body.begin(), body.begin() + (std::ptrdiff_t) cut);
        spit(p, reseal_body(good, b));
        std::string e;
        SavedConversation out;
        size_t mb;
        uint64_t rb;
        check(!chunked_read(*fx.pool, p, kId, out, mb, rb, e, {}) && !e.empty(), "a truncated body is refused");
        ++cuts;
    }
    check(cuts > 700, "truncation sweep ran");
    // a manifest that names a missing chunk or a torn one
    {
        spit(p, good);
        SavedConversation out;
        std::string e;
        const std::string cpath = fx.pool->path_of(w.refs[2].key);
        const auto keep = slurp(cpath);
        fs::remove(cpath);
        check(!read_from(fx, p, out, e) && e.find("missing") != std::string::npos, "a missing chunk is a miss");
        auto torn = keep;
        torn[5000] ^= 1;
        spit(cpath, torn);
        check(!read_from(fx, p, out, e), "a torn chunk is a miss");
        torn = keep;
        torn.resize(torn.size() - 4096);
        spit(cpath, torn);
        check(!read_from(fx, p, out, e), "a short chunk is a miss");
        spit(cpath, keep);
        check(read_from(fx, p, out, e) && same(img, out), "restored");
    }
    // links to the manifest
    {
        fs::remove(p);
        fs::create_symlink(fx.man, p);
        check(refused(fx, p), "a symbolic link as the manifest is refused");
        fs::remove(p);
        fs::create_hard_link(fx.man, p);
        check(refused(fx, p), "a hard link as the manifest is refused");
        fs::remove(p);
        check(refused(fx, fx.path("none.bin")), "a missing manifest");
    }
}

void test_limits() {
    Fx fx("limits");
    Spec sp;
    sp.n = 400;
    sp.cps = {100};
    sp.live_gdn = 5000;
    const SavedConversation img = make_image(sp);
    W w;
    check(write_to(fx, fx.man.string(), img, w, 64), "write");
    const std::string p = fx.man.string();
    SavedConversation sentinel = make_image(Spec{.n = 8});
    auto run = [&](const SessionReadLimits& lim, SessionStatus* st = nullptr) {
        SavedConversation out = sentinel;
        std::string e;
        const bool ok = read_from(fx, p, out, e, lim, st);
        if (!ok) check(same(out, sentinel), "a refused read leaves the output unchanged");
        return ok;
    };
    check(run({}), "defaults accept");
    {
        SessionReadLimits l;
        l.max_tokens = 399;
        check(!run(l), "max_tokens");
        l.max_tokens = 400;
        check(run(l), "max_tokens at the edge");
    }
    { SessionReadLimits l; l.max_checkpoints = 0; check(!run(l), "max_checkpoints"); l.max_checkpoints = 1; check(run(l), "max_checkpoints at the edge"); }
    { SessionReadLimits l; l.max_kv_layers = 2; check(!run(l), "max_kv_layers"); l.max_kv_layers = 3; check(run(l), "max_kv_layers at the edge"); }
    { SessionReadLimits l; l.max_state_bytes[0] = 4999; check(!run(l), "max gdn bytes"); l.max_state_bytes[0] = 5000; check(run(l), "max gdn bytes at the edge"); }
    { SessionReadLimits l; l.max_state_bytes[1] = 10; check(!run(l), "max ple bytes"); }
    {
        SessionReadLimits l;
        l.max_kv_bytes.assign(3, {UINT64_MAX, UINT64_MAX, UINT64_MAX, UINT64_MAX, UINT64_MAX});
        l.max_kv_bytes[1][0] = img.kv[1].k.size() - 1;
        check(!run(l), "max_kv_bytes of one part");
        l.max_kv_bytes[1][0] = img.kv[1].k.size();
        check(run(l), "max_kv_bytes at the edge");
    }
    { SessionReadLimits l; l.geometry = img.geometry; check(run(l), "the runtime's geometry"); l.geometry->at(3) += 1; check(!run(l), "another geometry"); }
    { SessionReadLimits l; l.layer_range = {0, 48}; check(run(l), "the layer range"); l.layer_range = {0, 24}; check(!run(l), "another layer range"); }
    {
        SessionReadLimits l;
        l.max_file_bytes = 1000;   // no other limit set: it bounds the file, and the image size is checked against it
        check(!run(l), "the image size is bounded by max_file_bytes");
    }
    // admit: asked once with the peak, before anything is allocated; a refusal is a memory failure
    {
        int calls = 0;
        uint64_t need = 0;
        SessionReadLimits l;
        l.admit = [&](uint64_t n, std::string&) { ++calls; need = n; return true; };
        check(run(l) && calls == 1, "admit is asked exactly once");
        uint64_t raw_total = rd<uint64_t>(slurp(p), 48);
        std::vector<ChunkRef> mrefs;
        uint64_t mbytes = 0, max_raw = 0;
        std::string merr;
        check(chunked_manifest_refs(p, kId, mrefs, mbytes, merr), "the manifest's references");
        for (const auto& r : mrefs) max_raw = std::max<uint64_t>(max_raw, r.raw_len);
        check(need == chunked_read_peak_bytes(raw_total, 1, 64, max_raw), "with chunked_read_peak_bytes(raw_total, workers, chunk_tokens, the largest chunk)");
        check(need > raw_total, "the peak is more than the image");
        l.admit = [&](uint64_t, std::string& why) { why = "not enough RAM"; return false; };
        SessionStatus st;
        check(!run(l, &st) && st.error == SessionError::memory, "a refusal is a memory failure");
    }
    {   // (prefix reads: test_prefix)
        SavedConversation out = sentinel;
        size_t mb = 0;
        uint64_t rb = 0;
        std::string e;
        ChunkedReadOptions o;
        o.upto = 1000000;
        check(!chunked_read(*fx.pool, p, kId, out, mb, rb, e, {}, o) && !e.empty(), "a prefix beyond the image is refused");
    }
    // raw_bytes_read counts every chunk
    {
        SavedConversation out;
        size_t mb = 0;
        uint64_t rb = 0;
        std::string e;
        check(chunked_read(*fx.pool, p, kId, out, mb, rb, e, {}) && mb == fs::file_size(p), "manifest bytes");
        uint64_t sum = 0;
        for (const auto& r : w.refs) sum += r.raw_len;
        check(rb == sum, "raw bytes read = every chunk's raw length");
    }
    // the progress callback reports the chunks
    {
        uint64_t last = 0, total = 0;
        int n = 0;
        SessionReadLimits l;
        l.progress = [&](uint64_t d, uint64_t t) { last = d; total = t; ++n; };
        check(run(l) && last == total && (size_t) n == w.refs.size(), "read progress after every chunk");
    }
}

void test_write_failures() {
    Fx fx("wfail");
    Spec sp;
    sp.n = 600;
    sp.cps = {200};
    sp.live_gdn = 5000;
    const SavedConversation img = make_image(sp);
    W w;
    check(write_to(fx, fx.man.string(), img, w, 64), "write");
    const auto old_manifest = slurp(fx.man);
    const auto old_refs = w.refs;
    Spec sp2 = sp;
    sp2.n = 650;
    const SavedConversation img2 = make_image(sp2);   // shares chunks with the first
    auto untouched = [&](const char* label) {
        check(slurp(fx.man) == old_manifest, label);
        for (const auto& r : old_refs) check(fx.pool->refs_of(r.key) >= 1, "old chunks keep their refs");
        check(count_prefixed(fx.root, ".manifest.bin") == 0, "no manifest temporary left");
    };
    std::vector<uint32_t> before;
    for (const auto& r : old_refs) before.push_back(fx.pool->refs_of(r.key));
    auto same_refs = [&](const char* label) {
        for (size_t i = 0; i < old_refs.size(); ++i)
            if (fx.pool->refs_of(old_refs[i].key) != before[i]) check(false, label);
    };
    // cancel after a few chunks
    {
        int polls = 0;
        ChunkedWriteOptions o;
        o.cancel = [&] { return ++polls > 4; };
        W f;
        check(!write_to(fx, fx.man.string(), img2, f, 64, o) && f.error.find("cancelled") != std::string::npos, "cancel gives the write up");
        check(f.refs.empty() && f.st.error == SessionError::io, "cancel: no refs, an io status");
        untouched("cancel: the old manifest is untouched");
        same_refs("cancel: the chunks' refs are back where they were");
    }
    // the manifest's steps
    for (const char* step : {"write", "rename"}) {
        ChunkedWriteOptions o;
        o.fault = [&](const char* s) { return !std::strcmp(s, step) ? EIO : 0; };
        W f;
        check(!write_to(fx, fx.man.string(), img2, f, 64, o), "a manifest step fails");
        check(f.refs.empty() && !f.st.published, "failed manifest: no refs, not published");
        untouched("a failed manifest step leaves the old manifest");
        same_refs("a failed manifest step gives the refs back");
    }
    {
        ChunkedWriteOptions o;
        o.fault = [&](const char* s) { return !std::strcmp(s, "write") ? ENOSPC : 0; };
        W f;
        check(!write_to(fx, fx.man.string(), img2, f, 64, o) && f.st.error == SessionError::storage, "ENOSPC on the manifest is a storage failure");
    }
    // a chunk fails in the middle: the manifest is not published
    {
        int n = 0;
        fx.fault = [&](const char* s) { return !std::strcmp(s, "chunk_write") && ++n == 2 ? ENOSPC : 0; };
        Spec fresh;
        fresh.n = 900;
        fresh.seed = 9;       // nothing in common: new chunks are written
        W f;
        check(!write_to(fx, fx.path("fresh.bin"), make_image(fresh), f, 64) && f.st.error == SessionError::storage, "ENOSPC mid-write is a storage failure");
        check(!fs::exists(fx.path("fresh.bin")) && f.refs.empty(), "and publishes nothing");
        fx.fault = nullptr;
        same_refs("the first chunk of the failed write is released");
    }
    // the folder flush after the rename: published, but a failure
    {
        ChunkedWriteOptions o;
        o.durable = true;
        o.fault = [&](const char* s) { return !std::strcmp(s, "dir_flush") ? EIO : 0; };
        W f;
        check(!write_to(fx, fx.path("df.bin"), img2, f, 64, o) && f.st.published && f.st.error == SessionError::io, "a folder flush failure after the rename is reported as published");
        check(f.refs.empty(), "and gives the refs back (the caller treats it as failed)");
        same_refs("flush failure: refs back");
        o.fault = [&](const char* s) { return !std::strcmp(s, "dir_flush") ? EINVAL : 0; };
        check(write_to(fx, fx.path("df2.bin"), img2, f, 64, o) && f.st.dir_flush_unsupported, "EINVAL on the folder flush is not a failure");
        fx.pool->unref(f.refs.data(), f.refs.size());
    }
    // durable: the shard folders are flushed before the manifest
    {
        fx.steps.clear();
        ChunkedWriteOptions o;
        o.durable = true;
        std::vector<std::string> phases;
        o.phase = [&](const char* ph, uint64_t) { phases.push_back(ph); };
        Spec fresh;
        fresh.n = 300;
        fresh.seed = 4;
        W f;
        check(write_to(fx, fx.path("dur.bin"), make_image(fresh), f, 64, o), "durable write");
        check(std::count(fx.steps.begin(), fx.steps.end(), std::string("shard_sync")) >= 1, "durable: the shard folders were flushed");
        check(std::count(fx.steps.begin(), fx.steps.end(), std::string("chunk_flush")) == (long) f.stats.misses, "durable: every new chunk was flushed");
        check(phases == std::vector<std::string>({"flush", "publish"}), "durable: flush then publish phases");
        fx.fault = [](const char* s) { return !std::strcmp(s, "shard_sync") ? EIO : 0; };
        W g;
        Spec other = fresh;
        other.seed = 5;
        check(!write_to(fx, fx.path("dur2.bin"), make_image(other), g, 64, o) && !fs::exists(fx.path("dur2.bin")), "a failed shard flush stops before the manifest");
        fx.fault = nullptr;
    }
    // a progress callback
    {
        uint64_t last = 0, total = 0;
        int calls = 0;
        ChunkedWriteOptions o;
        o.progress = [&](uint64_t d, uint64_t t) { last = d; total = t; ++calls; };
        W f;
        const bool pok = write_to(fx, fx.path("prog.bin"), img2, f, 64, o);
        if (!(pok && last == total && calls == (int) f.refs.size() + 1)) std::fprintf(stderr, "  ok=%d last=%llu total=%llu calls=%d refs=%zu err=%s\n", pok, (unsigned long long) last, (unsigned long long) total, calls, f.refs.size(), f.error.c_str());
        check(pok && last == total && calls == (int) f.refs.size() + 1, "write progress: 0, then after every chunk");
        fx.pool->unref(f.refs.data(), f.refs.size());
    }
    // the good write over the old one still works, and replaces it
    {
        W f;
        check(write_to(fx, fx.man.string(), img2, f, 64), "overwrite");
        SavedConversation back;
        std::string err;
        check(read_from(fx, fx.man.string(), back, err) && same(img2, back), "the replacement reads back");
    }
}
} // namespace

// ------------------------------------------------------------------------------------------------ codec, pipeline

std::vector<uint8_t> shuffled_of(const std::vector<uint8_t>& v, uint8_t w) {
    std::vector<uint8_t> o(v.size());
    shuffle_bytes(v.data(), o.data(), v.size(), w);
    return o;
}


// The prefix image the reader must build for `upto` tokens of the Spec's image, from an independent construction: the
// image of `upto` tokens (it shares the prefix bytes), its live state the checkpoint of that length (used = 0), the
// checkpoints of at most that length, and each main layer's spare row the checkpoint's own `dead`.
SavedConversation expect_prefix(const Spec& sp, int64_t upto) {
    Spec e = sp;
    e.n = upto;
    e.live_gdn = sp.cp_gdn;
    e.cps.clear();
    for (int64_t t : sp.cps) if (t <= upto) e.cps.push_back(t);
    SavedConversation x = make_image(e);
    x.live.used = 0;
    for (size_t l = 0; l < 2; ++l) {
        auto& b = x.kv[l].pooled;
        check(x.live.dead.size() == 2 * kRowBytes, "test: dead holds a row per main layer");
        b.visit(b.size() - kRowBytes, kRowBytes, [&](uint8_t* q, size_t c, size_t at) {
            std::memcpy(q, x.live.dead.data() + l * kRowBytes + (at - (b.size() - kRowBytes)), c);
            return true;
        });
    }
    return x;
}

void test_prefix_codec(ChunkCodec codec) {
    Fx fx(codec == ChunkCodec::zstd ? "prefixz" : "prefix");
    fx.codec = codec;
    fx.make_pool(true);
    Spec sp;
    sp.n = 1500;
    sp.cps = {101, 700, 1103};
    sp.cp_gdn = 5000;
    sp.live_gdn = (4u << 20) + 10;
    const SavedConversation sentinel = make_image(Spec{});
    for (uint32_t T : {64u, 512u}) {
        const SavedConversation img = make_image(sp);
        W w;
        check(write_to(fx, fx.path("full.bin"), img, w, T), "prefix: the full image is written");
        SavedConversation whole;
        size_t mb = 0;
        uint64_t rb_whole = 0;
        std::string e;
        check(chunked_read(*fx.pool, fx.path("full.bin"), kId, whole, mb, rb_whole, e, {}) && same(whole, img), "prefix: whole read");
        for (int64_t upto : {101, 700, 1103}) {
            const SavedConversation want = expect_prefix(sp, upto);
            for (int workers : {1, 3}) {
                SavedConversation out = sentinel;
                ChunkedReadOptions o;
                o.upto = upto;
                o.workers = workers;
                uint64_t rb = 0;
                int asked = 0;
                uint64_t need = 0;
                SessionReadLimits lim;
                lim.admit = [&](uint64_t n, std::string&) { ++asked; need = n; return true; };
                check(chunked_read(*fx.pool, fx.path("full.bin"), kId, out, mb, rb, e, lim, o), "prefix: read");
                check(same(out, want), "prefix: the image is the independent construction, byte for byte");
                check(asked == 1, "prefix: admit once");
                check(rb < rb_whole, "prefix: fewer bytes read than the whole image");
                check(need < chunked_read_peak_bytes(rd<uint64_t>(slurp(fx.path("full.bin")), 48), workers, T), "prefix: admitted for the prefix, not the file");
                check(out.live.ids.size() == (size_t) upto && out.kv[0].cells == (upto + 3) / 4 * 4 &&
                      out.kv[0].pooled_rows == upto / 4 + 1 && out.kv[2].pooled_rows == 0, "prefix: geometry of the prefix");
                // what the engine does next: a checkpoint of exactly `upto` is among the checkpoints
                bool mount = false;
                for (const auto& c : out.checkpoints) mount = mount || (int64_t) c.ids.size() == upto;
                check(mount, "prefix: the checkpoint at upto is in the list");
            }
        }
        // the whole length is the whole image; beyond it, below every checkpoint, or negative is refused and leaves `out`
        {
            SavedConversation out;
            ChunkedReadOptions o;
            uint64_t rb = 0;
            o.upto = 1500;
            check(chunked_read(*fx.pool, fx.path("full.bin"), kId, out, mb, rb, e, {}, o) && same(out, img), "prefix: upto == the live length is the whole image");
            for (int64_t bad : {1501, 300, 100, 1, -5}) {
                SavedConversation keep = sentinel;
                o.upto = bad;
                SessionStatus st;
                check(!chunked_read(*fx.pool, fx.path("full.bin"), kId, keep, mb, rb, e, {}, o, &st) && !e.empty() && same(keep, sentinel),
                      "prefix: no checkpoint of that length: refused, output untouched");
            }
        }
        // only the chunks the prefix needs are read: the deepest slab may be gone
        {
            // slab refs come first in manifest order: the last slab is index n_slabs - 1
            const uint64_t n_slabs = (uint64_t) ((1500 + T - 1) / T);
            const ChunkRef deep = w.refs[n_slabs - 1];
            const fs::path f = fx.pool->path_of(deep.key);
            const auto bytes = slurp(f);
            fs::remove(f);
            fx.make_pool(true);   // forget the chunk
            SavedConversation out;
            ChunkedReadOptions o;
            uint64_t rb = 0;
            o.upto = 700;
            check(chunked_read(*fx.pool, fx.path("full.bin"), kId, out, mb, rb, e, {}, o) && same(out, expect_prefix(sp, 700)),
                  "prefix: a chunk past the prefix is not needed");
            SavedConversation all;
            check(!chunked_read(*fx.pool, fx.path("full.bin"), kId, all, mb, rb, e, {}) && e.find("missing") != std::string::npos, "prefix: the whole read needs it");
            spit(f, bytes);   // back (the pool finds it again)
            fx.make_pool(true);
            check(chunked_read(*fx.pool, fx.path("full.bin"), kId, all, mb, rb, e, {}) && same(all, img), "prefix: whole read again");
        }
        // write "as of" a checkpoint: the image of that length, from an image and from sources, and the same as the prefix read
        for (int64_t upto : {101, 700, 1103}) {
            ChunkedWriteOptions o;
            o.as_of_tokens = upto;
            SavedConversation meta = img;
            meta.kv.clear();
            auto src = sources_of(img);
            W wi, ws;
            check(write_to(fx, fx.path("a.bin"), img, wi, T, o), "as-of: image write");
            check(write_to(fx, fx.path("b.bin"), meta, ws, T, o, &src), "as-of: source write");
            check(slurp(fx.path("a.bin")) == slurp(fx.path("b.bin")) && wi.refs == ws.refs, "as-of: sources == image, byte for byte");
            SavedConversation back, pre;
            check(read_from(fx, fx.path("a.bin"), back, e) && same(back, expect_prefix(sp, upto)), "as-of: reads back as the prefix image");
            ChunkedReadOptions po;
            po.upto = upto;
            uint64_t rb = 0;
            check(chunked_read(*fx.pool, fx.path("full.bin"), kId, pre, mb, rb, e, {}, po) && same(pre, back), "as-of: == the prefix read of the full image");
            check(fs::file_size(fx.path("a.bin")) < fs::file_size(fx.path("full.bin")) + 1, "as-of: manifest is not larger");
            if (upto >= 700) check(wi.stats.raw_bytes < w.stats.raw_bytes, "as-of: writes fewer bytes");
            check(wi.stats.hits > 0, "as-of: the full image's slabs are hit");
        }
        for (int64_t bad : {300, 1500, 2000, -1}) {
            ChunkedWriteOptions o;
            o.as_of_tokens = bad;
            W wf;
            fs::remove(fx.path("c.bin"));
            const bool ok = write_to(fx, fx.path("c.bin"), img, wf, T, o);
            if (bad == -1) check(ok, "as-of: a negative value is no request");   // (= 0)
            else check(!ok && wf.refs.empty() && !fs::exists(fx.path("c.bin")), "as-of: no such checkpoint: refused, nothing published");
        }
    }
}

void test_prefix() {
    test_prefix_codec(ChunkCodec::raw);
    if (codec_available(ChunkCodec::zstd)) { g_low_entropy = true; test_prefix_codec(ChunkCodec::zstd); g_low_entropy = false; }
}

void test_shuffle_and_skip() {
    // the layout: plane b holds byte b of every element; a tail stays
    const std::vector<uint8_t> a{0xa0, 0xa1, 0xa2, 0xa3, 0xb0, 0xb1, 0xb2, 0xb3, 0xee, 0xff};
    check(shuffled_of(a, 4) == std::vector<uint8_t>({0xa0, 0xb0, 0xa1, 0xb1, 0xa2, 0xb2, 0xa3, 0xb3, 0xee, 0xff}), "shuffle width 4 layout, tail kept");
    check(shuffled_of(a, 2) == std::vector<uint8_t>({0xa0, 0xa2, 0xb0, 0xb2, 0xee, 0xa1, 0xa3, 0xb1, 0xb3, 0xff}), "shuffle width 2 layout");
    check(shuffled_of(a, 0) == a && shuffled_of(a, 1) == a, "width 0 and 1 copy");
    for (uint8_t w : {0, 1, 2, 3, 4, 5, 8})
        for (size_t n : {0, 1, 3, 7, 64, 1001, 4099}) {
            const auto v = fill_vec(n, 9 + w);
            std::vector<uint8_t> sh = shuffled_of(v, w), back(n);
            unshuffle_bytes(sh.data(), back.data(), n, w);
            check(back == v, "unshuffle(shuffle(x)) == x");
            if (n >= (size_t) w && w > 1) {
                std::multiset<uint8_t> x(v.begin(), v.end()), y(sh.begin(), sh.end());
                check(x == y, "a shuffle permutes the bytes");
            }
        }
    // adaptive skip
    AdaptiveSkip sk;
    check(!sk.attempt(0, 1000) && !sk.attempt(0, AdaptiveSkip::kMinSegment - 1), "a small segment is never tried");
    check(sk.attempt(0, AdaptiveSkip::kMinSegment), "a fresh kind is tried");
    sk.record(0, 1000000, 990000);
    check(sk.ratio(0) > 0.93, "the ratio is recorded");
    int tried = 0;
    for (int i = 0; i < 16; ++i) tried += sk.attempt(0, 1 << 20) ? 1 : 0;
    check(tried == 2, "a kind that does not pay is probed every 8th time");
    check(sk.attempt(1, 1 << 20), "kinds are independent");
    for (int i = 0; i < 8; ++i) sk.record(0, 1000000, 400000);
    check(sk.ratio(0) < 0.5 && sk.attempt(0, 1 << 20), "a kind that pays is tried again");
    // concurrent use (a data race here is what the TSAN run is for)
    std::vector<std::thread> th;
    for (int t = 0; t < 4; ++t) th.emplace_back([&] { for (int i = 0; i < 2000; ++i) { sk.attempt(i % 8, 1 << 20); sk.record(i % 8, 100000, 50000 + i % 40000); } });
    for (auto& t : th) t.join();
    check(true, "concurrent skip use");
}

std::vector<uint8_t> seg(uint32_t rl, uint32_t sl, uint8_t mode, uint8_t width, const std::vector<uint8_t>& body, uint16_t rsv = 0) {
    std::vector<uint8_t> o(12);
    wr<uint32_t>(o, 0, rl); wr<uint32_t>(o, 4, sl); o[8] = mode; o[9] = width; wr<uint16_t>(o, 10, rsv);
    o.insert(o.end(), body.begin(), body.end());
    return o;
}
std::vector<uint8_t> cat(std::vector<uint8_t> a, const std::vector<uint8_t>& b) { a.insert(a.end(), b.begin(), b.end()); return a; }

void test_codec_units() {
    check(codec_available(ChunkCodec::raw) && !codec_available(ChunkCodec::lz4), "raw is always there, lz4 never");
    const bool zs = codec_available(ChunkCodec::zstd);
    std::printf("  zstd in this build: %s\n", zs ? "yes" : "no");
    // raw decode
    {
        const auto v = fill_vec(100, 1);
        std::vector<uint8_t> out(100);
        check(codec_decode(ChunkCodec::raw, false, 0, v.data(), 100, out.data(), 100) && out == v, "raw decode");
        check(!codec_decode(ChunkCodec::raw, false, 0, v.data(), 99, out.data(), 100), "raw decode refuses a length mismatch");
        check(!codec_decode(ChunkCodec::raw, true, 4, v.data(), 100, out.data(), 100), "raw decode refuses a shuffled flag");
        check(!codec_decode(ChunkCodec::lz4, false, 0, v.data(), 100, out.data(), 100), "lz4 is refused");
    }
    // the container without a codec: raw segments decode in every build; anything else is refused cleanly
    {
        const auto x = fill_vec(300, 2), y = fill_vec(500, 3);
        auto good = cat(seg(300, 300, 0, 0, x), seg(500, 500, 0, 0, y));
        std::vector<uint8_t> out(800), want = x;
        want.insert(want.end(), y.begin(), y.end());
        check(segmented_decode(good.data(), good.size(), out.data(), 800) && out == want, "a container of raw segments decodes");
        auto bad = [&](const std::vector<uint8_t>& c, size_t rl, const char* label) {
            std::vector<uint8_t> o(rl);
            check(!segmented_decode(c.data(), c.size(), o.data(), rl), label);
        };
        bad(good, 801, "a container shorter than raw_len is refused");
        bad(good, 799, "a container longer than raw_len is refused");
        bad(std::vector<uint8_t>(good.begin(), good.end() - 1), 800, "a truncated container is refused");
        bad(std::vector<uint8_t>(good.begin(), good.begin() + 11), 800, "a truncated header is refused");
        bad(cat(good, {1, 2, 3}), 800, "trailing bytes are refused");
        bad(seg(0, 0, 0, 0, {}), 0 + 1, "a zero-length segment is refused");
        bad(seg(300, 301, 0, 0, fill_vec(301, 1)), 300, "raw: stored != raw is refused");
        bad(seg(300, 300, 0, 4, x), 300, "raw: a width is refused");
        bad(seg(300, 300, 0, 0, x, 1), 300, "reserved bytes must be zero");
        bad(seg(300, 300, 9, 0, x), 300, "an unknown mode is refused");
        bad(seg(300, 5000, 0, 0, x), 300, "a stored length past the end is refused");
        bad(seg(900, 900, 0, 0, fill_vec(900, 1)), 800, "a raw length past the capacity is refused");
        bad(seg(300, 40, 1, 0, fill_vec(40, 4)), 300, "a bogus zstd frame is refused (or no zstd here)");
        bad(seg(300, 40, 2, 4, fill_vec(40, 4)), 300, "a bogus shuffled frame is refused (or no zstd here)");
        bad(seg(300, 40, 2, 3, fill_vec(40, 4)), 300, "a shuffled segment of width 3 is refused");
        bad(seg(300, 40, 1, 4, fill_vec(40, 4)), 300, "a plain zstd segment with a width is refused");
    }
    if (!zs) {
        std::vector<uint8_t> out;
        bool raw = false;
        const auto v = fill_vec(100000, 5);
        check(codec_encode(ChunkCodec::zstd, 1, 4, v.data(), v.size(), out, raw) && raw, "no zstd: encode stores raw");
        std::vector<CodecSegment> segs{{100000, 4, 0}};
        bool sh = true;
        check(segmented_encode(1, segs.data(), 1, v.data(), v.size(), nullptr, out, sh) && !sh && out.size() == v.size() + 12, "no zstd: the container holds raw segments");
        return;
    }
    // zstd: single segment
    for (uint8_t w : {0, 2, 4}) {
        g_low_entropy = true;
        const auto v = fill_vec(200000, 6);
        g_low_entropy = false;
        std::vector<uint8_t> enc, back(v.size());
        bool raw = true;
        check(codec_encode(ChunkCodec::zstd, 1, w, v.data(), v.size(), enc, raw) && !raw && enc.size() < v.size() * 6 / 10, "zstd encodes compressible bytes");
        check(codec_decode(ChunkCodec::zstd, w > 1, w, enc.data(), enc.size(), back.data(), back.size()) && back == v, "zstd round trip");
        std::vector<uint8_t> small(v.size() - 1), big(v.size() + 1);
        check(!codec_decode(ChunkCodec::zstd, w > 1, w, enc.data(), enc.size(), small.data(), small.size()), "a frame larger than the capacity is refused");
        check(!codec_decode(ChunkCodec::zstd, w > 1, w, enc.data(), enc.size(), big.data(), big.size()), "a frame smaller than raw_len is refused");
        auto torn = enc;
        torn.resize(torn.size() / 2);
        check(!codec_decode(ChunkCodec::zstd, w > 1, w, torn.data(), torn.size(), back.data(), back.size()), "a torn frame is refused");
        for (size_t at : {size_t(0), enc.size() / 3, enc.size() - 1}) {
            auto bad = enc;
            bad[at] ^= 0x55;
            (void) codec_decode(ChunkCodec::zstd, w > 1, w, bad.data(), bad.size(), back.data(), back.size());   // (never crashes; the chunk's key check catches a silent change)
            check(true, "a flipped byte does not crash the decoder");
        }
        check(!codec_decode(ChunkCodec::zstd, true, 0, enc.data(), enc.size(), back.data(), back.size()), "shuffled without a width is refused");
    }
    {
        const auto v = fill_vec(200000, 7);   // random
        std::vector<uint8_t> enc;
        bool raw = false;
        check(codec_encode(ChunkCodec::zstd, 1, 0, v.data(), v.size(), enc, raw) && raw, "random bytes: stored as they are");
    }
    // shuffling helps f32-like data (constant high bytes) and never hurts the round trip
    {
        std::vector<uint8_t> f(400000);
        for (size_t i = 0; i < f.size(); i += 4) { f[i] = hb(1, i); f[i + 1] = hb(2, i) & 0x0f; f[i + 2] = 0x3f; f[i + 3] = 0x40; }
        std::vector<uint8_t> a, b, back(f.size());
        bool raw = false;
        check(codec_encode(ChunkCodec::zstd, 1, 0, f.data(), f.size(), a, raw) && !raw, "f32-like bytes compress");
        check(codec_encode(ChunkCodec::zstd, 1, 4, f.data(), f.size(), b, raw) && !raw && b.size() <= a.size(), "and shuffled they do not compress worse");
        check(codec_decode(ChunkCodec::zstd, true, 4, b.data(), b.size(), back.data(), back.size()) && back == f, "shuffled f32 round trip");
    }
    // the container: mixed segments (a small one stays raw, a random one stays raw, the compressible ones are packed)
    {
        g_low_entropy = true;
        auto k = fill_vec(100000, 11), kscale = fill_vec(40000, 12), tiny = fill_vec(5000, 13);
        g_low_entropy = false;
        const auto rnd = fill_vec(70000, 14);
        std::vector<uint8_t> raw_all;
        for (const auto& p : {std::cref(k), std::cref(tiny), std::cref(kscale), std::cref(rnd)}) raw_all.insert(raw_all.end(), p.get().begin(), p.get().end());
        std::vector<CodecSegment> segs{{100000, 0, 0}, {5000, 4, 1}, {40000, 2, 2}, {70000, 0, 3}};
        AdaptiveSkip sk;
        std::vector<uint8_t> enc, back(raw_all.size());
        bool sh = false;
        check(segmented_encode(1, segs.data(), segs.size(), raw_all.data(), raw_all.size(), &sk, enc, sh), "container encodes");
        check(sh, "a shuffled segment is reported");
        check(enc.size() < raw_all.size() * 7 / 10, "the container is smaller");
        check(segmented_decode(enc.data(), enc.size(), back.data(), back.size()) && back == raw_all, "container round trip");
        // walk the segment headers: modes 1, 0 (too small), 2, 0 (random)
        std::vector<uint8_t> modes;
        for (size_t pos = 0; pos < enc.size();) { modes.push_back(enc[pos + 8]); pos += 12 + rd<uint32_t>(enc, pos + 4); }
        check(modes == std::vector<uint8_t>({1, 0, 2, 0}), "the segments are stored on their own terms");
        // a lying raw_len: decoding must not write past the destination
        auto lie = enc;
        wr<uint32_t>(lie, 0, 1000);
        std::vector<uint8_t> small(raw_all.size());
        check(!segmented_decode(lie.data(), lie.size(), small.data(), small.size()), "a lying segment length is refused");
        // sums that do not match the data
        std::vector<CodecSegment> wrong{{100, 0, 0}};
        check(!segmented_encode(1, wrong.data(), 1, raw_all.data(), 99, nullptr, enc, sh), "segments must add up to the length");
        // without the skip hint, with the hint after incompressible history
        AdaptiveSkip cold;
        for (int i = 0; i < 4; ++i) cold.record(3, 1 << 20, 1 << 20);
        std::vector<CodecSegment> one{{100000, 0, 3}};
        check(segmented_encode(1, one.data(), 1, k.data(), 100000, &cold, enc, sh) && enc[8] == 0, "a kind that did not pay is stored raw without trying");
    }
}

std::vector<uint8_t> chunk_header_reseal(std::vector<uint8_t> f) {
    wr<uint64_t>(f, 40, session_hash64(f.data(), 40, 0));
    return f;
}

void test_codec_chunks() {
    const bool zs = codec_available(ChunkCodec::zstd);
    Fx fx("codec");
    fx.codec = ChunkCodec::zstd;
    fx.make_pool(true);
    Fx rawfx("codecraw");
    g_low_entropy = true;
    const auto low = fill_vec(3u << 20, 21);
    g_low_entropy = false;
    const auto rnd = fill_vec(1u << 20, 22);
    ChunkRef r, rr;
    ChunkPutInfo info;
    std::string err;
    check(fx.pool->put(ChunkKind::state_slice, low.data(), low.size(), 4, r, info, err), "put compressible");
    check(rawfx.pool->put(ChunkKind::state_slice, low.data(), low.size(), 4, rr, info, err), "put into the raw pool");
    check(r.key == rr.key && r.raw_len == rr.raw_len, "the key does not depend on the codec");
    std::vector<uint8_t> out(low.size());
    if (zs) {
        check(r.stored_len < r.raw_len * 6 / 10, "the chunk is stored smaller");
        check(fs::file_size(fx.pool->path_of(r.key)) == chunk_file_size(r.stored_len), "and its file follows the size formula");
        ChunkPutInfo i2;
        ChunkRef r2;
        check(fx.pool->put(ChunkKind::state_slice, low.data(), low.size(), 4, r2, i2, err) && i2.hit && r2 == r, "a second put is a hit");
        const auto file = slurp(fx.pool->path_of(r.key));
        check(file[16] == 2 && (file[17] & 3) == 3, "header: codec 2, shuffled + segmented flags");
        check(fx.pool->get(ChunkKind::state_slice, r, out.data(), out.size(), err) && out == low, "zstd chunk reads back");
        // a pool without the codec option reads it too (the codec is in the chunk)
        Fx other("codecother");
        fs::remove_all(other.pool_dir);
        fs::copy(fx.pool_dir, other.pool_dir, fs::copy_options::recursive);
        other.make_pool(true);
        std::fill(out.begin(), out.end(), 0);
        check(other.pool->get(ChunkKind::state_slice, r, out.data(), out.size(), err) && out == low, "a raw-codec pool reads a zstd chunk");
        // corruption
        auto flip = [&](size_t at, uint8_t x, bool reseal, const char* label) {
            auto f = file;
            f[at] ^= x;
            if (reseal) f = chunk_header_reseal(f);
            fs::path p = fx.pool->path_of(r.key);
            spit(p, f);
            std::fill(out.begin(), out.end(), 0);
            check(!fx.pool->get(ChunkKind::state_slice, r, out.data(), out.size(), err), label);
            spit(p, file);
        };
        flip(4096 + 5, 0x40, false, "a flipped payload byte is refused");
        flip(4096 + 12 + 100, 0x01, false, "a flipped frame byte is refused");
        flip(4096 + r.stored_len - 1, 0x80, false, "a flipped last byte is refused");
        flip(17, 2, true, "the segmented flag cleared on a zstd chunk is refused");
        {   // the shuffled flag is informational (each segment says its own mode): clearing it changes nothing
            auto f = file;
            f[17] &= ~1;
            spit(fx.pool->path_of(r.key), chunk_header_reseal(f));
            std::fill(out.begin(), out.end(), 0);
            check(fx.pool->get(ChunkKind::state_slice, r, out.data(), out.size(), err) && out == low, "the shuffled flag is informational");
            spit(fx.pool->path_of(r.key), file);
        }
        flip(16, 0x01, true, "an unknown codec is refused");
        flip(16, 0x03, true, "lz4 is refused");
        check(fx.pool->get(ChunkKind::state_slice, r, out.data(), out.size(), err) && out == low, "restored chunk reads");
        // random bytes stay raw
        ChunkRef r3;
        check(fx.pool->put(ChunkKind::kv_slab, rnd.data(), rnd.size(), 0, r3, info, err) && info.codec == ChunkCodec::raw && r3.stored_len == r3.raw_len && !info.shuffled, "incompressible bytes are stored raw");
    } else {
        check(r.stored_len == r.raw_len && info.codec == ChunkCodec::raw, "no zstd: the writer emits a raw chunk");
        check(slurp(fx.pool->path_of(r.key)) == slurp(rawfx.pool->path_of(rr.key)), "byte-identical to the raw pool's");
    }
    // a codec-2 chunk made by hand out of raw segments reads in every build; one with a zstd segment is a clean refusal
    {
        const auto v = fill_vec(50000, 23);
        const ChunkKey key = chunk_key(v.data(), v.size());
        auto mk = [&](const std::vector<uint8_t>& payload, const ChunkKey& k, uint32_t raw_len) {
            std::vector<uint8_t> f(chunk_file_size((uint32_t) payload.size()), 0);
            std::memcpy(f.data(), "STRCHNK1", 8);
            wr<uint32_t>(f, 8, raw_len); wr<uint32_t>(f, 12, (uint32_t) payload.size());
            f[16] = 2; f[17] = 2;
            std::memcpy(f.data() + 24, k.b.data(), 16);
            f = chunk_header_reseal(f);
            std::memcpy(f.data() + 4096, payload.data(), payload.size());
            return f;
        };
        Fx hx("codechand");
        ChunkRef ref;
        ref.key = key; ref.raw_len = (uint32_t) v.size();
        auto ok_payload = seg((uint32_t) v.size(), (uint32_t) v.size(), 0, 0, v);
        ref.stored_len = (uint32_t) ok_payload.size();
        // stored_len must be <= raw_len for a chunk: pad the check by splitting the data in two segments of a tiny header cost
        // (a container is larger than its raw bytes only when nothing was packed; a real writer would have stored raw)
        const bool header_ok = ref.stored_len <= ref.raw_len;
        if (!header_ok) {
            // an all-raw container is bigger than raw_len: the reader refuses it (stored_len over raw_len)
            fs::create_directories(fs::path(hx.pool->path_of(key)).parent_path());
            spit(hx.pool->path_of(key), mk(ok_payload, key, (uint32_t) v.size()));
            std::string e;
            check(!hx.pool->get(ChunkKind::kv_slab, ref, out.data(), out.size(), e), "a codec-2 chunk larger than its raw bytes is refused");
        }
    }
}

// the options of a pipeline test: one image big enough for 16 KiB codec segments
Spec big_spec(bool low) {
    (void) low;
    Spec sp;
    sp.n = 9000;
    sp.cps = {2500, 6000};
    sp.live_gdn = (4u << 20) + 4097;
    sp.cp_gdn = (4u << 20) / 2;
    return sp;
}

struct Pw {
    bool ok = false;
    std::vector<uint8_t> manifest;
    W w;
};
Pw write_with(Fx& fx, const SavedConversation& img, const char* leaf, int workers, uint32_t T = 2048, ChunkedWriteOptions o = {}) {
    Pw r;
    o.workers = workers;
    r.ok = write_to(fx, fx.path(leaf), img, r.w, T, o);
    if (r.ok) r.manifest = slurp(fx.path(leaf));
    return r;
}

void test_pipeline(ChunkCodec codec) {
    const bool zs = codec == ChunkCodec::zstd && codec_available(codec);
    const char* name = zs ? "zstd" : "raw";
    std::printf("  pipeline, codec %s\n", name);
    for (bool low : {false, true}) {
        g_low_entropy = low;
        const Spec sp = big_spec(low);
        const SavedConversation img = make_image(sp);
        g_low_entropy = false;
        Fx base("pipe-base");
        base.codec = codec;
        base.make_pool(true);
        const Pw one = write_with(base, img, "m.bin", 1);
        check(one.ok, "workers=1 writes");
        if (zs && low) check(one.w.stats.new_file_bytes < one.w.stats.raw_bytes * 8 / 10, "zstd: compressible data takes fewer file bytes");
        if (zs && !low) check(one.w.stats.new_file_bytes > one.w.stats.raw_bytes * 95 / 100, "zstd: random data stays about raw");
        for (int workers : {2, 3, 4, 8}) {
            Fx fx("pipe-w");
            fx.codec = codec;
            fx.make_pool(true);
            std::atomic<int> inside{0};
            std::atomic<bool> overlap{false};
            uint64_t last = 0, total = 0;
            bool monotonic = true;
            ChunkedWriteOptions o;
            o.progress = [&](uint64_t done, uint64_t all) {
                if (inside++) overlap = true;
                if (done < last) monotonic = false;
                last = done; total = all;
                --inside;
            };
            const Pw p = write_with(fx, img, "m.bin", workers, 2048, o);
            check(p.ok, "a pipelined write succeeds");
            check(!overlap && monotonic && last == total && total > 0, "progress: serialized, monotonic, complete");
            if (!zs) check(p.manifest == one.manifest, "raw: the manifest is byte-identical to the serial one");
            check(p.w.stats.chunks == one.w.stats.chunks && p.w.stats.raw_bytes == one.w.stats.raw_bytes &&
                  p.w.stats.misses == one.w.stats.misses && p.w.refs.size() == one.w.refs.size(), "the same chunks");
            for (size_t i = 0; i < p.w.refs.size(); ++i) check(p.w.refs[i].key == one.w.refs[i].key && p.w.refs[i].raw_len == one.w.refs[i].raw_len, "refs in manifest order");
            for (int rw : {1, 2, 4}) {
                SavedConversation back;
                std::string err;
                size_t mb = 0;
                uint64_t rb = 0;
                ChunkedReadOptions ro;
                ro.workers = rw;
                SessionReadLimits lim;
                std::atomic<int> in2{0};
                std::atomic<bool> ov2{false};
                uint64_t lastd = 0;
                lim.progress = [&](uint64_t d, uint64_t) { if (in2++) ov2 = true; lastd = std::max(lastd, d); --in2; };
                check(chunked_read(*fx.pool, fx.path("m.bin"), kId, back, mb, rb, err, lim, ro) && same(img, back), "a pipelined read reproduces the image");
                check(!ov2 && rb == lastd, "read progress is serialized and complete");
            }
        }
        // writes of the same image through sources, pipelined
        {
            Fx fx("pipe-src");
            fx.codec = codec;
            fx.make_pool(true);
            const auto srcs = sources_of(img);
            SavedConversation meta = img;
            for (auto& k : meta.kv) { k.k = {}; k.v = {}; k.k_scale = {}; k.v_scale = {}; k.pooled = {}; }
            // (the meta carries no K/V: the sources do)
            meta.kv.clear();
            W w;
            ChunkedWriteOptions o;
            o.workers = 4;
            check(write_to(fx, fx.path("s.bin"), meta, w, 2048, o, &srcs), "pipelined write from sources");
            SavedConversation back;
            std::string err;
            check(read_from(fx, fx.path("s.bin"), back, err) && same(img, back), "and it reads back");
            if (!zs) check(slurp(fx.path("s.bin")) == one.manifest, "sources and image give the same manifest");
        }
    }
    g_low_entropy = false;
}

void test_pipeline_failures(ChunkCodec codec) {
    g_low_entropy = true;
    const SavedConversation img = make_image(big_spec(true));
    g_low_entropy = false;
    for (int workers : {1, 2, 4}) {
        {   // cancel part-way
            Fx fx("pf-cancel");
            fx.codec = codec;
            fx.make_pool(true);
            std::atomic<int> polls{0};
            ChunkedWriteOptions o;
            o.cancel = [&] { return ++polls > 5; };
            const Pw p = write_with(fx, img, "m.bin", workers, 2048, o);
            check(!p.ok && p.w.error.find("cancelled") != std::string::npos && p.w.refs.empty(), "cancel under workers gives the write up");
            check(fx.pool->live_bytes() == 0 && !fs::exists(fx.path("m.bin")), "cancel: every ref is back, no manifest");
            check(count_prefixed(fx.root, ".tmp-") == 0, "cancel: no temporary chunk file left");
            // the same pool writes the image afterwards
            const Pw q = write_with(fx, img, "m.bin", workers);
            SavedConversation back;
            std::string err;
            check(q.ok && read_from(fx, fx.path("m.bin"), back, err) && same(img, back), "and writes it whole afterwards");
        }
        {   // a chunk's rename fails (the 4th)
            Fx fx("pf-rename");
            fx.codec = codec;
            fx.make_pool(true);
            std::atomic<int> n{0};
            fx.fault = [&](const char* s) { return !std::strcmp(s, "chunk_rename") && ++n == 4 ? EIO : 0; };
            const Pw p = write_with(fx, img, "m.bin", workers);
            check(!p.ok && p.w.refs.empty() && fx.pool->live_bytes() == 0, "a failing chunk gives the whole write up and releases the refs");
            check(count_prefixed(fx.root, ".tmp-") == 0, "no temporary chunk file left after a failed rename");
        }
        {   // not enough space for a chunk
            Fx fx("pf-space");
            fx.codec = codec;
            fx.make_pool(true, 600, ~uint64_t(0) >> 2);
            const Pw p = write_with(fx, img, "m.bin", workers);
            check(!p.ok && fx.pool->live_bytes() == 0 && p.w.st.error == SessionError::storage, "no space: a storage refusal, refs back");
        }
        {   // the K/V source fails at the 3rd slab
            Fx fx("pf-src");
            fx.codec = codec;
            fx.make_pool(true);
            auto srcs = sources_of(img);
            std::atomic<int> reads{0};
            for (auto& s : srcs) {
                auto inner = s.read;
                s.read = [&reads, inner](size_t part, size_t off, void* dst, size_t n) { return ++reads <= 40 ? inner(part, off, dst, n) : false; };
                s.error = [] { return std::string("device gone"); };
            }
            SavedConversation meta = img;
            meta.kv.clear();
            W w;
            ChunkedWriteOptions o;
            o.workers = workers;
            check(!write_to(fx, fx.path("m.bin"), meta, w, 2048, o, &srcs) && w.error.find("device gone") != std::string::npos, "a failing source gives the write up");
            check(fx.pool->live_bytes() == 0 && w.refs.empty(), "and the refs are back");
        }
    }
    {   // a bad chunk in the middle of a pipelined read: refused, the output untouched
        Fx fx("pf-read");
        fx.codec = codec;
        fx.make_pool(true);
        const Pw p = write_with(fx, img, "m.bin", 4);
        check(p.ok, "write for the read failure");
        const auto victim = p.w.refs[p.w.refs.size() / 2];
        auto f = slurp(fx.pool->path_of(victim.key));
        f[4096 + victim.stored_len / 2] ^= 0x10;
        spit(fx.pool->path_of(victim.key), f);
        for (int rw : {1, 4}) {
            SavedConversation back;
            back.live.used = 77;
            std::string err;
            size_t mb = 0;
            uint64_t rb = 0;
            ChunkedReadOptions ro;
            ro.workers = rw;
            SessionStatus st;
            check(!chunked_read(*fx.pool, fx.path("m.bin"), kId, back, mb, rb, err, {}, ro, &st) && !err.empty() && back.live.used == 77, "a corrupt chunk fails a pipelined read and leaves the output alone");
        }
    }
    g_low_entropy = false;
}

void test_pipeline_concurrent_writers(ChunkCodec codec) {
    // four threads write overlapping conversations (and read each other's) through one pool, each pipelined
    Fx fx("pc");
    fx.codec = codec;
    fx.make_pool(true);
    std::vector<SavedConversation> imgs;
    g_low_entropy = true;
    for (int i = 0; i < 4; ++i) {
        Spec sp;
        sp.n = 3000 + 700 * i;
        sp.cps = {800, 1500};
        sp.live_gdn = (4u << 20) + 17;
        imgs.push_back(make_image(sp));
    }
    g_low_entropy = false;
    std::atomic<int> bad{0};
    std::vector<std::thread> th;
    for (int t = 0; t < 4; ++t)
        th.emplace_back([&, t] {
            for (int round = 0; round < 3; ++round) {
                W w;
                ChunkedWriteOptions o;
                o.workers = 2;
                o.chunk_tokens = 1024;
                const std::string path = fx.path("c" + std::to_string(t) + ".bin");
                if (!chunked_write(*fx.pool, path, imgs[(size_t) t], nullptr, kId, o, w.refs, w.stats, w.error, &w.st)) { ++bad; continue; }
                SavedConversation back;
                size_t mb = 0;
                uint64_t rb = 0;
                std::string err;
                ChunkedReadOptions ro;
                ro.workers = 2;
                if (!chunked_read(*fx.pool, path, kId, back, mb, rb, err, {}, ro) || !same(imgs[(size_t) t], back)) ++bad;
                fx.pool->unref(w.refs.data(), w.refs.size());
            }
        });
    for (auto& t : th) t.join();
    check(bad == 0, "concurrent pipelined writers and readers on one pool");
    check(fx.pool->live_bytes() == 0, "all refs given back");
}

void test_codec_suite() {
    test_pipeline(ChunkCodec::raw);
    test_pipeline_failures(ChunkCodec::raw);
    test_pipeline_concurrent_writers(ChunkCodec::raw);
    test_pipeline(ChunkCodec::zstd);   // without zstd in the build the writer emits raw: the same checks hold
    test_pipeline_failures(ChunkCodec::zstd);
    test_pipeline_concurrent_writers(ChunkCodec::zstd);
}


// the review round: corrupt chunks are repaired, durable hits are made durable, the peak estimate, the shared stamp
void test_review_fixes() {
    std::string err;
    ChunkPutInfo info;
    {   // a chunk whose payload rotted is moved aside by the read that finds it, and the next put writes a fresh one
        Fx fx("rot");
        ChunkPool& pool = *fx.pool;
        std::vector<uint8_t> raw = fill_vec(9000, 21);
        ChunkRef r;
        check(pool.put(ChunkKind::kv_slab, raw.data(), raw.size(), 0, r, info, err), "rot: put");
        auto f = slurp(pool.path_of(r.key));
        f[kChunkAlign + 10] ^= 0x40;   // the payload, past the header block a hit looks at
        spit(pool.path_of(r.key), f);
        std::vector<uint8_t> back(raw.size());
        check(!pool.get(ChunkKind::kv_slab, r, back.data(), back.size(), err) && err.find("content hash mismatch") != std::string::npos,
              "rot: the read fails on the hash");
        check(!fs::exists(pool.path_of(r.key)) && count_prefixed(fx.pool_dir / r.key.hex().substr(0, 2), ".tmp-bad-") == 1,
              "rot: the bad file is moved aside (a leftover the next open sweeps)");
        check(!pool.exists(r), "rot: the reader's exists() check now misses, so a manifest naming it is not trusted");
        ChunkRef again;
        check(pool.put(ChunkKind::kv_slab, raw.data(), raw.size(), 0, again, info, err) && !info.hit && again == r,
              "rot: a put of the same bytes writes a fresh chunk, not a hit on the bad one");
        check(pool.get(ChunkKind::kv_slab, r, back.data(), back.size(), err) && back == raw, "rot: and it reads back");
        check(pool.refs_of(r.key) == 2, "rot: the references the manifests hold are kept (one more for the new writer)");
        // a ghost that rots goes from the index too
        std::vector<uint8_t> raw2 = fill_vec(5000, 22);
        ChunkRef r2;
        check(pool.put(ChunkKind::kv_slab, raw2.data(), raw2.size(), 0, r2, info, err), "rot: put 2");
        pool.unref(&r2, 1);
        auto f2 = slurp(pool.path_of(r2.key));
        f2[kChunkAlign + 3] ^= 1;
        spit(pool.path_of(r2.key), f2);
        const size_t n = pool.count();
        std::vector<uint8_t> back2(raw2.size());
        check(!pool.get(ChunkKind::kv_slab, r2, back2.data(), back2.size(), err) && pool.count() == n - 1, "rot: an unreferenced bad chunk leaves the index");
    }
    {   // durable: a hit on a chunk a non-durable write made flushes it and its folder; a failed folder sync is not forgotten
        Fx fx("durable");
        ChunkPool& pool = *fx.pool;
        std::vector<uint8_t> raw = fill_vec(7000, 31);
        ChunkRef r;
        check(pool.put(ChunkKind::kv_slab, raw.data(), raw.size(), 0, r, info, err, false), "durable: a non-durable put");
        auto count = [&](const char* step) { std::lock_guard<std::mutex> g(fx.steps_mu); return (int) std::count(fx.steps.begin(), fx.steps.end(), std::string(step)); };
        check(pool.sync_touched_shards(err), "durable: (the pool folder the first chunk created is flushed)");
        const int base = count("shard_sync");
        const int flushes = count("chunk_flush");
        ChunkRef r2;
        check(pool.put(ChunkKind::kv_slab, raw.data(), raw.size(), 0, r2, info, err, true) && info.hit, "durable: a durable put hits it");
        check(count("chunk_flush") == flushes + 1, "durable: the hit chunk is flushed (fdatasync)");
        check(pool.sync_touched_shards(err) && count("shard_sync") == base + 1, "durable: and its shard folder is queued for the sync");
        // a failed folder sync: the same shards are flushed again by the retry
        check(pool.put(ChunkKind::kv_slab, raw.data(), raw.size(), 0, r2, info, err, true) && info.hit, "durable: hit again");
        fx.fault = [](const char* s) { return !std::strcmp(s, "shard_sync") ? EIO : 0; };
        const int syncs = count("shard_sync");
        check(!pool.sync_touched_shards(err) && count("shard_sync") == syncs + 1, "durable: the sync fails");
        fx.fault = nullptr;
        check(pool.sync_touched_shards(err) && count("shard_sync") == syncs + 2, "durable: the retry flushes the shard again (the failed one is not forgotten)");
        check(pool.sync_touched_shards(err) && count("shard_sync") == syncs + 2, "durable: and then there is nothing left");
        // a failed flush of a hit chunk fails the put and leaves the (valid) file alone
        fx.fault = [](const char* s) { return !std::strcmp(s, "chunk_flush") ? EIO : 0; };
        check(!pool.put(ChunkKind::kv_slab, raw.data(), raw.size(), 0, r2, info, err, true) && info.error_kind == SessionError::io && !info.hit,
              "durable: a failed flush of a hit chunk fails the put");
        fx.fault = nullptr;
        std::vector<uint8_t> back(raw.size());
        check(pool.get(ChunkKind::kv_slab, r, back.data(), back.size(), err) && back == raw, "durable: the chunk is intact");
        // two durable writers at once: the second does not return while the first still flushes the folders it took
        std::atomic<int> started{0};
        fx.fault = [&](const char* s) {
            if (!std::strcmp(s, "shard_sync")) { ++started; std::this_thread::sleep_for(std::chrono::milliseconds(150)); }
            return 0;
        };
        std::vector<uint8_t> raw_b = fill_vec(7001, 32);
        ChunkRef rb;
        check(pool.put(ChunkKind::kv_slab, raw_b.data(), raw_b.size(), 0, rb, info, err, true), "durable: writer B's chunk (its folder is queued)");
        std::thread ta([&] { std::string e2; pool.sync_touched_shards(e2); });   // writer A's sync takes B's folder too
        while (started.load() == 0) std::this_thread::yield();
        const auto t0 = std::chrono::steady_clock::now();
        std::string e3;
        check(pool.sync_touched_shards(e3), "durable: B's sync succeeds");
        const double waited = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        ta.join();
        fx.fault = nullptr;
        check(waited > 60, "durable: B's sync waited for A's flush to finish instead of returning on an emptied set");
    }
    {   // a look at an existing chunk that fails (resources, I/O) says nothing about the file: the put fails, the file stays
        Fx fx("look");
        ChunkPool& pool = *fx.pool;
        std::vector<uint8_t> raw = fill_vec(8000, 41);
        ChunkRef r;
        check(pool.put(ChunkKind::kv_slab, raw.data(), raw.size(), 0, r, info, err), "look: put");
        fx.fault = [](const char* s) { return !std::strcmp(s, "chunk_validate") ? EMFILE : 0; };
        ChunkRef r2;
        check(!pool.put(ChunkKind::kv_slab, raw.data(), raw.size(), 0, r2, info, err) && !info.hit && info.error_kind != SessionError::none,
              "look: a failed look fails the put");
        fx.fault = nullptr;
        check(fs::exists(pool.path_of(r.key)) && pool.refs_of(r.key) == 1, "look: the chunk file is not touched, its reference is unchanged");
        std::vector<uint8_t> back(raw.size());
        check(pool.get(ChunkKind::kv_slab, r, back.data(), back.size(), err) && back == raw, "look: and it still reads");
        check(pool.put(ChunkKind::kv_slab, raw.data(), raw.size(), 0, r2, info, err) && info.hit, "look: the next put hits it");
    }
    {   // the RAM preflight counts the manifest's largest chunk and three buffers per worker
        const uint64_t raw_total = 1ull << 30;
        const uint64_t big = 48ull << 20;
        const uint64_t need = chunked_read_peak_bytes(raw_total, 8, 512, big);
        check(need >= raw_total + 8 * 3 * big, "peak: three buffers of the largest chunk for each of 8 workers");
        check(chunked_read_peak_bytes(raw_total, 8, 512, 1 << 20) < need, "peak: a smaller chunk needs less");
        check(chunked_read_peak_bytes(raw_total, 1, 512, big) < need && chunked_read_peak_bytes(raw_total, 1, 64) > raw_total, "peak: per worker, and the guess without a manifest");
        // a read of an image whose chunks are large is asked for more than the guess from chunk_tokens (small tokens, big rows)
        Fx fx("peak");
        Spec sp;
        sp.n = 200; sp.cps = {};
        SavedConversation img = make_image(sp);
        W w;
        check(write_to(fx, fx.path("p.bin"), img, w, 64), "peak: written");
        uint64_t asked_max = 0;
        SessionReadLimits lim;
        lim.admit = [&](uint64_t n, std::string&) { asked_max = std::max(asked_max, n); return true; };
        SavedConversation back;
        std::string e;
        check(read_from(fx, fx.path("p.bin"), back, e, lim), "peak: read");
        uint64_t max_raw = 0;
        for (const auto& r : w.refs) max_raw = std::max<uint64_t>(max_raw, r.raw_len);
        const uint64_t rt = rd<uint64_t>(slurp(fx.path("p.bin")), 48);
        check(asked_max >= std::max(chunked_read_peak_bytes(rt, 1, 64, max_raw), chunked_read_peak_bytes(rt, 1, 64)),
              "peak: the read is asked for the manifest's largest chunk when that is more than the guess");
    }
    {   // shared: a chunk another engine wrote is adopted with the file's own age, so the first hit touches it
        Fx fx("adopt", true, 600);
        ChunkPoolOptions o;
        o.dir = fx.pool_dir.string();
        o.grace_s = 600;
        o.clock_s = [&fx] { return fx.clock; };
        ChunkPool b(o);   // engine B: opened before A wrote
        check(b.open(err), "adopt: B opens");
        std::vector<uint8_t> raw = fill_vec(4000, 51);
        ChunkRef r;
        check(fx.pool->put(ChunkKind::kv_slab, raw.data(), raw.size(), 0, r, info, err), "adopt: A writes");
        const int64_t written = fx.clock;
        fx.clock += 5000;   // hours later by the grace's measure
        b.ref(&r, 1);       // B finds it through a manifest
        ChunkRef again;
        check(b.put(ChunkKind::kv_slab, raw.data(), raw.size(), 0, again, info, err) && info.hit, "adopt: B hits it");
        struct stat st {};
        check(::stat(b.path_of(r.key).c_str(), &st) == 0 && (int64_t) st.st_mtime == fx.clock && (int64_t) st.st_mtime > written,
              "adopt: the hit touched the file (its age was the file's, not the moment of adoption)");
    }
    {   // the shared sweep re-checks the size on the disk: a file that is not the one that was indexed stays
        Fx fx("sweepsize", true, 600);
        ChunkPool& pool = *fx.pool;
        std::vector<uint8_t> raw = fill_vec(5000, 61);
        ChunkRef r;
        check(pool.put(ChunkKind::kv_slab, raw.data(), raw.size(), 0, r, info, err), "sweep size: put");
        pool.unref(&r, 1);
        fx.clock += 5000;
        std::vector<uint8_t> other(chunk_file_size(5000) + 2 * kChunkAlign, 7);   // another size under the name
        spit(pool.path_of(r.key), other);
        struct timespec ts[2] = {{0, UTIME_OMIT}, {(time_t) (fx.clock - 3000), 0}};
        ::utimensat(AT_FDCWD, pool.path_of(r.key).c_str(), ts, 0);
        std::set<ChunkKey> none;
        check(pool.sweep(true, false, &none) == 0 && fs::exists(pool.path_of(r.key)), "sweep size: an old unnamed file of another size is not unlinked");
    }
    {   // before_publish: a chunk gone before the manifest fails the write and publishes nothing
        Fx fx("publish");
        Spec sp;
        sp.n = 100;
        SavedConversation img = make_image(sp);
        ChunkedWriteOptions o;
        o.before_publish = [](const std::vector<ChunkRef>&, std::string& why) { why = "chunk gone"; return false; };
        W w;
        check(!write_to(fx, fx.path("bp.bin"), img, w, 64, o) && w.error.find("chunk gone") != std::string::npos && !fs::exists(fx.path("bp.bin")),
              "publish: a refusing before_publish fails the write and leaves no manifest");
        size_t seen = 0;
        o.before_publish = [&](const std::vector<ChunkRef>& refs, std::string&) { seen = refs.size(); return true; };
        W w2;
        check(write_to(fx, fx.path("bp.bin"), img, w2, 64, o) && seen == w2.refs.size() && seen > 0 && fs::exists(fx.path("bp.bin")),
              "publish: asked with every reference, and an accepting one publishes");
    }
    {   // the digest's tail (bytes after the last full 8 and 32): golden keys, and little-endian byte order
        const char* gp = std::getenv("STRATA_CHUNKS_GOLDEN_PRINT");
        struct G { size_t n; const char* hex; };
        const G golden[] = {
            {1, "537dd636a33bbd3c117409b501406729"}, {7, "99250e5e33ed5dae22dcb7afbf810509"},
            {8, "1b09f5a555835a2afecfed2fbc0c3689"}, {9, "21e516b4f647c8a8dce83bbb86b4c257"},
            {31, "d74fc5361be9a3541cf44e96376d2be5"}, {33, "647685ed21fdc96249878015b4ce4d79"},
            {39, "fcb140fd6c81c521b72711d4a75657fa"}, {1001, "12069432236b6f96c2386f69de634858"}};
        for (const auto& g : golden) {
            std::vector<uint8_t> d(g.n);
            for (size_t i = 0; i < g.n; ++i) d[i] = (uint8_t) (i * 31 + 7);
            if (gp) std::printf("golden tail %zu %s\n", g.n, chunk_key(d.data(), d.size()).hex().c_str());
            else check(chunk_key(d.data(), d.size()).hex() == g.hex, "digest128: golden key of a patterned buffer with a tail");
        }
        SessionHasher h(0);
        const uint8_t one[3] = {1, 2, 3};
        h.update(one, 3);
        const auto dg = h.digest128();
        uint64_t lo = 0;
        for (int i = 7; i >= 0; --i) lo = lo << 8 | dg[(size_t) i];
        check(lo == h.digest(), "digest128: the low half is digest() as a little-endian number");
    }
}

// the review round 2: the read's worker count, a decoder that cannot be made, a replaced chunk is not quarantined
void test_review_fixes2() {
    std::string err;
    ChunkPutInfo info;
    {   // the calling thread works beside the workers: one more set of buffers is live than there are workers
        const uint64_t raw_total = 1ull << 30, big = 48ull << 20;
        check(chunked_read_peak_bytes(raw_total, 2, 512, big) >= raw_total + 3 * 3 * big, "peak: workers + the calling thread hold buffers");
        check(chunked_read_peak_bytes(raw_total, 8, 512, big) >= raw_total + 9 * 3 * big, "peak: 8 workers + the caller");
        check(chunked_read_peak_bytes(raw_total, 1, 512, big) < raw_total + 2 * 3 * big, "peak: one worker is the caller alone");
        check(chunked_read_peak_bytes(raw_total, 2, 512, big) > chunked_read_peak_bytes(raw_total, 1, 512, big), "peak: two workers need more than one");
    }
    {   // a chunk that is replaced between the reader's stat and its quarantine is the writer's: it stays
        Fx fx("quar");
        ChunkPool& pool = *fx.pool;
        std::vector<uint8_t> raw = fill_vec(9000, 31);
        ChunkRef r;
        check(pool.put(ChunkKind::kv_slab, raw.data(), raw.size(), 0, r, info, err), "quarantine race: put");
        const auto good = slurp(pool.path_of(r.key));
        auto bad = good;
        bad[kChunkAlign + 10] ^= 0x40;
        spit(pool.path_of(r.key), bad);
        fx.fault = [&](const char* step) {
            if (std::strcmp(step, "chunk_quarantine") == 0) {   // a concurrent writer publishes a fresh file (a new inode)
                const std::string tmp = pool.path_of(r.key) + ".replacement";
                spit(tmp, good);
                fs::rename(tmp, pool.path_of(r.key));
            }
            return 0;
        };
        std::vector<uint8_t> back(raw.size());
        check(!pool.get(ChunkKind::kv_slab, r, back.data(), back.size(), err), "quarantine race: the read of the rotten file fails");
        fx.fault = nullptr;
        check(fs::exists(pool.path_of(r.key)) && count_prefixed(fx.pool_dir / r.key.hex().substr(0, 2), ".tmp-bad-") == 0,
              "quarantine race: the replacement is not moved aside");
        check(pool.get(ChunkKind::kv_slab, r, back.data(), back.size(), err) && back == raw, "quarantine race: and it reads");
    }
#ifdef STRATA_DISK_ZSTD
    {   // a decoder that cannot get its context is not a corrupt chunk: the file stays
        Fx fx("noctx");
        fx.codec = ChunkCodec::zstd;
        fx.make_pool(true);
        ChunkPool& pool = *fx.pool;
        std::vector<uint8_t> raw(200000);
        for (size_t i = 0; i < raw.size(); ++i) raw[i] = uint8_t((i / 64) % 7);   // compresses well: stored as zstd
        ChunkRef r;
        check(pool.put(ChunkKind::kv_slab, raw.data(), raw.size(), 0, r, info, err) && r.stored_len < r.raw_len, "no decoder context: a zstd chunk");
        std::vector<uint8_t> back(raw.size());
        SessionError kind = SessionError::none;
        codec_test_fail_decoder_context(true);
        const bool failed = !pool.get(ChunkKind::kv_slab, r, back.data(), back.size(), err, &kind);
        codec_test_fail_decoder_context(false);
        check(failed && kind == SessionError::memory, "no decoder context: the read fails as a memory failure");
        check(fs::exists(pool.path_of(r.key)) && count_prefixed(fx.pool_dir / r.key.hex().substr(0, 2), ".tmp-bad-") == 0,
              "no decoder context: the chunk is not quarantined");
        check(pool.get(ChunkKind::kv_slab, r, back.data(), back.size(), err) && back == raw, "no decoder context: it reads once the decoder works");
        bool res = false;
        std::vector<uint8_t> junk(64, 0xee);
        check(!segmented_decode(junk.data(), junk.size(), back.data(), back.size(), &res) && !res, "no decoder context: a bad stream is not a resource failure");
    }
#endif
}

int main() {
    struct Cleanup {
        ~Cleanup() {
            std::error_code ec;
            for (const auto& e : fs::directory_iterator(fs::temp_directory_path(), ec))
                if (e.path().filename().string().rfind(own_prefix(), 0) == 0) fs::remove_all(e.path(), ec);
        }
    } cleanup;
    test_hash_and_names();
    test_chunk_files(true);
    test_chunk_corruption();
    test_pool_lifecycle();
    test_failures();
    test_review_fixes();
    test_review_fixes2();
    test_concurrent_put();
    test_round_trip_geometry();
    test_padding_and_spare();
    test_source_equals_image();
    test_dedup();
    test_manifest_format();
    test_manifest_corruption();
    test_limits();
    test_prefix();
    test_write_failures();
    test_shuffle_and_skip();
    test_codec_units();
    test_codec_chunks();
    test_codec_suite();
    std::printf("conversation_chunks_test: %d checks passed\n", checks);
    return 0;
}
