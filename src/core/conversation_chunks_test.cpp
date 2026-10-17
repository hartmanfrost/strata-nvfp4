// CPU-only tests for the chunked disk format (include/strata/core/conversation_chunks.hpp): chunk objects, the pool,
// slab geometry, the manifest and its reader's limits.  Built with -DSTRATA_BUILD_CONVERSATION_TESTS=ON; no CUDA.
// Run it twice for the two I/O modes: as is (O_DIRECT where the filesystem takes it) and STRATA_SESSION_BUFFERED=1.
#include "strata/core/conversation_chunks.hpp"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <mutex>
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

uint8_t hb(uint64_t seed, uint64_t i) {
    uint64_t x = (i + 1) * 0x9E3779B97F4A7C15ull ^ (seed * 0xD6E8FEB86659FD93ull);
    x ^= x >> 32; x *= 0xD6E8FEB86659FD93ull; x ^= x >> 32;
    return (uint8_t) x;
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
    {
        SavedConversation out = sentinel;
        size_t mb = 0;
        uint64_t rb = 0;
        std::string e;
        ChunkedReadOptions o;
        o.upto = 10;
        check(!chunked_read(*fx.pool, p, kId, out, mb, rb, e, {}, o) && !e.empty(), "prefix reads are refused until they exist");
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
    test_concurrent_put();
    test_round_trip_geometry();
    test_padding_and_spare();
    test_source_equals_image();
    test_dedup();
    test_manifest_format();
    test_manifest_corruption();
    test_limits();
    test_write_failures();
    std::printf("conversation_chunks_test: %d checks passed\n", checks);
    return 0;
}
