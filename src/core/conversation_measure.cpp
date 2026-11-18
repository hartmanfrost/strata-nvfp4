// conversation_measure: offline measurement of a parked conversation image, to size the chunked disk format (format v2,
// conversation_chunks.hpp) on real data.  CPU only, no engine, no GPU; nothing is written next to the input (the
// cross-check and the selftest use a scratch folder).
//
//   conversation_measure measure FILE [--pool DIR] [--chunk-tokens N] [--levels 1,3] [--no-compress] [--per-layer]
//                                     [--vs FILE2 [--vs-pool DIR2]] [--cross-check] [--tmp DIR]
//   conversation_measure gen OUT.bin --tokens N [--seed S] [--low-entropy]
//   conversation_measure selftest [--tmp DIR]
//
// FILE is a v1 session file (what a /slots save or the disk tier's v1 writes: magic "STRSESS\x01") or a v2 manifest
// ("STRSESS\x02") together with its chunk pool (--pool DIR: the strata-pool-<ident> folder beside it).  The model and
// configuration fingerprints are taken from the file's own header (this tool measures; it does not authenticate).
// The pool folder is opened like the tier opens it, which removes ".tmp-*" files older than 10 minutes: run it on a copy
// of a live folder if that matters.
//
// The report has
//   - the sections of the image: K, V, k_scale, v_scale, pooled (indexer rows), GDN state, PLE, tails, dead, block_pos,
//     token ids, image keys: raw bytes, the share of the image and the count of parts; --per-layer adds the K/V layers.
//     Sections that live in chunks (the K/V parts and GDN) are compressed; the others are inline in a v2 manifest.
//   - compression ratios (stored / raw, smaller is better) of every chunked section for zstd-1 and zstd-3 (--levels), each
//     plain and with the production byte shuffle ("+shuf": k_scale/v_scale width 2, pooled width 4, GDN width 4), encoded
//     the way the tier encodes: one segment per part of a slab (all layers of a part together), 4 MiB slices of GDN
//     state.  A segment that does not shrink counts raw, as in the tier.  Every encode is decoded and compared (a '!'
//     after a ratio: that failed).  Encode and decode time and throughput per config; hashing time of the chunks.
//   - with --vs FILE2: the dedup of FILE2 against FILE by chunk key (the SessionHasher::digest128 keys the pool uses): the
//     slabs and GDN slices of FILE2 whose key FILE already has, in raw bytes and as a share, the repeats inside FILE2, and
//     the raw bytes FILE2 would still write after FILE.  --cross-check does it for real as well (chunked_write of FILE
//     then FILE2 into scratch pools, raw codec) and fails when the estimate and the pool's own hit counts differ.
// The chunk size is --chunk-tokens (default 512, a multiple of 4 in 64..4096), the tier's flag of the same name.
// The image is parsed into RAM: expect about the file's size in memory (and the cross-check writes both into --tmp).
// zstd columns need a build with STRATA_DISK_ZSTD (otherwise only raw is reported).
//
// Build (CMake: -DSTRATA_BUILD_CONVERSATION_TESTS=ON, target conversation_measure; ctest runs its selftest; by hand):
//   g++ -std=c++20 -O2 -DSTRATA_DISK_ZSTD=1 -Iinclude -pthread src/core/conversation_measure.cpp src/core/conversation_chunks.cpp
//       src/core/conversation_codec.cpp src/core/conversation_file.cpp -lzstd -o conversation_measure
// Try it without real data:
//   conversation_measure gen a.bin --tokens 640 --low-entropy; conversation_measure gen b.bin --tokens 1000 --low-entropy
//   conversation_measure measure a.bin --vs b.bin --chunk-tokens 64 --cross-check
// `selftest` generates synthetic v1 files, checks the report against sizes computed from the geometry by hand, checks
// the dedup estimate against the real pool and a v2 round trip.
#include "strata/core/conversation_chunks.hpp"
#include "strata/core/conversation_file.hpp"

#include <unistd.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

using namespace strata::core;
namespace fs = std::filesystem;

namespace {

using Clock = std::chrono::steady_clock;
double ms_since(Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); }

constexpr const char* kPartNames[5] = {"K", "V", "k_scale", "v_scale", "pooled"};
constexpr uint8_t kWidths[5] = {0, 0, 2, 2, 4};     // production shuffle widths (LayerGeom::widths of conversation_chunks.cpp)
constexpr uint8_t kGdnWidth = 4;
constexpr uint8_t kGdnKind = 5;

// ---------------------------------------------------------------------------------------------------------------- identity

bool read_head(const std::string& path, uint8_t (&h)[32], std::string& err) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { err = "cannot open " + path; return false; }
    f.read(reinterpret_cast<char*>(h), 32);
    if (f.gcount() != 32) { err = path + ": too short to be a session file"; return false; }
    return true;
}
// 1 or 2 (the version byte of the magic), 0 when it is neither
int format_of(const uint8_t (&h)[32]) {
    if (std::memcmp(h, "STRSESS", 7) != 0) return 0;
    return h[7] == 1 ? 1 : h[7] == 2 ? 2 : 0;
}
SessionFileIdentity identity_of(const uint8_t (&h)[32]) {
    SessionFileIdentity id;
    std::memcpy(&id.model, h + 16, 8);
    std::memcpy(&id.config, h + 24, 8);
    return id;
}

// ---------------------------------------------------------------------------------------------------------------- loading

struct Loaded {
    SavedConversation img;
    SessionFileIdentity id;
    int format = 0;
    uint64_t file_bytes = 0;        // v1: the file; v2: the manifest
    double parse_ms = 0;
};

bool load(const std::string& path, const std::string& pool_dir, Loaded& out, std::string& err) {
    uint8_t h[32];
    if (!read_head(path, h, err)) return false;
    out.format = format_of(h);
    if (!out.format) { err = path + ": not a Strata session file or manifest (magic)"; return false; }
    out.id = identity_of(h);
    const auto t = Clock::now();
    size_t bytes = 0;
    if (out.format == 1) {
        if (!session_file_read(path, out.id, out.img, bytes, err)) return false;
    } else {
        if (pool_dir.empty()) { err = path + " is a v2 manifest: --pool DIR (its strata-pool-* folder) is required"; return false; }
        if (!fs::is_directory(pool_dir)) { err = "pool folder " + pool_dir + " does not exist"; return false; }
        ChunkPoolOptions o;
        o.dir = pool_dir;
        o.direct = false;
        ChunkPool pool(o);
        if (!pool.open(err)) return false;
        uint64_t raw = 0;
        if (!chunked_read(pool, path, out.id, out.img, bytes, raw, err, SessionReadLimits{})) return false;
    }
    out.file_bytes = bytes;
    out.parse_ms = ms_since(t);
    return true;
}

// ---------------------------------------------------------------------------------------------------------------- sections

struct Section {
    uint64_t raw = 0;
    uint64_t parts = 0;
};
struct Sections {
    std::map<std::string, Section> by_name;
    std::vector<std::string> order;
    void add(const std::string& name, uint64_t raw) {
        if (!by_name.count(name)) order.push_back(name);
        auto& s = by_name[name];
        s.raw += raw;
        ++s.parts;
    }
    uint64_t total() const { uint64_t n = 0; for (const auto& kv : by_name) n += kv.second.raw; return n; }
};

std::vector<const ConversationCheckpoint*> cps_of(const SavedConversation& s) {
    std::vector<const ConversationCheckpoint*> v{&s.live};
    for (const auto& c : s.checkpoints) v.push_back(&c);
    return v;
}

Sections sections_of(const SavedConversation& s) {
    Sections out;
    for (const char* n : kPartNames) { out.order.push_back(n); out.by_name[n] = {}; }
    for (const auto& kv : s.kv) {
        const ConversationBuffer* b[5] = {&kv.k, &kv.v, &kv.k_scale, &kv.v_scale, &kv.pooled};
        for (int p = 0; p < 5; ++p) out.add(kPartNames[p], b[p]->size());
    }
    for (const auto* c : cps_of(s)) {
        out.add("GDN", c->gdn.size());
        out.add("PLE", c->ple.size());
        out.add("tails", c->tails.size());
        out.add("dead", c->dead.size());
        out.add("block_pos", c->block_pos.size());
        out.add("ids", c->ids.size() * 4);
        out.add("image keys", c->imgs.size() * 16);
    }
    return out;
}

// ---------------------------------------------------------------------------------------------------------------- slabs

// The slab cut of conversation_chunks.cpp (Plan): slab i holds cells [i*T, (i+1)*T) of every layer and pooled rows
// [i*T/4, (i+1)*T/4) of the rows that are not the spare one (the last), part-major.
struct Layout {
    uint32_t T = 0;
    uint64_t n_slabs = 0;
    std::vector<std::array<uint64_t, 4>> unit;   // bytes per cell, per layer and part
    std::vector<uint64_t> row, data_rows, cells;
};

bool make_layout(const SavedConversation& s, uint32_t T, Layout& L, std::string& err) {
    if (T < 64 || T > 4096 || T % 4) { err = "--chunk-tokens must be a multiple of 4 between 64 and 4096"; return false; }
    L = Layout{};
    L.T = T;
    uint64_t max_cells = 0;
    for (const auto& kv : s.kv) {
        const ConversationBuffer* b[4] = {&kv.k, &kv.v, &kv.k_scale, &kv.v_scale};
        std::array<uint64_t, 4> u{};
        const uint64_t cells = kv.cells < 0 ? 0 : (uint64_t) kv.cells;
        for (int p = 0; p < 4; ++p) {
            if (cells == 0) { if (b[p]->size()) { err = "K/V part has bytes but no cells"; return false; } continue; }
            if (b[p]->size() % cells) { err = "K/V part is not a whole number of cells"; return false; }
            u[p] = b[p]->size() / cells;
        }
        const uint64_t rows = kv.pooled_rows < 0 ? 0 : (uint64_t) kv.pooled_rows;
        uint64_t rb = 0, dr = 0;
        if (rows) {
            if (kv.pooled.size() % rows) { err = "pooled part is not a whole number of rows"; return false; }
            rb = kv.pooled.size() / rows;
            dr = rows - 1;
        }
        L.unit.push_back(u); L.row.push_back(rb); L.data_rows.push_back(dr); L.cells.push_back(cells);
        max_cells = std::max(max_cells, cells);
    }
    L.n_slabs = (max_cells + T - 1) / T;
    return true;
}

void slice_of(const Layout& L, size_t l, size_t part, uint64_t i, uint64_t& off, uint64_t& len) {
    if (part < 4) {
        const uint64_t c0 = std::min<uint64_t>(i * L.T, L.cells[l]), c1 = std::min<uint64_t>((i + 1) * L.T, L.cells[l]);
        off = c0 * L.unit[l][part];
        len = (c1 - c0) * L.unit[l][part];
    } else {
        const uint64_t r0 = std::min<uint64_t>(i * (L.T / 4), L.data_rows[l]), r1 = std::min<uint64_t>((i + 1) * (L.T / 4), L.data_rows[l]);
        off = r0 * L.row[l];
        len = (r1 - r0) * L.row[l];
    }
}

// One chunk's raw bytes with the way the tier segments them for the codec.
struct Piece {
    ChunkKey key;
    uint64_t raw = 0;
    std::vector<CodecSegment> segs;     // lengths add up to raw
    std::vector<uint8_t> bytes;
};

// Walks every chunk of the image in manifest order (slabs, then the GDN slices of the live state and of each checkpoint),
// building the bytes the way chunked_write builds them; `fn(Piece&)` returns false to stop.
template<class Fn>
bool for_each_chunk(const SavedConversation& s, const Layout& L, double* hash_ms, Fn&& fn, std::string& err) {
    Piece pc;
    for (uint64_t i = 0; i < L.n_slabs; ++i) {
        pc.bytes.clear();
        pc.segs.clear();
        for (size_t part = 0; part < 5; ++part)
            for (size_t l = 0; l < s.kv.size(); ++l) {
                uint64_t off, n;
                slice_of(L, l, part, i, off, n);
                if (!n) continue;
                const auto& kv = s.kv[l];
                const ConversationBuffer* b[5] = {&kv.k, &kv.v, &kv.k_scale, &kv.v_scale, &kv.pooled};
                const size_t at = pc.bytes.size();
                pc.bytes.resize(at + (size_t) n);
                if (!b[part]->read(pc.bytes.data() + at, (size_t) off, (size_t) n)) { err = "internal: K/V read"; return false; }
                if (!pc.segs.empty() && pc.segs.back().kind == part && pc.segs.back().len + n <= UINT32_MAX) pc.segs.back().len += (uint32_t) n;
                else pc.segs.push_back({(uint32_t) n, kWidths[part], (uint8_t) part});
            }
        pc.raw = pc.bytes.size();
        const auto t = Clock::now();
        pc.key = chunk_key(pc.bytes.data(), pc.bytes.size());
        if (hash_ms) *hash_ms += ms_since(t);
        if (!fn(pc)) return true;
    }
    for (const auto* c : cps_of(s)) {
        for (size_t off = 0; off < c->gdn.size(); off += kStateSlice) {
            const size_t n = std::min<size_t>(kStateSlice, c->gdn.size() - off);
            pc.segs.assign(1, CodecSegment{(uint32_t) n, kGdnWidth, kGdnKind});
            pc.bytes.assign(c->gdn.begin() + (std::ptrdiff_t) off, c->gdn.begin() + (std::ptrdiff_t) (off + n));
            pc.raw = n;
            const auto t = Clock::now();
            pc.key = chunk_key(pc.bytes.data(), pc.bytes.size());
            if (hash_ms) *hash_ms += ms_since(t);
            if (!fn(pc)) return true;
        }
    }
    return true;
}

// ---------------------------------------------------------------------------------------------------------------- compression

struct Config {
    std::string name;
    int level = 0;          // 0 = raw
    bool shuffle = false;
};
struct Cell {               // result of one config on one section
    uint64_t raw = 0, stored = 0;
    double enc_ms = 0, dec_ms = 0;
    bool bad = false;       // an encode that failed or a round trip that did not give the bytes back
};

std::string section_of_seg(const CodecSegment& g) { return g.kind == kGdnKind ? "GDN" : kPartNames[g.kind]; }

bool compress_all(const SavedConversation& s, const Layout& L, const std::vector<Config>& cfgs,
                  std::map<std::string, std::vector<Cell>>& out, std::string& err) {
    std::vector<uint8_t> enc, back;
    auto fn = [&](Piece& pc) {
        size_t at = 0;
        for (const auto& g : pc.segs) {
            auto& row = out[section_of_seg(g)];
            if (row.empty()) row.resize(cfgs.size());
            for (size_t c = 0; c < cfgs.size(); ++c) {
                Cell& cell = row[c];
                cell.raw += g.len;
                if (cfgs[c].level == 0) { cell.stored += g.len; continue; }
                const uint8_t w = cfgs[c].shuffle ? g.width : 0;
                bool stored_raw = false;
                auto t = Clock::now();
                const bool ok = codec_encode(ChunkCodec::zstd, cfgs[c].level, w, pc.bytes.data() + at, g.len, enc, stored_raw);
                cell.enc_ms += ms_since(t);
                if (!ok) { cell.bad = true; cell.stored += g.len; continue; }
                if (stored_raw) { cell.stored += g.len; continue; }
                cell.stored += enc.size();
                back.resize(g.len);
                t = Clock::now();
                const bool dok = codec_decode(ChunkCodec::zstd, w > 1, w, enc.data(), enc.size(), back.data(), g.len);
                cell.dec_ms += ms_since(t);
                if (!dok || std::memcmp(back.data(), pc.bytes.data() + at, g.len) != 0) cell.bad = true;
            }
            at += g.len;
        }
        return true;
    };
    return for_each_chunk(s, L, nullptr, fn, err);
}

// ---------------------------------------------------------------------------------------------------------------- dedup

struct KeySet {
    std::map<ChunkKey, uint64_t> raw;        // key -> raw bytes
    uint64_t total_raw = 0, chunks = 0;
    double hash_ms = 0;
};
bool keys_of(const SavedConversation& s, uint32_t T, KeySet& ks, std::string& err) {
    Layout L;
    if (!make_layout(s, T, L, err)) return false;
    auto fn = [&](Piece& pc) {
        ks.raw.emplace(pc.key, pc.raw);
        ks.total_raw += pc.raw;
        ++ks.chunks;
        return true;
    };
    return for_each_chunk(s, L, &ks.hash_ms, fn, err);
}

struct Dedup {
    uint64_t b_chunks = 0, b_raw = 0, shared_chunks = 0, shared_raw = 0;   // of B: what A already has
    uint64_t self_chunks = 0, self_raw = 0;                                 // repeats inside B (after the first)
};
bool dedup_of(const SavedConversation& a, const SavedConversation& b, uint32_t T, Dedup& d, std::string& err) {
    KeySet ka;
    if (!keys_of(a, T, ka, err)) return false;
    Layout L;
    if (!make_layout(b, T, L, err)) return false;
    std::set<ChunkKey> seen;
    auto fn = [&](Piece& pc) {
        ++d.b_chunks; d.b_raw += pc.raw;
        if (ka.raw.count(pc.key)) { ++d.shared_chunks; d.shared_raw += pc.raw; }
        else if (!seen.insert(pc.key).second) { ++d.self_chunks; d.self_raw += pc.raw; }
        return true;
    };
    return for_each_chunk(b, L, nullptr, fn, err);
}

// The same for real: chunked_write of A then B into a scratch pool; B's stats as the pool counted them.
bool real_dedup(const SavedConversation& a, const SavedConversation& b, const SessionFileIdentity& id, uint32_t T,
                const std::string& tmp, ChunkedWriteStats& sb, std::string& err) {
    std::error_code ec;
    const fs::path dir = fs::path(tmp) / ("measure-xcheck-" + std::to_string((uint64_t) ::getpid()));
    fs::create_directories(dir, ec);
    struct Cleanup { fs::path p; ~Cleanup() { std::error_code e; fs::remove_all(p, e); } } cleanup{dir};
    ChunkPoolOptions o;
    o.dir = (dir / "pool").string();
    o.direct = false;
    ChunkPool pool(o);
    if (!pool.open(err)) return false;
    ChunkedWriteOptions w;
    w.chunk_tokens = T;
    std::vector<ChunkRef> refs;
    ChunkedWriteStats sa;
    if (!chunked_write(pool, (dir / "a.manifest").string(), a, nullptr, id, w, refs, sa, err)) return false;
    refs.clear();
    return chunked_write(pool, (dir / "b.manifest").string(), b, nullptr, id, w, refs, sb, err);
}

// ---------------------------------------------------------------------------------------------------------------- report

std::string human(uint64_t n) {
    char buf[40];
    if (n >= (1ull << 30)) std::snprintf(buf, sizeof buf, "%.2f GiB", (double) n / (1ull << 30));
    else if (n >= (1ull << 20)) std::snprintf(buf, sizeof buf, "%.2f MiB", (double) n / (1ull << 20));
    else if (n >= (1ull << 10)) std::snprintf(buf, sizeof buf, "%.2f KiB", (double) n / (1ull << 10));
    else std::snprintf(buf, sizeof buf, "%" PRIu64 " B", n);
    return buf;
}

struct Args {
    std::string cmd, file, pool, vs, vs_pool, tmp;
    uint32_t chunk_tokens = 512;
    std::vector<int> levels{1, 3};
    bool compress = true, per_layer = false, cross = false, low_entropy = false;
    int64_t tokens = 0;
    uint64_t seed = 1;
};

int do_measure(const Args& a, std::FILE* out, std::string& err, Loaded* keep_a = nullptr) {
    Loaded A;
    if (!load(a.file, a.pool, A, err)) return 1;
    Layout L;
    if (!make_layout(A.img, a.chunk_tokens, L, err)) return 1;
    const Sections sec = sections_of(A.img);
    const uint64_t total = sec.total();
    std::fprintf(out, "file        %s\nformat      v%d  (%s)\nidentity    model %016" PRIx64 " config %016" PRIx64 "\n",
                 a.file.c_str(), A.format, A.format == 1 ? "single file" : "manifest + chunk pool", A.id.model, A.id.config);
    std::fprintf(out, "file bytes  %" PRIu64 " (%s)%s\n", A.file_bytes, human(A.file_bytes).c_str(),
                 A.format == 2 ? "  [manifest only; the chunks are in the pool]" : "");
    std::fprintf(out, "tokens      live %zu, checkpoints %zu, K/V layers %zu (%zu with pooled rows), chunk %u tokens -> %" PRIu64 " slabs\n",
                 A.img.live.ids.size(), A.img.checkpoints.size(), A.img.kv.size(),
                 (size_t) std::count_if(A.img.kv.begin(), A.img.kv.end(), [](const ConversationKv& k) { return k.pooled_rows > 0; }),
                 a.chunk_tokens, L.n_slabs);
    std::fprintf(out, "parse       %.1f ms\n\n", A.parse_ms);

    std::vector<Config> cfgs{{"raw", 0, false}};
    const bool zstd = codec_available(ChunkCodec::zstd);
    if (a.compress && zstd)
        for (int lv : a.levels) {
            cfgs.push_back({"zstd-" + std::to_string(lv), lv, false});
            cfgs.push_back({"zstd-" + std::to_string(lv) + "+shuf", lv, true});
        }
    if (a.compress && !zstd) std::fprintf(out, "(zstd is not available in this build: only raw is reported)\n\n");
    std::map<std::string, std::vector<Cell>> comp;
    if (!compress_all(A.img, L, cfgs, comp, err)) return 1;
    KeySet ks;
    if (!keys_of(A.img, a.chunk_tokens, ks, err)) return 1;

    std::fprintf(out, "%-12s %14s %7s %6s", "section", "raw bytes", "share", "parts");
    for (const auto& c : cfgs) if (c.level) std::fprintf(out, " %14s", c.name.c_str());
    std::fprintf(out, "\n");
    for (const auto& name : sec.order) {
        const Section& s = sec.by_name.at(name);
        if (!s.parts && !s.raw) continue;
        std::fprintf(out, "%-12s %14" PRIu64 " %6.2f%% %6" PRIu64, name.c_str(), s.raw, total ? 100.0 * (double) s.raw / (double) total : 0.0, s.parts);
        auto it = comp.find(name);
        for (size_t c = 0; c < cfgs.size(); ++c) {
            if (!cfgs[c].level) continue;
            if (it == comp.end() || !it->second[c].raw) std::fprintf(out, " %14s", "inline");
            else std::fprintf(out, " %13.3f%s", (double) it->second[c].stored / (double) it->second[c].raw, it->second[c].bad ? "!" : " ");
        }
        std::fprintf(out, "\n");
    }
    std::fprintf(out, "%-12s %14" PRIu64 "  (%s)\n\n", "total", total, human(total).c_str());

    uint64_t chunked_raw = 0;
    for (const auto& kv : comp) chunked_raw += kv.second[0].raw;
    std::fprintf(out, "chunked bytes (K/V slabs + GDN slices): %" PRIu64 " (%s) in %" PRIu64 " chunks\n", chunked_raw, human(chunked_raw).c_str(), ks.chunks);
    if (cfgs.size() > 1) {
        std::fprintf(out, "%-16s %14s %8s %12s %12s %10s %10s\n", "codec", "stored bytes", "ratio", "encode ms", "decode ms", "enc MB/s", "dec MB/s");
        for (size_t c = 0; c < cfgs.size(); ++c) {
            uint64_t stored = 0; double e = 0, d = 0;
            for (const auto& kv : comp) { stored += kv.second[c].stored; e += kv.second[c].enc_ms; d += kv.second[c].dec_ms; }
            const double mb = (double) chunked_raw / 1e6;
            std::fprintf(out, "%-16s %14" PRIu64 " %8.3f", cfgs[c].name.c_str(), stored, chunked_raw ? (double) stored / (double) chunked_raw : 0.0);
            if (cfgs[c].level) std::fprintf(out, " %12.1f %12.1f %10.0f %10.0f\n", e, d, e > 0 ? mb / (e / 1e3) : 0.0, d > 0 ? mb / (d / 1e3) : 0.0);
            else std::fprintf(out, " %12s %12s %10s %10s\n", "-", "-", "-", "-");
        }
    }
    std::fprintf(out, "hashing     %.1f ms for %" PRIu64 " chunks (%.0f MB/s)\n", ks.hash_ms, ks.chunks, ks.hash_ms > 0 ? (double) ks.total_raw / 1e6 / (ks.hash_ms / 1e3) : 0.0);

    if (a.per_layer) {
        std::fprintf(out, "\nper layer (raw bytes)\n%-6s %8s %12s %12s %12s %12s %12s\n", "layer", "cells", "K", "V", "k_scale", "v_scale", "pooled");
        for (size_t l = 0; l < A.img.kv.size(); ++l) {
            const auto& k = A.img.kv[l];
            std::fprintf(out, "%-6zu %8" PRId64 " %12zu %12zu %12zu %12zu %12zu\n", l, k.cells, k.k.size(), k.v.size(), k.k_scale.size(), k.v_scale.size(), k.pooled.size());
        }
    }

    if (!a.vs.empty()) {
        Loaded B;
        if (!load(a.vs, a.vs_pool.empty() ? a.pool : a.vs_pool, B, err)) return 1;
        Dedup d;
        if (!dedup_of(A.img, B.img, a.chunk_tokens, d, err)) return 1;
        const uint64_t to_write = d.b_raw - d.shared_raw - d.self_raw;
        std::fprintf(out, "\ndedup of %s against %s (chunk %u tokens)\n", a.vs.c_str(), a.file.c_str(), a.chunk_tokens);
        std::fprintf(out, "  chunks in the second image   %" PRIu64 "  (%" PRIu64 " raw bytes, %s)\n", d.b_chunks, d.b_raw, human(d.b_raw).c_str());
        std::fprintf(out, "  already in the first         %" PRIu64 " chunks, %" PRIu64 " raw bytes (%.1f%%)\n", d.shared_chunks, d.shared_raw, d.b_raw ? 100.0 * (double) d.shared_raw / (double) d.b_raw : 0.0);
        std::fprintf(out, "  repeated inside the second   %" PRIu64 " chunks, %" PRIu64 " raw bytes\n", d.self_chunks, d.self_raw);
        std::fprintf(out, "  still to write (raw)         %" PRIu64 " bytes (%s)\n", to_write, human(to_write).c_str());
        if (a.cross) {
            const std::string tmp = a.tmp.empty() ? fs::temp_directory_path().string() : a.tmp;
            ChunkedWriteStats sb;
            if (!real_dedup(A.img, B.img, A.id, a.chunk_tokens, tmp, sb, err)) return 1;
            std::fprintf(out, "  cross-check (real pool, raw) %u hits / %u misses, hit raw bytes %" PRIu64 "\n", sb.hits, sb.misses, sb.hit_raw_bytes);
            if (sb.hit_raw_bytes != d.shared_raw + d.self_raw || sb.hits != d.shared_chunks + d.self_chunks) {
                err = "cross-check FAILED: the estimate and the pool disagree";
                return 1;
            }
            std::fprintf(out, "  cross-check agrees with the estimate\n");
        }
    }
    if (keep_a) *keep_a = std::move(A);
    return 0;
}

// ---------------------------------------------------------------------------------------------------------------- generator

// A small image of the real structure (conversation_chunks_test.cpp's make_image): every byte is a function of (seed,
// index), so an image of more tokens with the same seed continues the shorter one.
bool g_low = false;
uint8_t hb(uint64_t seed, uint64_t i) {
    uint64_t x = (i + 1) * 0x9E3779B97F4A7C15ull ^ (seed * 0xD6E8FEB86659FD93ull);
    x ^= x >> 32; x *= 0xD6E8FEB86659FD93ull; x ^= x >> 32;
    return g_low ? (uint8_t) (x & 7) : (uint8_t) x;
}
ConversationBuffer fill_buf(size_t n, uint64_t seed) {
    ConversationBuffer b;
    b.resize(n);
    b.visit(0, n, [&](uint8_t* p, size_t c, size_t at) { for (size_t i = 0; i < c; ++i) p[i] = hb(seed, at + i); return true; });
    return b;
}
std::vector<uint8_t> fill_vec(size_t n, uint64_t seed) {
    std::vector<uint8_t> v(n);
    for (size_t i = 0; i < n; ++i) v[i] = hb(seed, i);
    return v;
}
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
constexpr size_t kGenLiveGdn = (4u << 20) + 4097, kGenCpGdn = 1000;
// n tokens (a multiple of 4), checkpoints at n/4 and n/2
SavedConversation make_image(int64_t n, uint64_t seed, bool low) {
    g_low = low;
    SavedConversation s;
    for (size_t i = 0; i < s.geometry.size(); ++i) s.geometry[i] = int64_t(100 + i);
    s.layer_lo = 0; s.layer_hi = 48;
    s.cvec = false;
    s.live = make_cp(n, kGenLiveGdn, seed * 77);
    for (int64_t t : {n / 4, n / 2}) s.checkpoints.push_back(make_cp(t, kGenCpGdn, seed * 77));
    for (int l = 0; l < 3; ++l) {
        const bool draft = l == 2;
        const size_t* u = draft ? kDraftUnit : kMainUnit;
        ConversationKv kv;
        kv.format = draft ? 1 : 2; kv.cells = n; kv.heads = 2; kv.head_dim = 256; kv.page_size = 4; kv.idx_dim = 8;
        kv.pooled_rows = draft ? 0 : n / 4 + 1;
        const uint64_t ls = seed * 1000 + (uint64_t) l * 10;
        kv.k = fill_buf((size_t) n * u[0], ls + 0);
        kv.v = fill_buf((size_t) n * u[1], ls + 1);
        kv.k_scale = fill_buf((size_t) n * u[2], ls + 2);
        kv.v_scale = fill_buf((size_t) n * u[3], ls + 3);
        kv.pooled = fill_buf((size_t) kv.pooled_rows * kRowBytes, ls + 4);
        s.kv.push_back(std::move(kv));
    }
    g_low = false;
    return s;
}
constexpr SessionFileIdentity kGenId{0x1111222233334444ull, 0x5555666677778888ull};

bool gen_file(const std::string& path, int64_t n, uint64_t seed, bool low, std::string& err) {
    const SavedConversation img = make_image(n, seed, low);
    size_t bytes = 0;
    return session_file_write(path, img, kGenId, bytes, err);
}

// ---------------------------------------------------------------------------------------------------------------- selftest

int g_checks = 0;
#define CHECK(cond, msg) do { ++g_checks; if (!(cond)) { std::fprintf(stderr, "FAIL: %s (line %d)\n", msg, __LINE__); return 1; } } while (0)

std::string slurp(std::FILE* f) {
    std::string s;
    std::fseek(f, 0, SEEK_SET);
    char buf[4096];
    size_t n;
    while ((n = std::fread(buf, 1, sizeof buf, f)) > 0) s.append(buf, n);
    return s;
}
bool has(const std::string& s, const std::string& needle) { return s.find(needle) != std::string::npos; }

int selftest(const Args& a) {
    const fs::path base = fs::path(a.tmp.empty() ? fs::temp_directory_path().string() : a.tmp) / ("measure-selftest-" + std::to_string((uint64_t) ::getpid()));
    std::error_code ec;
    fs::remove_all(base, ec);
    fs::create_directories(base);
    struct Cleanup { fs::path p; ~Cleanup() { std::error_code e; fs::remove_all(p, e); } } cleanup{base};
    std::string err;

    const int64_t nA = 640, nB = 1000;   // 640 = ten whole slabs of 64 cells
    CHECK(gen_file((base / "a.bin").string(), nA, 1, true, err), err.c_str());
    CHECK(gen_file((base / "b.bin").string(), nB, 1, true, err), err.c_str());
    CHECK(gen_file((base / "r.bin").string(), nA, 9, false, err), err.c_str());

    Args m;
    m.file = (base / "a.bin").string();
    m.vs = (base / "b.bin").string();
    m.chunk_tokens = 64;
    m.cross = true;
    m.tmp = base.string();
    m.per_layer = true;
    std::FILE* f = std::tmpfile();
    CHECK(f, "tmpfile");
    Loaded keep;
    CHECK(do_measure(m, f, err, &keep) == 0, err.c_str());
    const std::string rep = slurp(f);
    std::fclose(f);

    // sizes from the geometry by hand: three layers (two main with pooled rows, one draft), cells = n
    const uint64_t K = nA * 8 * 3, V = nA * (4 * 2 + 8), KS = nA * 2 * 3, VS = nA * 2;
    const uint64_t PO = 2 * (nA / 4 + 1) * kRowBytes;
    const Sections sec = sections_of(keep.img);
    CHECK(sec.by_name.at("K").raw == K, "K bytes");
    CHECK(sec.by_name.at("V").raw == V, "V bytes");
    CHECK(sec.by_name.at("k_scale").raw == KS, "k_scale bytes");
    CHECK(sec.by_name.at("v_scale").raw == VS, "v_scale bytes");
    CHECK(sec.by_name.at("pooled").raw == PO, "pooled bytes");
    CHECK(sec.by_name.at("GDN").raw == kGenLiveGdn + 2 * kGenCpGdn, "GDN bytes");
    CHECK(sec.by_name.at("ids").raw == 4 * (uint64_t) (nA + nA / 4 + nA / 2), "ids bytes");
    CHECK(sec.by_name.at("GDN").parts == 3 && sec.by_name.at("K").parts == 3, "part counts");
    CHECK(has(rep, "format      v1") && has(rep, "-> 10 slabs"), "report header");
    CHECK(!codec_available(ChunkCodec::zstd) || (has(rep, "zstd-1+shuf") && has(rep, "zstd-3")), "zstd columns");
    CHECK(has(rep, "cross-check agrees"), "dedup estimate agrees with the real pool");

    // dedup by hand: B (1000 tokens) continues A: A's ten slabs are the first ten of B's sixteen, nothing else is shared
    KeySet ka, kb;
    CHECK(keys_of(keep.img, 64, ka, err), err.c_str());
    Loaded B;
    CHECK(load(m.vs, "", B, err), err.c_str());
    CHECK(keys_of(B.img, 64, kb, err), err.c_str());
    size_t shared = 0;
    for (const auto& kv : kb.raw) if (ka.raw.count(kv.first)) ++shared;
    CHECK(shared == 10, "ten chunks shared");
    CHECK(ka.chunks == 10 + 2 + 2 && kb.chunks == 16 + 2 + 2, "chunk counts (slabs + live 2 slices + 2 checkpoints)");
    Dedup d;
    CHECK(dedup_of(keep.img, B.img, 64, d, err), err.c_str());
    CHECK(d.shared_chunks == 10 && d.self_chunks == 0 && d.b_chunks == 20, "dedup counts");
    Layout L;
    CHECK(make_layout(keep.img, 64, L, err), err.c_str());
    CHECK(d.shared_raw == K + V + KS + VS + PO - 2 * kRowBytes, "the shared bytes are all of A's slabs");

    // compression: 3 bits of entropy compress, full entropy does not, raw is exactly 1, round trips hold
    {
        std::map<std::string, std::vector<Cell>> comp;
        const std::vector<Config> cfgs{{"raw", 0, false}, {"z1", 1, false}, {"z1s", 1, true}};
        CHECK(compress_all(keep.img, L, cfgs, comp, err), err.c_str());
        for (const char* n : {"K", "V", "k_scale", "v_scale", "pooled", "GDN"}) CHECK(comp.count(n), n);
        uint64_t chunked = 0;
        for (const auto& kv : comp) {
            CHECK(kv.second[0].stored == kv.second[0].raw, "raw is 1.0");
            chunked += kv.second[0].raw;
            CHECK(!kv.second[1].bad && !kv.second[2].bad, "round trips");
            CHECK(kv.second[1].stored <= kv.second[1].raw, "never above raw");
        }
        CHECK(chunked == K + V + KS + VS + PO - 2 * kRowBytes + sec.by_name.at("GDN").raw, "chunked bytes = sections minus the two spare rows");
        if (codec_available(ChunkCodec::zstd)) {
            CHECK(comp["K"][1].stored * 10 < comp["K"][1].raw * 7, "low-entropy K compresses under 0.7");
            Loaded R;
            CHECK(load((base / "r.bin").string(), "", R, err), err.c_str());
            std::map<std::string, std::vector<Cell>> rc;
            Layout RL;
            CHECK(make_layout(R.img, 64, RL, err) && compress_all(R.img, RL, cfgs, rc, err), err.c_str());
            CHECK(rc["K"][1].stored * 100 >= rc["K"][1].raw * 99, "random K does not compress");
        }
    }

    // v2: the same image through a pool, measured from its manifest, has the same sections
    {
        ChunkPoolOptions o;
        o.dir = (base / "strata-pool-test").string();
        o.direct = false;
        {
            ChunkPool pool(o);
            CHECK(pool.open(err), err.c_str());
            ChunkedWriteOptions w;
            w.chunk_tokens = 64;
            std::vector<ChunkRef> refs;
            ChunkedWriteStats st;
            CHECK(chunked_write(pool, (base / "a.v2").string(), keep.img, nullptr, kGenId, w, refs, st, err), err.c_str());
        }
        Args m2;
        m2.file = (base / "a.v2").string();
        m2.pool = o.dir;
        m2.chunk_tokens = 64;
        m2.compress = false;
        std::FILE* g = std::tmpfile();
        Loaded v2;
        CHECK(do_measure(m2, g, err, &v2) == 0, err.c_str());
        const std::string r2 = slurp(g);
        std::fclose(g);
        CHECK(has(r2, "format      v2") && has(r2, "manifest only"), "v2 detected");
        const Sections s2 = sections_of(v2.img);
        for (const auto& name : sec.order) CHECK(s2.by_name.at(name).raw == sec.by_name.at(name).raw, name.c_str());
        Args m3 = m2;
        m3.pool.clear();
        std::FILE* h = std::tmpfile();
        std::string e3;
        CHECK(do_measure(m3, h, e3) != 0 && has(e3, "--pool"), "v2 without a pool is refused");
        std::fclose(h);
    }
    {
        std::ofstream(base / "junk.bin") << "this is not a session file at all, but it is long enough to read a header";
        Args j = m; j.file = (base / "junk.bin").string(); j.vs.clear();
        std::FILE* g = std::tmpfile();
        std::string e;
        CHECK(do_measure(j, g, e) != 0 && has(e, "magic"), "junk refused");
        std::fclose(g);
    }
    std::printf("conversation_measure selftest: %d checks passed (zstd %s)\n", g_checks, codec_available(ChunkCodec::zstd) ? "on" : "off");
    return 0;
}

int usage() {
    std::fprintf(stderr,
        "usage: conversation_measure measure FILE [--pool DIR] [--chunk-tokens N] [--levels 1,3] [--no-compress] [--per-layer]\n"
        "                                         [--vs FILE2 [--vs-pool DIR2]] [--cross-check] [--tmp DIR]\n"
        "       conversation_measure gen OUT.bin --tokens N [--seed S] [--low-entropy]\n"
        "       conversation_measure selftest [--tmp DIR]\n");
    return 2;
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 2) return usage();
    Args a;
    a.cmd = argv[1];
    int i = 2;
    if ((a.cmd == "measure" || a.cmd == "gen") && argc >= 3) a.file = argv[i++];
    for (; i < argc; ++i) {
        const std::string k = argv[i];
        auto val = [&]() -> const char* { return i + 1 < argc ? argv[++i] : nullptr; };
        const char* v = nullptr;
        if (k == "--pool" && (v = val())) a.pool = v;
        else if (k == "--vs" && (v = val())) a.vs = v;
        else if (k == "--vs-pool" && (v = val())) a.vs_pool = v;
        else if (k == "--tmp" && (v = val())) a.tmp = v;
        else if (k == "--chunk-tokens" && (v = val())) a.chunk_tokens = (uint32_t) std::strtoul(v, nullptr, 10);
        else if (k == "--tokens" && (v = val())) a.tokens = std::strtoll(v, nullptr, 10);
        else if (k == "--seed" && (v = val())) a.seed = std::strtoull(v, nullptr, 10);
        else if (k == "--levels" && (v = val())) {
            a.levels.clear();
            for (char* p = const_cast<char*>(v); *p;) { a.levels.push_back((int) std::strtol(p, &p, 10)); if (*p == ',') ++p; else break; }
            if (a.levels.empty()) return usage();
            for (int lv : a.levels) if (lv < 1 || lv > 19) return usage();
        }
        else if (k == "--no-compress") a.compress = false;
        else if (k == "--per-layer") a.per_layer = true;
        else if (k == "--cross-check") a.cross = true;
        else if (k == "--low-entropy") a.low_entropy = true;
        else return usage();
    }
    std::string err;
    if (a.cmd == "selftest") return selftest(a);
    if (a.cmd == "gen") {
        if (a.file.empty() || a.tokens < 4) return usage();
        if (!gen_file(a.file, a.tokens / 4 * 4, a.seed, a.low_entropy, err)) { std::fprintf(stderr, "gen: %s\n", err.c_str()); return 1; }
        std::printf("wrote %s (%" PRId64 " tokens)\n", a.file.c_str(), a.tokens / 4 * 4);
        return 0;
    }
    if (a.cmd == "measure") {
        if (a.file.empty()) return usage();
        const int rc = do_measure(a, stdout, err);
        if (rc) std::fprintf(stderr, "conversation_measure: %s\n", err.c_str());
        return rc;
    }
    return usage();
}
