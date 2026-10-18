#include "strata/core/conversation_codec.hpp"

#include <cstring>
#include <memory>
#include <new>

#ifdef STRATA_DISK_ZSTD
#include <zstd.h>
#endif

namespace strata::core {

namespace {
template <class T> void put_le(uint8_t* p, T v) { std::memcpy(p, &v, sizeof v); }
template <class T> T get_le(const uint8_t* p) { T v; std::memcpy(&v, p, sizeof v); return v; }
constexpr size_t kSegHeader = 12;

template <size_t W> void shuffle_w(const uint8_t* src, uint8_t* dst, size_t elems) {
    for (size_t b = 0; b < W; ++b) {
        uint8_t* plane = dst + b * elems;
        for (size_t i = 0; i < elems; ++i) plane[i] = src[i * W + b];
    }
}
template <size_t W> void unshuffle_w(const uint8_t* src, uint8_t* dst, size_t elems) {
    for (size_t b = 0; b < W; ++b) {
        const uint8_t* plane = src + b * elems;
        for (size_t i = 0; i < elems; ++i) dst[i * W + b] = plane[i];
    }
}
} // namespace

bool codec_available(ChunkCodec codec) {
#ifdef STRATA_DISK_ZSTD
    return codec == ChunkCodec::raw || codec == ChunkCodec::zstd;
#else
    return codec == ChunkCodec::raw;
#endif
}

void shuffle_bytes(const uint8_t* src, uint8_t* dst, size_t n, uint8_t width) {
    if (width <= 1 || n < width) { if (n) std::memmove(dst, src, n); return; }
    const size_t elems = n / width, body = elems * width;
    switch (width) {
        case 2: shuffle_w<2>(src, dst, elems); break;
        case 4: shuffle_w<4>(src, dst, elems); break;
        case 8: shuffle_w<8>(src, dst, elems); break;
        default:
            for (size_t b = 0; b < width; ++b)
                for (size_t i = 0; i < elems; ++i) dst[b * elems + i] = src[i * width + b];
    }
    if (n > body) std::memcpy(dst + body, src + body, n - body);
}

void unshuffle_bytes(const uint8_t* src, uint8_t* dst, size_t n, uint8_t width) {
    if (width <= 1 || n < width) { if (n) std::memmove(dst, src, n); return; }
    const size_t elems = n / width, body = elems * width;
    switch (width) {
        case 2: unshuffle_w<2>(src, dst, elems); break;
        case 4: unshuffle_w<4>(src, dst, elems); break;
        case 8: unshuffle_w<8>(src, dst, elems); break;
        default:
            for (size_t b = 0; b < width; ++b)
                for (size_t i = 0; i < elems; ++i) dst[i * width + b] = src[b * elems + i];
    }
    if (n > body) std::memcpy(dst + body, src + body, n - body);
}

#ifdef STRATA_DISK_ZSTD
namespace {
struct CctxDel { void operator()(ZSTD_CCtx* c) const { ZSTD_freeCCtx(c); } };
struct DctxDel { void operator()(ZSTD_DCtx* c) const { ZSTD_freeDCtx(c); } };
ZSTD_CCtx* cctx() { thread_local std::unique_ptr<ZSTD_CCtx, CctxDel> c(ZSTD_createCCtx()); return c.get(); }
thread_local bool g_fail_dctx = false;   // test hook: the decoder context cannot be made
ZSTD_DCtx* dctx() {
    if (g_fail_dctx) return nullptr;
    thread_local std::unique_ptr<ZSTD_DCtx, DctxDel> c(ZSTD_createDCtx());
    return c.get();
}

// one zstd frame of src (shuffled first when width > 1) appended to out; false: the encoder failed
bool zstd_frame(int level, uint8_t width, const uint8_t* src, size_t n, std::vector<uint8_t>& out) {
    ZSTD_CCtx* c = cctx();
    if (!c) return false;
    std::vector<uint8_t> shuf;
    if (width > 1) { shuf.resize(n); shuffle_bytes(src, shuf.data(), n, width); src = shuf.data(); }
    const size_t at = out.size();
    out.resize(at + ZSTD_compressBound(n));
    const size_t got = ZSTD_compressCCtx(c, out.data() + at, out.size() - at, src, n, level);
    if (ZSTD_isError(got)) { out.resize(at); return false; }
    out.resize(at + got);
    return true;
}
// decodes exactly n bytes into dst (unshuffling when width > 1); `resource` is set when the failure is the decoder's
// own (no context, no memory), not the stream's
bool zstd_unframe(uint8_t width, const uint8_t* stored, size_t stored_len, uint8_t* dst, size_t n, bool* resource = nullptr) {
    ZSTD_DCtx* c = dctx();
    if (!c) { if (resource) *resource = true; return false; }
    if (width > 1) {
        std::vector<uint8_t> tmp;
        try { tmp.resize(n); } catch (const std::bad_alloc&) { if (resource) *resource = true; return false; }
        const size_t got = ZSTD_decompressDCtx(c, tmp.data(), n, stored, stored_len);
        if (ZSTD_isError(got) || got != n) return false;
        unshuffle_bytes(tmp.data(), dst, n, width);
        return true;
    }
    const size_t got = ZSTD_decompressDCtx(c, dst, n, stored, stored_len);
    return !ZSTD_isError(got) && got == n;
}
} // namespace
#endif

bool codec_encode(ChunkCodec codec, int level, uint8_t shuffle_width, const void* raw, size_t n,
                  std::vector<uint8_t>& out, bool& stored_raw) {
    stored_raw = true;
#ifdef STRATA_DISK_ZSTD
    if (codec == ChunkCodec::zstd && n) {
        std::vector<uint8_t> enc;
        if (!zstd_frame(level, shuffle_width, static_cast<const uint8_t*>(raw), n, enc)) return false;
        if (enc.size() < n) { out = std::move(enc); stored_raw = false; }
    }
#else
    (void) codec; (void) level; (void) shuffle_width; (void) raw; (void) n; (void) out;
#endif
    return true;
}

#ifdef STRATA_DISK_ZSTD
void codec_test_fail_decoder_context(bool on) { g_fail_dctx = on; }
#endif

bool codec_decode(ChunkCodec codec, bool shuffled, uint8_t shuffle_width, const void* stored, size_t stored_len,
                  void* raw, size_t raw_len) {
    if (codec == ChunkCodec::raw) {
        if (shuffled || stored_len != raw_len) return false;
        if (raw_len) std::memcpy(raw, stored, raw_len);
        return true;
    }
#ifdef STRATA_DISK_ZSTD
    if (codec == ChunkCodec::zstd) {
        if (shuffled && shuffle_width <= 1) return false;
        return zstd_unframe(shuffled ? shuffle_width : 0, static_cast<const uint8_t*>(stored), stored_len,
                            static_cast<uint8_t*>(raw), raw_len);
    }
#else
    (void) shuffled; (void) shuffle_width; (void) stored; (void) stored_len; (void) raw; (void) raw_len;
#endif
    return false;
}

// ------------------------------------------------------------------------------------------------ adaptive skip

bool AdaptiveSkip::attempt(uint8_t kind, size_t raw_len) {
    if (raw_len < kMinSegment) return false;
    kind %= kKinds;
    std::lock_guard<std::mutex> g(mu_);
    if (!seen_[kind] || ratio_[kind] <= kThreshold) return true;
    return ++skipped_[kind] % kProbeEvery == 0;
}

void AdaptiveSkip::record(uint8_t kind, size_t raw_len, size_t stored_len) {
    if (!raw_len) return;
    kind %= kKinds;
    const double r = (double) stored_len / (double) raw_len;
    std::lock_guard<std::mutex> g(mu_);
    ratio_[kind] = seen_[kind] ? 0.75 * ratio_[kind] + 0.25 * r : r;
    seen_[kind] = true;
}

double AdaptiveSkip::ratio(uint8_t kind) const {
    std::lock_guard<std::mutex> g(mu_);
    return ratio_[kind % kKinds];
}

// ------------------------------------------------------------------------------------------------ container

bool segmented_encode(int level, const CodecSegment* segs, size_t nsegs, const void* raw, size_t n, AdaptiveSkip* skip,
                      std::vector<uint8_t>& out, bool& any_shuffled) {
    out.clear();
    any_shuffled = false;
    size_t total = 0;
    for (size_t i = 0; i < nsegs; ++i) total += segs[i].len;
    if (total != n) return false;
    const auto* src = static_cast<const uint8_t*>(raw);
    size_t at = 0;
    for (size_t i = 0; i < nsegs; ++i) {
        const CodecSegment& s = segs[i];
        if (!s.len) continue;
        const size_t head = out.size();
        out.resize(head + kSegHeader);
        uint8_t mode = 0, width = 0;
        bool packed = false;
#ifdef STRATA_DISK_ZSTD
        if (!skip || skip->attempt(s.kind, s.len)) {
            const size_t body = out.size();
            if (!zstd_frame(level, s.width, src + at, s.len, out)) return false;
            const size_t enc = out.size() - body;
            if (skip) skip->record(s.kind, s.len, enc);
            if ((double) enc <= AdaptiveSkip::kThreshold * (double) s.len) {
                packed = true;
                mode = s.width > 1 ? 2 : 1;
                width = s.width > 1 ? s.width : 0;
            } else out.resize(body);
        }
#else
        (void) level; (void) skip;
#endif
        if (!packed) out.insert(out.end(), src + at, src + at + s.len);
        put_le<uint32_t>(out.data() + head, s.len);
        put_le<uint32_t>(out.data() + head + 4, (uint32_t) (out.size() - head - kSegHeader));
        out[head + 8] = mode;
        out[head + 9] = width;
        out[head + 10] = out[head + 11] = 0;
        if (mode == 2) any_shuffled = true;
        at += s.len;
    }
    return true;
}

bool segmented_decode(const void* stored, size_t stored_len, void* raw, size_t raw_len, bool* resource) {
    if (resource) *resource = false;
    const auto* in = static_cast<const uint8_t*>(stored);
    auto* dst = static_cast<uint8_t*>(raw);
    size_t pos = 0, done = 0;
    while (pos < stored_len) {
        if (stored_len - pos < kSegHeader) return false;
        const uint32_t rl = get_le<uint32_t>(in + pos), sl = get_le<uint32_t>(in + pos + 4);
        const uint8_t mode = in[pos + 8], width = in[pos + 9];
        if (in[pos + 10] || in[pos + 11]) return false;
        pos += kSegHeader;
        if (rl == 0 || rl > raw_len - done || sl > stored_len - pos) return false;
        switch (mode) {
            case 0:
                if (sl != rl || width != 0) return false;
                std::memcpy(dst + done, in + pos, rl);
                break;
            case 1:
            case 2:
#ifdef STRATA_DISK_ZSTD
                if (mode == 1 ? width != 0 : (width != 2 && width != 4 && width != 8)) return false;
                if (!zstd_unframe(width, in + pos, sl, dst + done, rl, resource)) return false;
                break;
#else
                return false;   // no zstd in this build: a clean refusal (the cache misses)
#endif
            default: return false;
        }
        pos += sl;
        done += rl;
    }
    return done == raw_len;
}

} // namespace strata::core
