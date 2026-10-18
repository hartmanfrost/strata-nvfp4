// Codecs of the chunked disk tier's chunks (conversation_chunks.hpp).  The codec id lives in each chunk's header, so the
// choice of codec is not a format decision: the reader decodes whatever the writer chose, and raw (no dependency) is
// always available.  zstd exists only in a build with STRATA_DISK_ZSTD (codec_available says); a chunk written with it
// is refused cleanly (a cache miss) by a build without.  Pure host code.
//
// A zstd chunk's payload is a SEGMENTED container, so the reader needs nothing but the chunk to decode it: the parts of
// a K/V slab have different element widths (and different compressibility), each segment is stored on its own terms:
//   segment = u32 raw_len | u32 stored_len | u8 mode | u8 width | u16 0 | stored bytes
//   mode 0 raw (stored_len == raw_len, width 0) | 1 zstd frame | 2 zstd frame of the byte-shuffled segment (width 2/4/8)
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <vector>

namespace strata::core {

enum class ChunkCodec : uint8_t { raw = 0, lz4 = 1 /* reserved: never written */, zstd = 2 };

// Whether this build can encode and decode `codec` (raw: always).
bool codec_available(ChunkCodec codec);

// Byte-plane transposition of `n` bytes of `width`-byte elements (plain loops): plane b holds byte b of every element,
// a tail shorter than an element stays as it is.  Width 0 or 1 copies.  `dst` and `src` do not overlap.
void shuffle_bytes(const uint8_t* src, uint8_t* dst, size_t n, uint8_t width);
void unshuffle_bytes(const uint8_t* src, uint8_t* dst, size_t n, uint8_t width);

// Encodes `n` raw bytes as ONE segment.  True with `stored_raw` set: the bytes are to be stored as they are (the codec is
// raw or not available in this build, or the result was not smaller); `out` is then untouched.  True with `stored_raw`
// clear: `out` holds the encoded bytes (a zstd frame of the shuffled bytes when `shuffle_width` > 1), fewer than `n`.
// False: the encoder failed.
bool codec_encode(ChunkCodec codec, int level, uint8_t shuffle_width, const void* raw, size_t n,
                  std::vector<uint8_t>& out, bool& stored_raw);

// Decodes into `raw`, which holds exactly `raw_len` bytes: the result must be exactly that long (a stream that
// decodes to less or more is refused, so a lying length cannot make the decoder allocate or write past it).
bool codec_decode(ChunkCodec codec, bool shuffled, uint8_t shuffle_width, const void* stored, size_t stored_len,
                  void* raw, size_t raw_len);

// A running compression ratio per kind of data: a kind that does not pay (ratio over kThreshold) is stored raw without
// trying, except every kProbeEvery-th segment of it (the data may change); a segment under kMinSegment is never tried.
// Thread-safe.  Affects what is stored, never a chunk's key.
class AdaptiveSkip {
public:
    static constexpr size_t kMinSegment = 16u << 10;
    static constexpr double kThreshold = 0.93;
    static constexpr int kProbeEvery = 8;
    static constexpr int kKinds = 8;
    bool attempt(uint8_t kind, size_t raw_len);              // should this segment be tried?
    void record(uint8_t kind, size_t raw_len, size_t stored_len);   // the outcome of an attempt
    double ratio(uint8_t kind) const;                        // the running ratio (0 = nothing seen yet)
private:
    mutable std::mutex mu_;
    std::array<double, kKinds> ratio_{};
    std::array<bool, kKinds> seen_{};
    std::array<uint32_t, kKinds> skipped_{};
};

struct CodecSegment {
    uint32_t len = 0;
    uint8_t width = 0;      // shuffle element width (0 / 1 = none)
    uint8_t kind = 0;       // the AdaptiveSkip class
};

// Encodes `n` raw bytes cut into `segs` (their lengths add up to n) as the container above.  `any_shuffled` tells
// whether some segment is mode 2.  The container may come out larger than `n`: the caller then stores raw.
bool segmented_encode(int level, const CodecSegment* segs, size_t nsegs, const void* raw, size_t n, AdaptiveSkip* skip,
                      std::vector<uint8_t>& out, bool& any_shuffled);
// Decodes a container into `raw` of exactly `raw_len` bytes; every length is checked against what is left on both sides.
// `resource`, when given, tells a failure that is the decoder's own (no zstd context, no memory) from a bad stream.
bool segmented_decode(const void* stored, size_t stored_len, void* raw, size_t raw_len, bool* resource = nullptr);
#ifdef STRATA_DISK_ZSTD
// Test hook: while on, the calling thread's zstd decoder context cannot be made.
void codec_test_fail_decoder_context(bool on);
#endif

} // namespace strata::core
