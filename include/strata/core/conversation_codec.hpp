// Codecs of the chunked disk tier's chunks (conversation_chunks.hpp).  The codec id lives in each chunk's header, so the
// choice of codec is not a format decision: the reader decodes whatever the writer chose, and raw (no dependency) is
// always available.  Pure host code.
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace strata::core {

enum class ChunkCodec : uint8_t { raw = 0, lz4 = 1 /* reserved: never written */, zstd = 2 };

// Whether this build can encode and decode `codec` (raw: always).
bool codec_available(ChunkCodec codec);

// Encodes `n` raw bytes.  True with `stored_raw` set: the bytes are to be stored as they are (the codec is raw or not
// available in this build, or the result was not worth it); `out` is then untouched.  True with `stored_raw` clear:
// `out` holds the encoded bytes, fewer than `n`.  False: the encoder failed.  `shuffle_width` (0 = none) is the element
// width of the byte-plane transposition applied before the codec.
bool codec_encode(ChunkCodec codec, int level, uint8_t shuffle_width, const void* raw, size_t n,
                  std::vector<uint8_t>& out, bool& stored_raw);

// Decodes into `raw`, which holds exactly `raw_len` bytes: the result must be exactly that long (a stream that
// decodes to less or more is refused, so a lying length cannot make the decoder allocate or write past it).
bool codec_decode(ChunkCodec codec, bool shuffled, uint8_t shuffle_width, const void* stored, size_t stored_len,
                  void* raw, size_t raw_len);

} // namespace strata::core
