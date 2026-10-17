#include "strata/core/conversation_codec.hpp"

#include <cstring>

namespace strata::core {

bool codec_available(ChunkCodec codec) { return codec == ChunkCodec::raw; }

bool codec_encode(ChunkCodec, int, uint8_t, const void*, size_t, std::vector<uint8_t>&, bool& stored_raw) {
    // only the raw codec exists in this build: every codec asked for is stored as it is
    stored_raw = true;
    return true;
}

bool codec_decode(ChunkCodec codec, bool shuffled, uint8_t, const void* stored, size_t stored_len, void* raw,
                  size_t raw_len) {
    if (codec != ChunkCodec::raw || shuffled || stored_len != raw_len) return false;
    if (raw_len) std::memcpy(raw, stored, raw_len);
    return true;
}

} // namespace strata::core
