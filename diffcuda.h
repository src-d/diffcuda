#ifndef CUDIFF_CUDIFF_H
#define CUDIFF_CUDIFF_H

#include <cstdint>
#include <vector>
#include <tuple>

namespace diffcuda {

using HASH = uint32_t;

using Lines = std::tuple<std::vector<uint32_t>&&, std::vector<HASH>>;

struct Deletion {
  uint8_t *dest;
  uint32_t size;
};

struct Insertion {
  uint8_t *dest;
  uint8_t *source;
  uint32_t size;
};

using Script = std::tuple<std::vector<Deletion> &&, std::vector<Insertion> &&>;

Lines preprocess(const uint8_t *data, size_t size) noexcept;

std::vector<Script> diff(
    const uint8_t **old, const size_t *old_size, const uint8_t **now,
    const size_t *now_size, uint32_t pairs_number, int device) noexcept;

}

#endif //CUDIFF_CUDIFF_H
