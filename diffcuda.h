#ifndef CUDIFF_CUDIFF_H
#define CUDIFF_CUDIFF_H

#include <cstdint>
#include <vector>
#include <tuple>

namespace diffcuda {

using Lines = std::tuple<std::vector<uint32_t>&&, std::vector<uint64_t>>;

struct Insertion {
  uint32_t dest;
  uint32_t source;
};

using Script = std::tuple<std::vector<uint32_t> &&, std::vector<Insertion> &&>;

Lines preprocess(const uint8_t *data, size_t size) noexcept;

Script diff(const uint8_t *old, size_t old_size, const uint8_t *now,
            size_t now_size) noexcept;

}

#endif //CUDIFF_CUDIFF_H
