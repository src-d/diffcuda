#ifndef CUDIFF_CUDIFF_H
#define CUDIFF_CUDIFF_H

#include <cstdint>
#include <memory>
#include <tuple>
#include <vector>

namespace diffcuda {

using hash_t = uint64_t;

using Lines = std::tuple<std::shared_ptr<std::vector<uint32_t>>,
                         std::shared_ptr<std::vector<hash_t>>>;

struct Deletion {
  const uint8_t *data;
  uint32_t size;
  uint32_t line;
};

struct Insertion {
  const uint8_t *data;
  const uint8_t *where;
  uint32_t size;
  uint32_t line_to;
  uint32_t line_from;
};

using Script = std::tuple<std::shared_ptr<std::vector<Deletion>>,
                          std::shared_ptr<std::vector<Insertion>>>;

Lines preprocess(const uint8_t *data, size_t size) noexcept;

std::vector<Script> diff(
    const uint8_t **old, const size_t *old_size, const uint8_t **now,
    const size_t *now_size, uint32_t pairs_number, int device = 0) noexcept;

}

#endif //CUDIFF_CUDIFF_H
