#ifdef NDEBUG
// we disable disabling assertions
#undef NDEBUG
#endif

#include "diffcuda.h"

#include <immintrin.h>

#include <cassert>
#include <cstring>
#include <algorithm>

#include <cuda_runtime_api.h>

#include "private.h"
#include "xxhash.h"

namespace diffcuda {

constexpr int MEAN_LINES = 30;

using unique_devptr_parent = std::unique_ptr<void, std::function<void(void*)>>;

class unique_devptr : public unique_devptr_parent {
 public:
  explicit unique_devptr(void *ptr) : unique_devptr_parent(
      ptr, [](void *p){ cudaFree(p); }) {}
};

Lines preprocess(const uint8_t *data, size_t size) noexcept {
  assert(data);
  assert(size > 0);
  std::shared_ptr<std::vector<uint32_t>> lines(new std::vector<uint32_t>);
  lines->reserve(size / MEAN_LINES);
  lines->push_back(0);
  std::shared_ptr<std::vector<hash_t>> hashes(new std::vector<hash_t>);
  hashes->reserve(size / MEAN_LINES);
  uint32_t ppos = 0;
  const __m256i newline = _mm256_set1_epi8('\n');
  uint32_t j = 0;

  #define ADD_LINE \
    hashes->push_back(XXH64(data + ppos, pos - ppos, 0)); \
    lines->push_back(pos); \
    ppos = pos

  for (; reinterpret_cast<intptr_t>(data + j) & 0x1F; j++) {
    if (data[j] == '\n') {
      auto pos = j + 1;
      ADD_LINE;
    }
  }

  for (; j < size - 31; j += 32) {
    __m256i buffer = _mm256_load_si256(
        reinterpret_cast<const __m256i *>(data + j));
    buffer = _mm256_cmpeq_epi8(buffer, newline);
    uint32_t mask = _mm256_movemask_epi8(buffer);
    if (mask != 0) {
      int left = __builtin_ctz(mask);
      int right = 32 - __builtin_clz(mask);
      if (right - left < 8) {
        auto pos = j + left + 1;
        ADD_LINE;
        for (int k = left + 1; k < right; k++) {
          if (mask & (1 << k)) {
            pos = j + k + 1;
            ADD_LINE;
          }
        }
      } else {
        uint32_t lm = mask & 0xFFFF;
        uint32_t rm = mask >> 16;
        for (int k = __builtin_ctz(lm); k < 32 - __builtin_clz(lm); k++) {
          if (lm & (1 << k)) {
            auto pos = j + k + 1;
            ADD_LINE;
          }
        }
        for (int k = __builtin_ctz(rm); k < 32 - __builtin_clz(rm); k++) {
          if (rm & (1 << k)) {
            auto pos = j + k + 16 + 1;
            ADD_LINE;
          }
        }
      }
    }
  }
  for (; j < size; j++) {
    if (data[j] == '\n') {
      auto pos = j + 1;
      ADD_LINE;
    }
  }
  if (lines->empty() || lines->back() != size) {
    lines->push_back(size);
  }
  #undef ADD_LINE
  return Lines(lines, hashes);
}

constexpr size_t memo_size = doffset(MAXD + 1) * sizeof(uint32_t);

std::vector<Script> diff(
    const uint8_t **old, const size_t *old_size, const uint8_t **now,
    const size_t *now_size, uint32_t pairs_number, int device) noexcept {
  assert(old);
  assert(old_size);
  assert(now);
  assert(now_size);
  assert(pairs_number > 0);
  std::vector<Script> scripts;
  CUCH(cudaSetDevice(device), scripts);
  std::vector<Lines> old_lines(pairs_number), now_lines(pairs_number);
  #pragma omp parallel for schedule(guided)
  for (uint32_t i = 0; i < pairs_number; i++) {
    old_lines[i] = preprocess(old[i], old_size[i]);
    now_lines[i] = preprocess(now[i], now_size[i]);
  }
  const hash_t **old_cuda, **now_cuda;
  CUMALLOC(old_cuda, pairs_number * sizeof(uint32_t*), scripts);
  CUMALLOC(now_cuda, pairs_number * sizeof(uint32_t*), scripts);
  unique_devptr old_cuda_sentinel(old_cuda);
  unique_devptr now_cuda_sentinel(now_cuda);
  uint32_t *old_size_cuda, *now_size_cuda;
  CUMALLOC(old_size_cuda, pairs_number * sizeof(uint32_t), scripts);
  CUMALLOC(now_size_cuda, pairs_number * sizeof(uint32_t), scripts);
  unique_devptr old_size_cuda_sentinel(old_size_cuda);
  unique_devptr now_size_cuda_sentinel(now_size_cuda);
  std::vector<unique_devptr> old_cuda_ptrs, now_cuda_ptrs;
  for (uint32_t i = 0; i < pairs_number; i++) {
    uint32_t *old_cuda_i, *now_cuda_i;
    size_t size = std::get<1>(old_lines[i])->size() * sizeof(hash_t);
    CUMALLOC(old_cuda_i, size, scripts);
    CUMEMCPY_ASYNC(old_cuda_i, std::get<1>(old_lines[i])->data(), size,
                   cudaMemcpyHostToDevice, scripts);
    CUMEMCPY_ASYNC(old_cuda + i, &old_cuda_i, sizeof(uint32_t*),
                   cudaMemcpyHostToDevice, scripts);
    size /= sizeof(hash_t);
    CUMEMCPY(old_size_cuda + i, &size, sizeof(uint32_t),
             cudaMemcpyHostToDevice, scripts);
    old_cuda_ptrs.emplace_back(old_cuda_i);
    size = std::get<1>(now_lines[i])->size() * sizeof(hash_t);
    CUMALLOC(now_cuda_i, size, scripts);
    CUMEMCPY_ASYNC(now_cuda_i, std::get<1>(now_lines[i])->data(), size,
                   cudaMemcpyHostToDevice, scripts);
    CUMEMCPY_ASYNC(now_cuda + i, &now_cuda_i, sizeof(uint32_t*),
                   cudaMemcpyHostToDevice, scripts);
    size /= sizeof(hash_t);
    CUMEMCPY(now_size_cuda + i, &size, sizeof(uint32_t),
             cudaMemcpyHostToDevice, scripts);
    now_cuda_ptrs.emplace_back(now_cuda_i);
  }
  uint32_t *workspace_cuda;
  const size_t workspace_size =
      (memo_size + 2 * MAXD + 1) * pairs_number * sizeof(uint32_t);
  CUMALLOC(workspace_cuda, workspace_size, scripts);
  CUCH(cudaMemsetAsync(workspace_cuda, 0, workspace_size), scripts);
  unique_devptr workspace_sentinel(workspace_cuda);
  uint32_t *deletions_cuda;
  CUMALLOC(deletions_cuda, MAXD * sizeof(uint32_t) * pairs_number, scripts);
  unique_devptr deletions_sentinel(deletions_cuda);
  uint32_t *insertions_cuda;
  CUMALLOC(insertions_cuda, 2 * MAXD * sizeof(uint32_t) * pairs_number, scripts);
  unique_devptr insertions_sentinel(insertions_cuda);

  bool status = myers_diff(
      device, pairs_number, memo_size, old_cuda, old_size_cuda, now_cuda,
      now_size_cuda, workspace_cuda, deletions_cuda, insertions_cuda);
  if (!status) {
    PANIC("myers_diff", scripts);
  }

  std::unique_ptr<uint32_t[]> deletions(new uint32_t[MAXD * pairs_number]);
  CUMEMCPY_ASYNC(deletions.get(), deletions_cuda,
           MAXD * sizeof(uint32_t) * pairs_number,
           cudaMemcpyDeviceToHost, scripts);
  std::unique_ptr<uint32_t[]> insertions(new uint32_t[2 * MAXD * pairs_number]);
  CUMEMCPY(insertions.get(), insertions_cuda,
           2 * MAXD * sizeof(uint32_t) * pairs_number,
           cudaMemcpyDeviceToHost, scripts);
  scripts.resize(pairs_number);
  #pragma omp parallel for schedule(guided)
  for (uint32_t i = 0; i < pairs_number; i++) {
    std::shared_ptr<std::vector<Deletion>> dels(new std::vector<Deletion>);
    std::shared_ptr<std::vector<Insertion>> ins(new std::vector<Insertion>);
    size_t offset = i;
    offset *= MAXD;
    for (; deletions[offset] != UINT32_MAX; offset++) {
      auto di = deletions[offset];
      auto& off = *std::get<0>(old_lines[i]);
      auto ptr = old[i] + off[di];
      auto size = off[di + 1] - off[di];
      dels->push_back(Deletion{data: ptr, size: size, line: di + 1});
    }
    offset = i;
    offset *= 2 * MAXD;
    for (; insertions[offset] != UINT32_MAX; offset += 2) {
      auto oldi = insertions[offset];
      auto nowi = insertions[offset + 1];
      auto& oldoff = *std::get<0>(old_lines[i]);
      auto oldptr = old[i] + oldoff[oldi];
      auto& nowoff = *std::get<0>(now_lines[i]);
      auto nowptr = now[i] + nowoff[nowi];
      auto nowsize = nowoff[nowi + 1] - nowoff[nowi];
      ins->push_back(Insertion{data: nowptr, where: oldptr, size: nowsize,
                               line_to: oldi + 1, line_from: nowi + 1});
    }
    std::reverse(dels->begin(), dels->end());
    std::reverse(ins->begin(), ins->end());
    scripts[i] = std::make_tuple(dels, ins);
  }
  return std::move(scripts);
}

}  // namespace diffcuda
