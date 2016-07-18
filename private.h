#ifndef DIFFCUDA_PRIVATE_H
#define DIFFCUDA_PRIVATE_H

#include <cstdio>
#include <cstdint>
#include "diffcuda.h"

#define MAXD 1024

using hash_t = diffcuda::hash_t;

#define ERR(...) fprintf(stderr, __VA_ARGS__)
#define PANIC(name, ret, ...) do { \
  ERR("%s:%d: " name " failed: %s\n", __FILE__, __LINE__, \
      ##__VA_ARGS__, cudaGetErrorString(cudaGetLastError())); \
  return ret; \
} while (false)

#define CUCH(cuda_call, ret) \
do { \
  auto __res = cuda_call; \
  if (__res != cudaSuccess) { \
    PANIC(#cuda_call, ret); \
  } \
} while (false)

#define CUMEMCPY(dst, src, size, flag, ret) \
do { if (cudaMemcpy(dst, src, size, flag) != cudaSuccess) { \
  PANIC("cudaMemcpy", ret); \
} } while(false)

#define CUMEMCPY_ASYNC(dst, src, size, flag, ret) \
do { if (cudaMemcpyAsync(dst, src, size, flag) != cudaSuccess) { \
  PANIC("cudaMemcpyAsync", ret); \
} } while(false)

#define CUMALLOC(dest, size, ret) do { \
  if (cudaMalloc(reinterpret_cast<void**>(&dest), size) != cudaSuccess) { \
    PANIC("cudaMalloc(" #dest ", %zu)", ret, size); \
  } \
} while(false)

#define CEIL(x, r) (x + r * ((x & (r - 1)) > 0) - (x & (r - 1)))
#define CEIL_PTR(x, r) (x + r * ((reinterpret_cast<intptr_t>(x) & (r - 1)) > 0) \
    - (reinterpret_cast<intptr_t>(x) & (r - 1)))

extern "C" {
bool myers_diff(
    int device, uint32_t size, uint32_t memo_size_, const hash_t **old,
    const uint32_t *old_size, const hash_t **now, const uint32_t *now_size,
    uint32_t *workspace, uint32_t *deletions, uint32_t *insertions);
}

constexpr size_t _doffset(size_t prev, int currd, int maxd) {
  return currd == maxd? prev :
         _doffset(prev + (2 * currd + 1) / (8 * sizeof(uint32_t)) + 1,
                  currd + 1, maxd);
}

constexpr size_t doffset(int D) {
  return _doffset(0, 0, D);
}

#endif // DIFFCUDA_PRIVATE_H
