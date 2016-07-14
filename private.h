#ifndef DIFFCUDA_PRIVATE_H
#define DIFFCUDA_PRIVATE_H

#include <cstdio>

#define ERR(...) fprintf(stderr, __VA_ARGS__)
#define PANIC(name, ret, ...) do { \
  ERR("%s:%d: " name " failed: %s", __FILE__, __LINE__, \
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

#endif //DIFFCUDA_PRIVATE_H
