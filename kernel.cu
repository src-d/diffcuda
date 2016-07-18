#include "private.h"
#include <ciso646>

#define BLOCK_SIZE 1024

__constant__ uint32_t memo_size;
__constant__ int kindergarten_D;

__global__ void myers_diff_cuda(
    uint32_t size, const hash_t *__restrict__ *old, const uint32_t *old_size,
    const hash_t *__restrict__ *now, const uint32_t *now_size,
    uint32_t *workspace, uint32_t *deletions, uint32_t *insertions) {
  uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= size) {
    return;
  }
  const hash_t *oldh = old[index];
  const hash_t *nowh = now[index];
  uint32_t N = old_size[index];
  uint32_t M = now_size[index];
  uint32_t *adult_state = workspace + index * (2 * MAXD + 1);
  int adult_zp = M + N + 1;
  extern __shared__ uint32_t kindergarten[];
  int kindergarten_size = kindergarten_D * 2 + 1;
  uint32_t *kindergarten_state = kindergarten + blockIdx.x * kindergarten_size;
  memset(kindergarten_state, 0, kindergarten_size * sizeof(uint32_t));
  int zp = kindergarten_D + 1;
  uint32_t *state = kindergarten_state;
  uint32_t *memo = workspace + size * (2 * MAXD + 1) + memo_size * index;
  uint32_t *mydels = deletions + index * MAXD;
  uint32_t *myins = insertions + index * 2 * MAXD;
  for (int D = 0; D < MAXD; D++) {
    if (D == kindergarten_D + 1) {
      state = adult_state;
      memcpy(state + adult_zp - zp, kindergarten_state,
             kindergarten_size * sizeof(uint32_t));
      zp = adult_zp;
    }
    for (int k = -D; k <= D; k += 2) {
      uint32_t x;
      bool ref; // true - up, false - left
      uint32_t up_state, left_state;
      if (k == -D or (k != D and
          (left_state = state[k - 1 + zp]) < (up_state = state[k + 1 + zp]))) {
        x = up_state;
        ref = true;
      } else {
        x = left_state + 1;
        ref = false;
      }
      uint32_t y = x - k;
      while (x < N and y < M and oldh[x] == nowh[y]) {
        x++; y++;
      }
      auto pos = k + zp;
      state[pos] = x;
      if (ref) {
        memo[pos / 32] |= 1 << (pos % 32);
      }
      if (x >= N and y >= M) {
        for(; D > 0; D--) {
          if (x <= N and y <= M) {
            while (x > 0 and y > 0 and oldh[x - 1] == nowh[y - 1]) {
              x--; y--;
            }
          }
          if (x == 0 and y == 0) {
            break;
          }
          auto pos = k + zp;
          if (memo[pos / 32] & (1 << (pos % 32))) {
            k++; y--;
            myins[0] = x;
            myins[1] = y;
            myins += 2;
          } else {
            k--; x--;
            mydels[0] = x;
            mydels++;
          }
          memo -= (2 * D - 1) / 32 + 1;
        }
        mydels[0] = UINT32_MAX;
        myins[0] = myins[1] = UINT32_MAX;
        return;
      }
    }
    memo += (2 * D + 1) / 32 + 1;
  }
  // if we are here then we've failed
  mydels[0] = UINT32_MAX;
  myins[0] = myins[1] = UINT32_MAX;
}

extern "C" {
bool myers_diff(
    int device, uint32_t size, uint32_t memo_size_, const hash_t **old,
    const uint32_t *old_size, const hash_t **now, const uint32_t *now_size,
    uint32_t *workspace, uint32_t *deletions, uint32_t *insertions) {
  dim3 block(min(BLOCK_SIZE, CEIL(size, 32)), 1, 1);
  dim3 grid(size / block.x + (size % block.x == 0? 0 : 1), 1, 1);
  CUCH(cudaMemcpyToSymbol(memo_size, &memo_size_, sizeof(uint32_t)), false);
  #ifndef SHMEM_SIZE
  cudaDeviceProp props;
  CUCH(cudaGetDeviceProperties(&props, device), false);
  const size_t shmem_size = props.sharedMemPerBlock;
  #else
  const size_t shmem_size = SHMEM_SIZE;
  #endif
  uint32_t kindergarten_D_ = (shmem_size / (block.x * sizeof(uint32_t)) - 1) / 2;
  CUCH(cudaMemcpyToSymbol(kindergarten_D, &kindergarten_D_, sizeof(uint32_t)),
       false);
  myers_diff_cuda<<<block, grid, shmem_size>>>(
      size, old, old_size, now, now_size, workspace, deletions, insertions);
  return true;
}
}
