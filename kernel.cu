#include "private.h"
#include <ciso646>

#define BLOCK_SIZE 1024

__constant__ uint32_t memo_size;

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
  uint32_t zp = M + N + 1;
  uint32_t *state = workspace + index * (2 * MAXD + 1);
  uint32_t *memo = workspace + size * (2 * MAXD + 1) + memo_size * index;
  uint32_t *mydels = deletions + index * MAXD;
  uint32_t *myins = insertions + index * 2 * MAXD;
  for (int D = 0; D < MAXD; D++) {
    for (int k = -D; k <= D; k += 2) {
      uint32_t x;
      bool ref; // true - up, false - left
      if (k == -D or (k != D and state[k - 1 + zp] < state[k + 1 + zp])) {
        x = state[k + 1 + zp];
        ref = true;
      } else {
        x = state[k - 1 + zp] + 1;
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
    uint32_t size, uint32_t memo_size_, const hash_t **old,
    const uint32_t *old_size, const hash_t **now, const uint32_t *now_size,
    uint32_t *workspace, uint32_t *deletions, uint32_t *insertions) {
  dim3 block(BLOCK_SIZE, 1, 1);
  dim3 grid(size / block.x + 1, 1, 1);
  CUCH(cudaMemcpyToSymbol(memo_size, &memo_size_, sizeof(uint32_t)), false);
  myers_diff_cuda<<<block, grid>>>(size, old, old_size, now, now_size,
                                   workspace, deletions, insertions);
  return true;
}
}
