#include "private.h"

#define BLOCK_SIZE 1024

__global__ void myers_diff_cuda(
    uint32_t size, const hash_t *__restrict__ *old, const uint32_t *old_size,
    const hash_t *__restrict__ *now, const uint32_t *now_size,
    uint32_t *workspace, uint32_t *deletions, uint32_t *insertions) {

}

extern "C" {
void myers_diff(
    uint32_t size, const hash_t **old, const uint32_t *old_size,
    const hash_t **now, const uint32_t *now_size,
    uint32_t *workspace, uint32_t *deletions, uint32_t *insertions) {
  dim3 block(BLOCK_SIZE, 1, 1);
  dim3 grid(size / block.x + 1, 1, 1);
  myers_diff_cuda<<<block, grid>>>(size, old, old_size, now, now_size,
                                   workspace, deletions, insertions);
}
}