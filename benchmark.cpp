#include <fcntl.h>
#include <linux/fs.h>
#include <sys/stat.h>
#include <unistd.h>

#include <fstream>
#include <chrono>

#include <cuda_runtime_api.h>

#include "diffcuda.h"
#include "private.h"

int main(int argc, const char **argv) {
  if (argc != 2) {
    PANIC("Usage: %s <file with pairs of paths>", 1, argv[0]);
  }
  std::vector<const uint8_t*> data_old, data_now;
  std::vector<std::unique_ptr<uint8_t []>> storage;
  std::vector<size_t> sizes_old, sizes_now;
  std::ifstream paths(argv[1]);
  std::string pold, pnow;
  printf("Reading files...\n");
  while (paths >> pold >> pnow) {
    if (data_old.size() % 100 == 0) {
      printf("\r%zu", data_old.size());
      fflush(stdout);
    }
    size_t old_size;
    {
      struct stat sstat;
      stat(pold.c_str(), &sstat);
      old_size = static_cast<size_t>(sstat.st_size);
    }
    sizes_old.push_back(old_size);
    storage.emplace_back(new uint8_t[old_size + BLOCK_SIZE * 2]);
    auto old_ptr = storage.back().get();
    old_ptr = CEIL_PTR(old_ptr, BLOCK_SIZE);
    data_old.push_back(old_ptr);
    auto file = open(pold.c_str(), O_RDONLY | O_DIRECT);
    auto size_read = read(file, old_ptr, CEIL(old_size, BLOCK_SIZE));
    if (size_read != static_cast<ssize_t>(old_size)) {
      PANIC("Failed to read %s", 1, pold.c_str());
    }
    close(file);

    size_t now_size;
    {
      struct stat sstat;
      stat(pnow.c_str(), &sstat);
      now_size = static_cast<size_t>(sstat.st_size);
    }
    sizes_now.push_back(now_size);
    storage.emplace_back(new uint8_t[now_size + BLOCK_SIZE * 2]);
    auto now_ptr = storage.back().get();
    now_ptr = CEIL_PTR(now_ptr, BLOCK_SIZE);
    data_now.push_back(now_ptr);
    file = open(pnow.c_str(), O_RDONLY | O_DIRECT);
    size_read = read(file, now_ptr, CEIL(now_size, BLOCK_SIZE));
    if (size_read != static_cast<ssize_t>(now_size)) {
      PANIC("Failed to read %s", 1, pnow.c_str());
    }
    close(file);
  }

  printf("\nStarted benchmark\n");
  std::chrono::high_resolution_clock timer;
  auto start = timer.now();
  auto scripts = diffcuda::diff(
      data_old.data(), sizes_old.data(), data_now.data(), sizes_now.data(),
      data_old.size());
  auto finish = timer.now();
  printf("Processed %zu pairs in %li ms\n", scripts.size(),
         std::chrono::duration_cast<std::chrono::milliseconds>(finish - start).count());

  /*
  for (auto& s : scripts) {
    for (auto& del : *std::get<0>(s)) {
      printf("- %u %.*s", del.line, del.size, del.data);
    }
    for (auto& ins : *std::get<1>(s)) {
      printf("+ %u@%u %.*s", ins.line_to, ins.line_from, ins.size, ins.data);
    }
  }
  */

  return 0;
}