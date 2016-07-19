#include <fcntl.h>
#include <linux/fs.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cstdio>
#include <cstring>
#include <fstream>
#include <chrono>
#include <memory>
#include <vector>

#include <xdiff/xdiff.h>

#define CEIL(x, r) (x + r * ((x & (r - 1)) > 0) - (x & (r - 1)))
#define CEIL_PTR(x, r) (x + r * ((reinterpret_cast<intptr_t>(x) & (r - 1)) > 0) \
    - (reinterpret_cast<intptr_t>(x) & (r - 1)))

int outf(void *, mmbuffer_t *, int) {
  return 0;
}

int main(int argc, const char **argv) {
  if (argc != 2) {
    fprintf(stderr, "Usage: %s <file with pairs of paths>", argv[0]);
    return 1;
  }
  std::vector<mmfile_t> data_old, data_now;
  std::vector<std::unique_ptr<char []>> storage;
  std::ifstream paths(argv[1]);
  std::string pold, pnow;
  while (paths >> pold >> pnow) {
    size_t old_size;
    {
      struct stat sstat;
      stat(pold.c_str(), &sstat);
      old_size = static_cast<size_t>(sstat.st_size);
    }
    storage.emplace_back(new char[old_size + BLOCK_SIZE * 2]);
    auto old_ptr = storage.back().get();
    old_ptr = CEIL_PTR(old_ptr, BLOCK_SIZE);
    data_old.push_back(mmfile_t{old_ptr, static_cast<long>(old_size)});
    auto file = open(pold.c_str(), O_RDONLY | O_DIRECT);
    auto size_read = read(file, old_ptr, CEIL(old_size, BLOCK_SIZE));
    if (size_read != static_cast<ssize_t>(old_size)) {
      fprintf(stderr, "Failed to read %s", pold.c_str());
      return 1;
    }
    close(file);

    size_t now_size;
    {
      struct stat sstat;
      stat(pnow.c_str(), &sstat);
      now_size = static_cast<size_t>(sstat.st_size);
    }
    storage.emplace_back(new char[now_size + BLOCK_SIZE * 2]);
    auto now_ptr = storage.back().get();
    now_ptr = CEIL_PTR(now_ptr, BLOCK_SIZE);
    data_now.push_back(mmfile_t{now_ptr, static_cast<long>(now_size)});
    file = open(pnow.c_str(), O_RDONLY | O_DIRECT);
    size_read = read(file, now_ptr, CEIL(now_size, BLOCK_SIZE));
    if (size_read != static_cast<ssize_t>(now_size)) {
      fprintf(stderr, "Failed to read %s", pnow.c_str());
      return 1;
    }
    close(file);
  }

  xdemitcb_t ecb;
	xpparam_t xpp;
	xdemitconf_t xecfg;
  memset(&xpp, 0, sizeof(xpp));
	xpp.flags = 0;
	memset(&xecfg, 0, sizeof(xecfg));
	xecfg.ctxlen = 3;
	ecb.outf = outf;
	ecb.priv = nullptr;

  std::chrono::high_resolution_clock timer;
  auto start = timer.now();
  for (size_t i = 0; i < data_old.size(); i++) {
    xdl_diff(&data_old[i], &data_now[i], &xpp, &xecfg, &ecb);
  }
  auto finish = timer.now();
  printf("Processed %zu pairs in %li ms\n", data_old.size(),
         std::chrono::duration_cast<std::chrono::milliseconds>(finish - start).count());
  return 0;
}