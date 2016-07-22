Bulk text diff using NVIDIA CUDA
================================

This is an ongoing effort to accelerate diff-ing of text files in a data parallel fashion.
In other words, this project tries to improve on diff-ing many files at once and not
increasing the speed of a single diff operation.

The approach chosen is as follows:
- Extract line endings.
- Hash each line.
- Apply Myers algorithm to hashes on the GPU.

Results
-------
This project appeared to be 2x faster than a single-threaded libxdiff run.
The profiling results showed that 25% of time is spent on line endings and
hashing, which is done in OpenMP multithreaded mode.

License
-------
MIT license.
