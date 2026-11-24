// Reproducible matrix-transpose benchmarks for FlashGPU-Sim.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#define CUDA_CHECK(call) do {                                                   \
  cudaError_t error__ = (call);                                                 \
  if (error__ != cudaSuccess) {                                                 \
    std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,         \
                 cudaGetErrorString(error__));                                  \
    std::exit(EXIT_FAILURE);                                                    \
  }                                                                             \
} while (0)

constexpr int kTileDim = 32;
constexpr int kBlockRows = 8;

struct Options {
  std::string kernel = "shared";
  int rows = 1024;
  int cols = 1024;
  unsigned seed = 1;
};

static Options parse_options(int argc, char **argv) {
  Options options;
  for (int i = 1; i < argc; ++i) {
    const std::string argument(argv[i]);
    auto value = [&]() {
      if (++i >= argc) {
        std::fprintf(stderr, "Missing argument value\n");
        std::exit(EXIT_FAILURE);
      }
      return argv[i];
    };
    if (argument == "--kernel") options.kernel = value();
    else if (argument == "--rows") options.rows = std::atoi(value());
    else if (argument == "--cols") options.cols = std::atoi(value());
    else if (argument == "--seed") options.seed = unsigned(std::strtoul(value(), nullptr, 10));
    else {
      std::fprintf(stderr, "Unknown option: %s\n", argument.c_str());
      std::exit(EXIT_FAILURE);
    }
  }
  const bool known_kernel = options.kernel == "naive" ||
                            options.kernel == "shared" ||
                            options.kernel == "vector";
  if (!known_kernel || options.rows <= 0 || options.cols <= 0) {
    std::fprintf(stderr,
                 "Usage: --kernel naive|shared|vector --rows R --cols C [--seed S]\n");
    std::exit(EXIT_FAILURE);
  }
  if (options.kernel == "vector" &&
      ((options.rows % kTileDim) != 0 || (options.cols % kTileDim) != 0)) {
    std::fprintf(stderr,
                 "The vector kernel requires rows and cols divisible by %d\n",
                 kTileDim);
    std::exit(EXIT_FAILURE);
  }
  return options;
}

static void initialize(std::vector<float> &values, unsigned seed) {
  unsigned state = seed;
  for (float &value : values) {
    state = 1664525u * state + 1013904223u;
    value = float(int((state >> 8) & 0xffff) - 32768) / 8192.0f;
  }
}

__global__ void transpose_naive(const float *__restrict__ input,
                                float *__restrict__ output,
                                int rows, int cols) {
  const int x = int(blockIdx.x) * kTileDim + int(threadIdx.x);
  const int y = int(blockIdx.y) * kTileDim + int(threadIdx.y);
  #pragma unroll
  for (int offset = 0; offset < kTileDim; offset += kBlockRows) {
    if (x < cols && y + offset < rows) {
      output[size_t(x) * rows + y + offset] =
          input[size_t(y + offset) * cols + x];
    }
  }
}

__global__ void transpose_shared(const float *__restrict__ input,
                                 float *__restrict__ output,
                                 int rows, int cols) {
  __shared__ float tile[kTileDim][kTileDim + 1];
  int x = int(blockIdx.x) * kTileDim + int(threadIdx.x);
  int y = int(blockIdx.y) * kTileDim + int(threadIdx.y);
  #pragma unroll
  for (int offset = 0; offset < kTileDim; offset += kBlockRows) {
    if (x < cols && y + offset < rows) {
      tile[threadIdx.y + offset][threadIdx.x] =
          input[size_t(y + offset) * cols + x];
    }
  }
  __syncthreads();

  x = int(blockIdx.y) * kTileDim + int(threadIdx.x);
  y = int(blockIdx.x) * kTileDim + int(threadIdx.y);
  #pragma unroll
  for (int offset = 0; offset < kTileDim; offset += kBlockRows) {
    if (x < rows && y + offset < cols) {
      output[size_t(y + offset) * rows + x] =
          tile[threadIdx.x][threadIdx.y + offset];
    }
  }
}

__global__ void transpose_shared_vector(const float *__restrict__ input,
                                        float *__restrict__ output,
                                        int rows, int cols) {
  __shared__ float tile[kTileDim][kTileDim + 1];
  const int local_vector = int(threadIdx.x) * 4;
  const int local_row = int(threadIdx.y);
  const int input_x = int(blockIdx.x) * kTileDim + local_vector;
  const int input_y = int(blockIdx.y) * kTileDim + local_row;

  const float4 input_value = *reinterpret_cast<const float4 *>(
      input + size_t(input_y) * cols + input_x);
  tile[local_row][local_vector] = input_value.x;
  tile[local_row][local_vector + 1] = input_value.y;
  tile[local_row][local_vector + 2] = input_value.z;
  tile[local_row][local_vector + 3] = input_value.w;
  __syncthreads();

  const int output_x = int(blockIdx.y) * kTileDim + local_vector;
  const int output_y = int(blockIdx.x) * kTileDim + local_row;
  const float4 output_value = make_float4(
      tile[local_vector][local_row], tile[local_vector + 1][local_row],
      tile[local_vector + 2][local_row], tile[local_vector + 3][local_row]);
  *reinterpret_cast<float4 *>(output + size_t(output_y) * rows + output_x) =
      output_value;
}

static bool verify(const std::vector<float> &input,
                   const std::vector<float> &output,
                   int rows, int cols) {
  double max_error = 0.0;
  for (int row = 0; row < rows; ++row) {
    for (int col = 0; col < cols; ++col) {
      const double error = std::fabs(
          double(output[size_t(col) * rows + row]) -
          double(input[size_t(row) * cols + col]));
      max_error = std::max(max_error, error);
    }
  }
  std::printf("max_abs_error=%.8e\n", max_error);
  return max_error == 0.0;
}

int main(int argc, char **argv) {
  const Options options = parse_options(argc, argv);
  const size_t element_count = size_t(options.rows) * options.cols;
  const size_t matrix_bytes = element_count * sizeof(float);
  std::vector<float> input(element_count), output(element_count);
  initialize(input, options.seed);

  float *device_input = nullptr;
  float *device_output = nullptr;
  CUDA_CHECK(cudaMalloc(&device_input, matrix_bytes));
  CUDA_CHECK(cudaMalloc(&device_output, matrix_bytes));
  CUDA_CHECK(cudaMemcpy(device_input, input.data(), matrix_bytes,
                        cudaMemcpyHostToDevice));

  const dim3 grid((options.cols + kTileDim - 1) / kTileDim,
                  (options.rows + kTileDim - 1) / kTileDim);
  if (options.kernel == "naive") {
    transpose_naive<<<grid, dim3(kTileDim, kBlockRows)>>>(
        device_input, device_output, options.rows, options.cols);
  } else if (options.kernel == "shared") {
    transpose_shared<<<grid, dim3(kTileDim, kBlockRows)>>>(
        device_input, device_output, options.rows, options.cols);
  } else {
    transpose_shared_vector<<<grid, dim3(kTileDim / 4, kTileDim)>>>(
        device_input, device_output, options.rows, options.cols);
  }
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaMemcpy(output.data(), device_output, matrix_bytes,
                        cudaMemcpyDeviceToHost));

  std::printf("benchmark=matrix_transpose\n");
  std::printf("kernel_variant=%s\n", options.kernel.c_str());
  std::printf("rows=%d\ncols=%d\n", options.rows, options.cols);
  std::printf("algorithmic_read_bytes=%zu\n", matrix_bytes);
  std::printf("algorithmic_write_bytes=%zu\n", matrix_bytes);
  std::printf("algorithmic_total_bytes=%zu\n", matrix_bytes * 2);
  const bool passed = verify(input, output, options.rows, options.cols);
  std::printf(passed ? "Test PASSED\n" : "Test FAILED\n");

  CUDA_CHECK(cudaFree(device_output));
  CUDA_CHECK(cudaFree(device_input));
  return passed ? EXIT_SUCCESS : EXIT_FAILURE;
}
