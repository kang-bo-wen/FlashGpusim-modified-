// FlashGPU-Sim matrix transpose bandwidth benchmark.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#define CUDA_CHECK(call) do { \
  cudaError_t error__ = (call); \
  if (error__ != cudaSuccess) { \
    std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(error__)); \
    std::exit(EXIT_FAILURE); \
  } \
} while (0)

struct Options {
  int version = 2;
  int rows = 4096;
  int cols = 4096;
  int seed = 1;
};

static Options parse_options(int argc, char **argv) {
  Options o;
  for (int i = 1; i < argc; ++i) {
    std::string arg(argv[i]);
    auto value = [&]() {
      if (++i >= argc) { std::fprintf(stderr, "Missing argument value\n"); std::exit(EXIT_FAILURE); }
      return argv[i];
    };
    if (arg == "--version") o.version = std::atoi(value());
    else if (arg == "--rows") o.rows = std::atoi(value());
    else if (arg == "--cols") o.cols = std::atoi(value());
    else if (arg == "--seed") o.seed = std::atoi(value());
    else { std::fprintf(stderr, "Unknown option: %s\n", arg.c_str()); std::exit(EXIT_FAILURE); }
  }
  if (o.version < 0 || o.version > 3 || o.rows <= 0 || o.cols <= 0) {
    std::fprintf(stderr, "Usage: --version 0|1|2|3 --rows R --cols C\n");
    std::exit(EXIT_FAILURE);
  }
  return o;
}

static void initialize(std::vector<float> &values, int seed) {
  unsigned state = unsigned(seed);
  for (float &value : values) {
    state = 1664525u * state + 1013904223u;
    value = float(int((state >> 8) & 0xffff) - 32768) / 8192.0f;
  }
}

__global__ void copy_float4_grid_stride(const float4 *input, float4 *output, size_t vector_count) {
  size_t index = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  size_t stride = size_t(gridDim.x) * blockDim.x;
  for (; index < vector_count; index += stride) output[index] = input[index];
}

__global__ void transpose_naive(const float *input, float *output, int rows, int cols) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x < cols && y < rows) output[size_t(x) * rows + y] = input[size_t(y) * cols + x];
}

template<int TILE_DIM, int BLOCK_ROWS>
__global__ void transpose_tiled(const float *input, float *output, int rows, int cols) {
  __shared__ float tile[TILE_DIM][TILE_DIM + 1];
  int x = blockIdx.x * TILE_DIM + threadIdx.x;
  int y = blockIdx.y * TILE_DIM + threadIdx.y;
  #pragma unroll
  for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
    if (x < cols && y + j < rows) tile[threadIdx.y + j][threadIdx.x] = input[size_t(y + j) * cols + x];
  __syncthreads();
  x = blockIdx.y * TILE_DIM + threadIdx.x;
  y = blockIdx.x * TILE_DIM + threadIdx.y;
  #pragma unroll
  for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
    if (x < rows && y + j < cols) output[size_t(y + j) * rows + x] = tile[threadIdx.x][threadIdx.y + j];
}

__global__ void transpose_tiled_float4(const float *input, float *output, int rows, int cols) {
  __shared__ float tile[32][33];
  int local_x = threadIdx.x * 4;
  int local_y = threadIdx.y;
  int input_x = blockIdx.x * 32 + local_x;
  int input_y = blockIdx.y * 32 + local_y;

  if (input_y < rows) {
    if (input_x + 3 < cols) {
      float4 value = *reinterpret_cast<const float4 *>(input + size_t(input_y) * cols + input_x);
      tile[local_y][local_x] = value.x;
      tile[local_y][local_x + 1] = value.y;
      tile[local_y][local_x + 2] = value.z;
      tile[local_y][local_x + 3] = value.w;
    } else {
      #pragma unroll
      for (int k = 0; k < 4; ++k)
        if (input_x + k < cols) tile[local_y][local_x + k] = input[size_t(input_y) * cols + input_x + k];
    }
  }
  __syncthreads();

  int output_x = blockIdx.y * 32 + local_x;
  int output_y = blockIdx.x * 32 + local_y;
  if (output_y < cols) {
    if (output_x + 3 < rows) {
      float4 value = make_float4(tile[local_x][local_y], tile[local_x + 1][local_y],
                                 tile[local_x + 2][local_y], tile[local_x + 3][local_y]);
      *reinterpret_cast<float4 *>(output + size_t(output_y) * rows + output_x) = value;
    } else {
      #pragma unroll
      for (int k = 0; k < 4; ++k)
        if (output_x + k < rows) output[size_t(output_y) * rows + output_x + k] = tile[local_x + k][local_y];
    }
  }
}

static bool verify(const std::vector<float> &input, const std::vector<float> &output, int rows, int cols, bool copy) {
  double max_error = 0.0;
  if (copy) {
    for (size_t i = 0; i < input.size(); ++i) max_error = std::max(max_error, std::fabs(double(output[i]) - input[i]));
  } else {
    for (int y = 0; y < rows; ++y)
      for (int x = 0; x < cols; ++x)
        max_error = std::max(max_error, std::fabs(double(output[size_t(x) * rows + y]) - input[size_t(y) * cols + x]));
  }
  std::printf("max_abs_error=%.8e\n", max_error);
  return max_error == 0.0;
}

int main(int argc, char **argv) {
  Options o = parse_options(argc, argv);
  size_t count = size_t(o.rows) * o.cols;
  size_t bytes = count * sizeof(float);
  std::vector<float> input(count), output(count);
  initialize(input, o.seed);
  float *d_input = nullptr, *d_output = nullptr;
  CUDA_CHECK(cudaMalloc(&d_input, bytes));
  CUDA_CHECK(cudaMalloc(&d_output, bytes));
  CUDA_CHECK(cudaMemcpy(d_input, input.data(), bytes, cudaMemcpyHostToDevice));

  if (o.version == 0) {
    if ((count & 3) != 0) {
      std::fprintf(stderr, "V0 requires element count divisible by 4\n");
      return EXIT_FAILURE;
    }
    int threads = 256;
    size_t vector_count = count / 4;
    int blocks = int(std::min<size_t>(4096, (vector_count + threads - 1) / threads));
    copy_float4_grid_stride<<<blocks, threads>>>(reinterpret_cast<const float4 *>(d_input),
                                                 reinterpret_cast<float4 *>(d_output), vector_count);
    std::printf("kernel_variant=v0_copy_float4_grid_stride\n");
  } else if (o.version == 1) {
    dim3 block(32, 8);
    dim3 grid((o.cols + 31) / 32, (o.rows + 7) / 8);
    transpose_naive<<<grid, block>>>(d_input, d_output, o.rows, o.cols);
    std::printf("kernel_variant=transpose_naive\n");
  } else if (o.version == 2) {
    dim3 block(32, 8);
    dim3 grid((o.cols + 31) / 32, (o.rows + 31) / 32);
    transpose_tiled<32, 8><<<grid, block>>>(d_input, d_output, o.rows, o.cols);
    std::printf("kernel_variant=v2_transpose_tiled_32x32_pad\n");
  } else {
    dim3 block(8, 32);
    dim3 grid((o.cols + 31) / 32, (o.rows + 31) / 32);
    transpose_tiled_float4<<<grid, block>>>(d_input, d_output, o.rows, o.cols);
    std::printf("kernel_variant=v3_transpose_tiled_float4\n");
  }

  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaMemcpy(output.data(), d_output, bytes, cudaMemcpyDeviceToHost));
  std::printf("operator=%s\nrows=%d\ncols=%d\nalgorithmic_bytes=%zu\n", o.version == 0 ? "copy" : "transpose", o.rows, o.cols, bytes * 2);
  bool passed = verify(input, output, o.rows, o.cols, o.version == 0);
  std::printf(passed ? "Test PASSED\n" : "Test FAILED\n");
  CUDA_CHECK(cudaFree(d_input));
  CUDA_CHECK(cudaFree(d_output));
  return passed ? EXIT_SUCCESS : EXIT_FAILURE;
}
