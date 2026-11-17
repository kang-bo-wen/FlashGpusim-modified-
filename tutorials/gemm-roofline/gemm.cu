// FlashGPU-Sim GEMM optimization ladder: V1-V3 plus CUTLASS SIMT.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"

#define CUDA_CHECK(call) do { \
  cudaError_t error__ = (call); \
  if (error__ != cudaSuccess) { \
    std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(error__)); \
    std::exit(EXIT_FAILURE); \
  } \
} while (0)

struct Options {
  int m = 256;
  int n = 256;
  int k = 256;
  int version = 1;
  int seed = 1;
  bool verify = true;
};

static void usage(const char *program) {
  std::printf("Usage: %s [--m M] [--n N] [--k K] [--version 1..4|cutlass] [--no-verify]\n", program);
}

static Options parse_options(int argc, char **argv) {
  Options options;
  for (int i = 1; i < argc; ++i) {
    std::string argument(argv[i]);
    auto require_value = [&](const char *name) -> const char * {
      if (i + 1 >= argc) {
        std::fprintf(stderr, "Missing value for %s\n", name);
        std::exit(EXIT_FAILURE);
      }
      return argv[++i];
    };
    if (argument == "--m") options.m = std::atoi(require_value("--m"));
    else if (argument == "--n") options.n = std::atoi(require_value("--n"));
    else if (argument == "--k") options.k = std::atoi(require_value("--k"));
    else if (argument == "--version") {
      std::string version(require_value("--version"));
      options.version = version == "cutlass" ? 4 : std::atoi(version.c_str());
    } else if (argument == "--seed") options.seed = std::atoi(require_value("--seed"));
    else if (argument == "--no-verify") options.verify = false;
    else if (argument == "--help" || argument == "-h") { usage(argv[0]); std::exit(EXIT_SUCCESS); }
    else { std::fprintf(stderr, "Unknown argument: %s\n", argv[i]); usage(argv[0]); std::exit(EXIT_FAILURE); }
  }
  if (options.m <= 0 || options.n <= 0 || options.k <= 0 || options.version < 1 || options.version > 4) {
    std::fprintf(stderr, "M, N, K must be positive and version must be 1..4 or cutlass.\n");
    std::exit(EXIT_FAILURE);
  }
  return options;
}

__global__ void gemm_v1(const float *a, const float *b, float *c, int m, int n, int k) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < m && col < n) {
    float sum = 0.0f;
    for (int inner = 0; inner < k; ++inner) sum = fmaf(a[row * k + inner], b[inner * n + col], sum);
    c[row * n + col] = sum;
  }
}

template<int Tile>
__global__ void gemm_v2(const float *a, const float *b, float *c, int m, int n, int k) {
  __shared__ float tile_a[Tile][Tile];
  __shared__ float tile_b[Tile][Tile];
  int row = blockIdx.y * Tile + threadIdx.y;
  int col = blockIdx.x * Tile + threadIdx.x;
  float sum = 0.0f;
  for (int base = 0; base < k; base += Tile) {
    tile_a[threadIdx.y][threadIdx.x] = (row < m && base + threadIdx.x < k) ? a[row * k + base + threadIdx.x] : 0.0f;
    tile_b[threadIdx.y][threadIdx.x] = (base + threadIdx.y < k && col < n) ? b[(base + threadIdx.y) * n + col] : 0.0f;
    __syncthreads();
    #pragma unroll
    for (int inner = 0; inner < Tile; ++inner) sum = fmaf(tile_a[threadIdx.y][inner], tile_b[inner][threadIdx.x], sum);
    __syncthreads();
  }
  if (row < m && col < n) c[row * n + col] = sum;
}

template<int BM, int BN, int BK, int TM, int TN>
__global__ void gemm_v3_register(const float *a, const float *b, float *c, int m, int n, int k) {
  __shared__ float tile_a[BM][BK + 1];
  __shared__ float tile_b[BK][BN + 1];
  int thread_linear = threadIdx.y * blockDim.x + threadIdx.x;
  int thread_count = blockDim.x * blockDim.y;
  int row_base = threadIdx.y * TM;
  // Keep each warp on adjacent columns. With a 32x8 block and TN=2, each
  // thread owns columns x and x+32, preserving fully coalesced warp stores.
  int col_base = threadIdx.x;
  int block_row = blockIdx.y * BM;
  int block_col = blockIdx.x * BN;
  float sums[TM][TN] = {};
  for (int base = 0; base < k; base += BK) {
    for (int linear = thread_linear; linear < BM * BK; linear += thread_count) {
      int tile_row = linear / BK;
      int tile_col = linear % BK;
      tile_a[tile_row][tile_col] = (block_row + tile_row < m && base + tile_col < k) ? a[(block_row + tile_row) * k + base + tile_col] : 0.0f;
    }
    for (int linear = thread_linear; linear < BK * BN; linear += thread_count) {
      int tile_row = linear / BN;
      int tile_col = linear % BN;
      tile_b[tile_row][tile_col] = (base + tile_row < k && block_col + tile_col < n) ? b[(base + tile_row) * n + block_col + tile_col] : 0.0f;
    }
    __syncthreads();
    #pragma unroll
    for (int inner = 0; inner < BK; ++inner) {
      float a_values[TM];
      float b_values[TN];
      #pragma unroll
      for (int row_offset = 0; row_offset < TM; ++row_offset) a_values[row_offset] = tile_a[row_base + row_offset][inner];
      #pragma unroll
      for (int col_offset = 0; col_offset < TN; ++col_offset)
        b_values[col_offset] =
            tile_b[inner][col_base + col_offset * blockDim.x];
      #pragma unroll
      for (int row_offset = 0; row_offset < TM; ++row_offset) {
        #pragma unroll
        for (int col_offset = 0; col_offset < TN; ++col_offset) sums[row_offset][col_offset] = fmaf(a_values[row_offset], b_values[col_offset], sums[row_offset][col_offset]);
      }
    }
    __syncthreads();
  }
  #pragma unroll
  for (int row_offset = 0; row_offset < TM; ++row_offset) {
    #pragma unroll
    for (int col_offset = 0; col_offset < TN; ++col_offset) {
      int row = block_row + row_base + row_offset;
      int col = block_col + col_base + col_offset * blockDim.x;
      if (row < m && col < n) c[row * n + col] = sums[row_offset][col_offset];
    }
  }
}

static void initialize(std::vector<float> &values, int seed) {
  unsigned state = static_cast<unsigned>(seed);
  for (float &value : values) {
    state = 1664525u * state + 1013904223u;
    value = static_cast<float>(int((state >> 8) & 0xffff) - 32768) / 32768.0f;
  }
}

static bool verify_result(const std::vector<float> &a, const std::vector<float> &b, const std::vector<float> &c, int m, int n, int k) {
  double max_error = 0.0;
  for (int row = 0; row < m; ++row) {
    for (int col = 0; col < n; ++col) {
      double reference = 0.0;
      for (int inner = 0; inner < k; ++inner) reference += double(a[row * k + inner]) * double(b[inner * n + col]);
      max_error = std::max(max_error, std::fabs(reference - double(c[row * n + col])));
    }
  }
  std::printf("max_abs_error=%.8e\n", max_error);
  return max_error <= 5e-3 * std::max(1, k);
}

int main(int argc, char **argv) {
  Options options = parse_options(argc, argv);
  size_t a_bytes = size_t(options.m) * options.k * sizeof(float);
  size_t b_bytes = size_t(options.k) * options.n * sizeof(float);
  size_t c_bytes = size_t(options.m) * options.n * sizeof(float);
  std::vector<float> h_a(size_t(options.m) * options.k);
  std::vector<float> h_b(size_t(options.k) * options.n);
  std::vector<float> h_c(size_t(options.m) * options.n, 0.0f);
  initialize(h_a, options.seed);
  initialize(h_b, options.seed + 17);
  float *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;
  CUDA_CHECK(cudaMalloc(&d_a, a_bytes));
  CUDA_CHECK(cudaMalloc(&d_b, b_bytes));
  CUDA_CHECK(cudaMalloc(&d_c, c_bytes));
  CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), a_bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), b_bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(d_c, 0, c_bytes));

  dim3 block;
  dim3 grid;
  const char *name = nullptr;
  if (options.version == 1) {
    name = "v1_naive"; block = dim3(16, 16); grid = dim3((options.n + 15) / 16, (options.m + 15) / 16);
    gemm_v1<<<grid, block>>>(d_a, d_b, d_c, options.m, options.n, options.k);
  } else if (options.version == 2) {
    name = "v2_shared_tiling"; block = dim3(16, 16); grid = dim3((options.n + 15) / 16, (options.m + 15) / 16);
    gemm_v2<16><<<grid, block>>>(d_a, d_b, d_c, options.m, options.n, options.k);
  } else if (options.version == 3) {
    name = "v3_register_tile_4x2_32x64"; block = dim3(32, 8); grid = dim3((options.n + 63) / 64, (options.m + 31) / 32);
    gemm_v3_register<32, 64, 8, 4, 2><<<grid, block>>>(d_a, d_b, d_c, options.m, options.n, options.k);
  } else {
    name = "v4_cutlass_simt";
    using CutlassGemm = cutlass::gemm::device::Gemm<
        float, cutlass::layout::RowMajor,
        float, cutlass::layout::RowMajor,
        float, cutlass::layout::RowMajor>;
    CutlassGemm gemm;
    CutlassGemm::Arguments arguments(
        {options.m, options.n, options.k},
        {d_a, options.k}, {d_b, options.n},
        {d_c, options.n}, {d_c, options.n}, {1.0f, 0.0f});
    cutlass::Status status = gemm(arguments);
    if (status != cutlass::Status::kSuccess) {
      std::fprintf(stderr, "CUTLASS GEMM failed with status %d\n", int(status));
      return EXIT_FAILURE;
    }
  }
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaMemcpy(h_c.data(), d_c, c_bytes, cudaMemcpyDeviceToHost));
  double flops = 2.0 * double(options.m) * options.n * options.k;
  double algorithmic_bytes = double(a_bytes + b_bytes + c_bytes);
  std::printf("kernel_variant=%s\nM=%d\nN=%d\nK=%d\nalgorithmic_flops=%.0f\nalgorithmic_bytes=%.0f\nalgorithmic_intensity=%.8f\n", name, options.m, options.n, options.k, flops, algorithmic_bytes, flops / algorithmic_bytes);
  bool passed = !options.verify || verify_result(h_a, h_b, h_c, options.m, options.n, options.k);
  std::printf(passed ? "Test PASSED\n" : "Test FAILED\n");
  CUDA_CHECK(cudaFree(d_a));
  CUDA_CHECK(cudaFree(d_b));
  CUDA_CHECK(cudaFree(d_c));
  return passed ? EXIT_SUCCESS : EXIT_FAILURE;
}
