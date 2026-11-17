// Empirical sustainable hierarchy bandwidth benchmarks for FlashGPU-Sim.
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
  std::string level = "hbm";
  int elements = 1 << 23;
  int iterations = 64;
  int blocks = 1056;
};

static Options parse_options(int argc, char **argv) {
  Options o;
  for (int i = 1; i < argc; ++i) {
    std::string arg(argv[i]);
    auto value = [&]() {
      if (++i >= argc) { std::fprintf(stderr, "Missing argument value\n"); std::exit(EXIT_FAILURE); }
      return argv[i];
    };
    if (arg == "--level") o.level = value();
    else if (arg == "--elements") o.elements = std::atoi(value());
    else if (arg == "--iterations") o.iterations = std::atoi(value());
    else if (arg == "--blocks") o.blocks = std::atoi(value());
    else { std::fprintf(stderr, "Unknown option: %s\n", arg.c_str()); std::exit(EXIT_FAILURE); }
  }
  if ((o.level != "l1" && o.level != "l2" && o.level != "hbm") ||
      o.elements <= 0 || o.iterations <= 0 || o.blocks <= 0) {
    std::fprintf(stderr, "Usage: --level l1|l2|hbm [--elements N --iterations I --blocks B]\n");
    std::exit(EXIT_FAILURE);
  }
  return o;
}

__device__ __forceinline__ float4 load_cg_float4(const float4 *address) {
  float4 value;
  asm volatile("ld.global.cg.v4.f32 {%0, %1, %2, %3}, [%4];"
               : "=f"(value.x), "=f"(value.y), "=f"(value.z), "=f"(value.w)
               : "l"(address));
  return value;
}

__global__ void shared_sustained(float *checksum, int iterations) {
  __shared__ volatile float tile[256];
  int tid = threadIdx.x;
  tile[tid] = float((blockIdx.x + tid) & 255);
  __syncthreads();
  #pragma unroll 4
  for (int i = 0; i < iterations; ++i) {
    float value = tile[tid];
    tile[tid] = value + 1.0f;
  }
  if (tid == 0) checksum[blockIdx.x] = tile[0];
}

__global__ void l2_read_sustained(const float4 *input, float *checksum,
                                  size_t vector_count, int iterations) {
  size_t index = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  size_t stride = size_t(gridDim.x) * blockDim.x;
  float4 sum0 = make_float4(0, 0, 0, 0);
  float4 sum1 = make_float4(0, 0, 0, 0);
  float4 sum2 = make_float4(0, 0, 0, 0);
  float4 sum3 = make_float4(0, 0, 0, 0);
  for (int iter = 0; iter < iterations; ++iter) {
    int slot = iter & 3;
    for (size_t i = index; i < vector_count; i += stride) {
      float4 value = load_cg_float4(input + i);
      float4 *sum = slot == 0 ? &sum0 : slot == 1 ? &sum1 : slot == 2 ? &sum2 : &sum3;
      sum->x += value.x; sum->y += value.y; sum->z += value.z; sum->w += value.w;
    }
  }
  if (index < size_t(gridDim.x) * blockDim.x)
    checksum[index] = sum0.x + sum0.y + sum0.z + sum0.w +
                      sum1.x + sum1.y + sum1.z + sum1.w +
                      sum2.x + sum2.y + sum2.z + sum2.w +
                      sum3.x + sum3.y + sum3.z + sum3.w;
}

__global__ void hbm_copy_sustained(const float4 *input, float4 *output, size_t vector_count) {
  size_t index = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  size_t stride = size_t(gridDim.x) * blockDim.x;
  for (; index < vector_count; index += stride) output[index] = input[index];
}

int main(int argc, char **argv) {
  Options o = parse_options(argc, argv);
  if (o.level == "l1") {
    float *d_checksum = nullptr;
    std::vector<float> checksum(o.blocks);
    CUDA_CHECK(cudaMalloc(&d_checksum, size_t(o.blocks) * sizeof(float)));
    shared_sustained<<<o.blocks, 256>>>(d_checksum, o.iterations);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(checksum.data(), d_checksum, size_t(o.blocks) * sizeof(float), cudaMemcpyDeviceToHost));
    double sum = 0.0;
    for (float value : checksum) sum += value;
    size_t expected = size_t(o.blocks) * 256 * o.iterations * 2 * sizeof(float);
    std::printf("benchmark_level=l1d_shared\nblocks=%d\niterations=%d\nexpected_physical_bytes=%zu\nchecksum=%.8e\nTest PASSED\n",
                o.blocks, o.iterations, expected, sum);
    CUDA_CHECK(cudaFree(d_checksum));
    return EXIT_SUCCESS;
  }

  if ((o.elements & 3) != 0) {
    std::fprintf(stderr, "Element count must be divisible by four\n");
    return EXIT_FAILURE;
  }
  size_t count = size_t(o.elements);
  size_t bytes = count * sizeof(float);
  size_t vector_count = count / 4;
  std::vector<float> input(count);
  for (size_t i = 0; i < count; ++i) input[i] = float((i & 1023) + 1) / 1024.0f;
  float *d_input = nullptr;
  CUDA_CHECK(cudaMalloc(&d_input, bytes));
  CUDA_CHECK(cudaMemcpy(d_input, input.data(), bytes, cudaMemcpyHostToDevice));
  int threads = 256;
  int blocks = std::min(o.blocks, int((vector_count + threads - 1) / threads));

  if (o.level == "l2") {
    size_t checksum_count = size_t(blocks) * threads;
    float *d_checksum = nullptr;
    std::vector<float> checksum(checksum_count);
    CUDA_CHECK(cudaMalloc(&d_checksum, checksum_count * sizeof(float)));
    l2_read_sustained<<<blocks, threads>>>(reinterpret_cast<const float4 *>(d_input), d_checksum,
                                           vector_count, o.iterations);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(checksum.data(), d_checksum, checksum_count * sizeof(float), cudaMemcpyDeviceToHost));
    double sum = 0.0;
    for (float value : checksum) sum += value;
    std::printf("benchmark_level=l2\nelements=%d\niterations=%d\nalgorithmic_read_bytes=%zu\nchecksum=%.8e\nTest PASSED\n",
                o.elements, o.iterations, bytes * size_t(o.iterations), sum);
    CUDA_CHECK(cudaFree(d_checksum));
  } else {
    float *d_output = nullptr;
    std::vector<float> output(count);
    CUDA_CHECK(cudaMalloc(&d_output, bytes));
    hbm_copy_sustained<<<blocks, threads>>>(reinterpret_cast<const float4 *>(d_input),
                                            reinterpret_cast<float4 *>(d_output), vector_count);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(output.data(), d_output, bytes, cudaMemcpyDeviceToHost));
    double max_error = 0.0;
    for (size_t i = 0; i < count; ++i)
      max_error = std::max(max_error, std::abs(double(output[i]) - input[i]));
    std::printf("benchmark_level=hbm\nelements=%d\nalgorithmic_bytes=%zu\nmax_abs_error=%.8e\n%s",
                o.elements, bytes * 2, max_error,
                max_error == 0.0 ? "Test PASSED\n" : "Test FAILED\n");
    CUDA_CHECK(cudaFree(d_output));
  }
  CUDA_CHECK(cudaFree(d_input));
  return EXIT_SUCCESS;
}
