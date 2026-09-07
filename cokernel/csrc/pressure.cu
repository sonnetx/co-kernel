// Synthetic co-runner ("pressure") kernel with register and shared-memory knobs.
//
// Design: one persistent block per SM (configurable), each spinning on an FMA
// chain over NREG live float accumulators until the host raises a stop flag.
// The loop-carried dependence on every accumulator keeps all NREG values live,
// so register consumption per thread tracks NREG. Thread count is held fixed
// across register levels by the caller, so issue pressure on the FMA pipe and
// thread-slot occupancy stay constant while the register-file footprint
// sweeps. Comparisons BETWEEN levels therefore isolate the register axis;
// comparisons between an idle device and any level do not, since the neighbor
// also consumes issue slots and cache.
//
// Shared memory is a second, independent axis, requested as dynamic shared
// memory at launch. Dynamic shared memory is reserved by the launch parameter
// whether or not the kernel reads it, so it constrains occupancy regardless;
// we touch one element anyway to keep the intent legible and defeat any
// future compiler cleverness.
//
// The stop flag lives in DEVICE memory (an L2 hit for the spinning kernel; a
// mapped-host flag turns 33K threads into a PCIe read flood that distorts the
// memory subsystem far beyond the intended contention, slowing every
// co-located kernel a uniform 500-770x). The host raises it with
// cudaMemcpyAsync from pinned memory on a dedicated non-blocking copy stream:
// the copy engine executes regardless of SM starvation, so the stop path
// cannot deadlock behind starved kernels. The flag is checked once per outer
// loop (~1024*NREG FMAs), tens of microseconds at H100 clocks.
//
// HACKING: to change the register sweep, edit COKERNEL_PRESSURE_LEVELS below.
// Both the launch dispatch and the reported attribute table are generated from
// that one list, and Python reads the compiled levels back via levels(), so
// nothing else needs to change. Each level costs one template instantiation of
// compile time.

#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

// Register-pressure levels, in live accumulators per thread. Add or remove a
// line to change the sweep. On H100 with 256 threads/block these compile to
// 40, 127, 255, and 255 registers per thread respectively; __launch_bounds__
// caps the achieved count at 255, so levels above ~168 are indistinguishable.
// Query the achieved counts with reg_info() rather than assuming them.
#define COKERNEL_PRESSURE_LEVELS(X) \
  X(32)                             \
  X(96)                             \
  X(168)                            \
  X(240)

#define CUDA_OK(expr)                                                       \
  do {                                                                      \
    cudaError_t _e = (expr);                                                \
    TORCH_CHECK(_e == cudaSuccess, "CUDA error: ", cudaGetErrorString(_e)); \
  } while (0)

namespace {

int* h_flag = nullptr;   // pinned staging for the async flag copy
int* d_stop = nullptr;   // device-memory stop flag
float* d_sink = nullptr;
unsigned long long* d_iters = nullptr;      // progress counter (device)
unsigned long long* h_iters = nullptr;      // pinned staging for reading it
cudaStream_t pressure_stream = nullptr;
cudaStream_t copy_stream = nullptr;
bool active = false;

template <int NREG>
__global__ void __launch_bounds__(256, 1)
    pressure_kernel(const volatile int* stop, float* sink, float seed,
                    int smem_floats, unsigned long long* iters) {
  extern __shared__ float smem[];
  float acc[NREG];
#pragma unroll
  for (int i = 0; i < NREG; ++i) {
    acc[i] = seed + static_cast<float>(i) + static_cast<float>(threadIdx.x);
  }
  // Touch the requested shared memory once so the request is unambiguously
  // live. Occupancy is constrained by the launch parameter either way.
  if (smem_floats > 0) {
    for (int i = threadIdx.x; i < smem_floats; i += blockDim.x) smem[i] = 1.0f;
    __syncthreads();
    acc[0] += smem[threadIdx.x % smem_floats];
  }
  while (*stop == 0) {
#pragma unroll 1
    for (int outer = 0; outer < 1024; ++outer) {
#pragma unroll
      for (int i = 0; i < NREG; ++i) {
        acc[i] = fmaf(acc[i], 1.0000001f, 1e-7f);
      }
    }
    // Progress tick: one 8-byte atomic per block per flag-check batch
    // (~10-100 us), sampled by the host through the copy engine while the
    // kernel runs. The rate a candidate suppresses these ticks is the
    // aggression measurement, so the tick itself must stay too cheap to be
    // part of the contention: one lane, once per batch.
    if (threadIdx.x == 0) atomicAdd(iters, 1ULL);
  }
  float s = 0.f;
#pragma unroll
  for (int i = 0; i < NREG; ++i) s += acc[i];
  // Never true for our seeds; defeats dead-code elimination.
  if (s == -1234.5678f) *sink = s;
}

void ensure_state() {
  if (h_flag == nullptr) {
    CUDA_OK(cudaHostAlloc((void**)&h_flag, sizeof(int), cudaHostAllocDefault));
    CUDA_OK(cudaHostAlloc((void**)&h_iters, sizeof(unsigned long long),
                          cudaHostAllocDefault));
    CUDA_OK(cudaMalloc(&d_stop, sizeof(int)));
    CUDA_OK(cudaMalloc(&d_sink, sizeof(float)));
    CUDA_OK(cudaMalloc(&d_iters, sizeof(unsigned long long)));
    CUDA_OK(cudaMemset(d_iters, 0, sizeof(unsigned long long)));
    CUDA_OK(cudaStreamCreateWithFlags(&pressure_stream, cudaStreamNonBlocking));
    CUDA_OK(cudaStreamCreateWithFlags(&copy_stream, cudaStreamNonBlocking));
  }
}

void set_flag(int value) {
  *h_flag = value;
  CUDA_OK(cudaMemcpyAsync(d_stop, h_flag, sizeof(int), cudaMemcpyHostToDevice,
                          copy_stream));
  CUDA_OK(cudaStreamSynchronize(copy_stream));
}

int num_sms() {
  int dev = 0, sms = 0;
  CUDA_OK(cudaGetDevice(&dev));
  CUDA_OK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev));
  return sms;
}

// Cumulative flag-check batches completed by all blocks since start(). Read
// through the copy engine, so sampling works while the kernel saturates the
// SMs. Units are arbitrary but consistent within one start/stop cycle, which
// is all a rate ratio needs.
int64_t progress() {
  ensure_state();
  CUDA_OK(cudaMemcpyAsync(h_iters, d_iters, sizeof(unsigned long long),
                          cudaMemcpyDeviceToHost, copy_stream));
  CUDA_OK(cudaStreamSynchronize(copy_stream));
  return (int64_t)*h_iters;
}

// The register levels this build supports, so Python need not duplicate them.
std::vector<int64_t> levels() {
  std::vector<int64_t> out;
#define COKERNEL_ONE_LEVEL(N) out.push_back(N);
  COKERNEL_PRESSURE_LEVELS(COKERNEL_ONE_LEVEL)
#undef COKERNEL_ONE_LEVEL
  return out;
}

// (nreg level, achieved regs/thread, static smem bytes, local bytes) per level.
std::vector<std::vector<int64_t>> reg_info() {
  std::vector<std::vector<int64_t>> out;
  cudaFuncAttributes attr;
#define COKERNEL_ONE_INFO(N)                                              \
  CUDA_OK(cudaFuncGetAttributes(&attr, pressure_kernel<N>));               \
  out.push_back({N, attr.numRegs, (int64_t)attr.sharedSizeBytes,           \
                 (int64_t)attr.localSizeBytes});
  COKERNEL_PRESSURE_LEVELS(COKERNEL_ONE_INFO)
#undef COKERNEL_ONE_INFO
  return out;
}

void start(int64_t nreg, int64_t blocks_per_sm, int64_t threads,
           int64_t smem_bytes) {
  ensure_state();
  TORCH_CHECK(!active, "pressure kernel already running");
  TORCH_CHECK(threads > 0 && threads <= 256, "threads must be in (0, 256]");
  TORCH_CHECK(smem_bytes >= 0, "smem_bytes must be non-negative");
  set_flag(0);
  CUDA_OK(cudaMemsetAsync(d_iters, 0, sizeof(unsigned long long), copy_stream));
  CUDA_OK(cudaStreamSynchronize(copy_stream));
  dim3 grid(num_sms() * (int)blocks_per_sm);
  dim3 block((int)threads);
  int smem = (int)smem_bytes;
  int smem_floats = smem / (int)sizeof(float);
  bool launched = false;

#define COKERNEL_ONE_LAUNCH(N)                                                \
  if (!launched && nreg == N) {                                               \
    /* Dynamic shared memory above 48 KB requires explicit opt-in. */         \
    if (smem > 48 * 1024) {                                                   \
      CUDA_OK(cudaFuncSetAttribute(                                           \
          pressure_kernel<N>, cudaFuncAttributeMaxDynamicSharedMemorySize,    \
          smem));                                                             \
    }                                                                         \
    pressure_kernel<N><<<grid, block, smem, pressure_stream>>>(               \
        d_stop, d_sink, 0.5f, smem_floats, d_iters);                          \
    launched = true;                                                          \
  }
  COKERNEL_PRESSURE_LEVELS(COKERNEL_ONE_LAUNCH)
#undef COKERNEL_ONE_LAUNCH

  TORCH_CHECK(launched, "nreg ", nreg,
              " is not a compiled pressure level; add it to "
              "COKERNEL_PRESSURE_LEVELS in pressure.cu and rebuild");
  CUDA_OK(cudaGetLastError());
  active = true;
}

void stop() {
  if (!active) return;
  set_flag(1);
  CUDA_OK(cudaStreamSynchronize(pressure_stream));
  active = false;
}

bool running() { return active; }

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("start", &start,
        "Launch pressure kernel (nreg, blocks_per_sm, threads, smem_bytes)",
        pybind11::arg("nreg"), pybind11::arg("blocks_per_sm") = 1,
        pybind11::arg("threads") = 256, pybind11::arg("smem_bytes") = 0);
  m.def("stop", &stop, "Stop pressure kernel and synchronize");
  m.def("running", &running, "Whether the pressure kernel is active");
  m.def("progress", &progress,
        "Flag-check batches completed by all blocks since start()");
  m.def("num_sms", &num_sms, "SM count of device 0");
  m.def("levels", &levels, "Register levels compiled into this build");
  m.def("reg_info", &reg_info,
        "Per-level (nreg, regs/thread, static smem, local) from "
        "cudaFuncGetAttributes");
}
