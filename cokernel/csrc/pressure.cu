// Synthetic co-runner ("pressure") kernel with a register-pressure knob.
//
// Design: one persistent block per SM (configurable), each spinning on an FMA
// chain over NREG live float accumulators until the host raises a stop flag in
// zero-copy mapped memory. The loop-carried dependence on every accumulator
// keeps all NREG values live, so register consumption per thread tracks NREG.
// Thread count is held fixed across register levels, so issue pressure on the
// FMA pipe and thread-slot occupancy stay constant while the register-file
// footprint sweeps. That makes the register axis the only thing that moves
// between pressure levels.
//
// The stop flag lives in cudaHostAllocMapped memory so the host can stop the
// kernel without queueing work behind it. The flag is checked once per outer
// loop (~256*NREG FMAs), a few microseconds at H100 clocks.

#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

#define CUDA_OK(expr)                                                       \
  do {                                                                      \
    cudaError_t _e = (expr);                                                \
    TORCH_CHECK(_e == cudaSuccess, "CUDA error: ", cudaGetErrorString(_e)); \
  } while (0)

namespace {

volatile int* h_stop = nullptr;
int* d_stop = nullptr;
float* d_sink = nullptr;
cudaStream_t pressure_stream = nullptr;
bool active = false;

template <int NREG>
__global__ void __launch_bounds__(256, 1)
    pressure_kernel(const volatile int* stop, float* sink, float seed) {
  float acc[NREG];
#pragma unroll
  for (int i = 0; i < NREG; ++i) {
    acc[i] = seed + static_cast<float>(i) + static_cast<float>(threadIdx.x);
  }
  while (*stop == 0) {
#pragma unroll 1
    for (int outer = 0; outer < 256; ++outer) {
#pragma unroll
      for (int i = 0; i < NREG; ++i) {
        acc[i] = fmaf(acc[i], 1.0000001f, 1e-7f);
      }
    }
  }
  float s = 0.f;
#pragma unroll
  for (int i = 0; i < NREG; ++i) s += acc[i];
  // Never true for our seeds; defeats dead-code elimination.
  if (s == -1234.5678f) *sink = s;
}

void ensure_state() {
  if (h_stop == nullptr) {
    CUDA_OK(cudaHostAlloc((void**)&h_stop, sizeof(int), cudaHostAllocMapped));
    CUDA_OK(cudaHostGetDevicePointer((void**)&d_stop, (void*)h_stop, 0));
    CUDA_OK(cudaMalloc(&d_sink, sizeof(float)));
    CUDA_OK(cudaStreamCreateWithFlags(&pressure_stream, cudaStreamNonBlocking));
  }
}

int num_sms() {
  int dev = 0, sms = 0;
  CUDA_OK(cudaGetDevice(&dev));
  CUDA_OK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev));
  return sms;
}

// (nreg_level, actual regs/thread, shared bytes, local bytes) per instantiation.
std::vector<std::vector<int64_t>> reg_info() {
  std::vector<std::vector<int64_t>> out;
  cudaFuncAttributes attr;
  CUDA_OK(cudaFuncGetAttributes(&attr, pressure_kernel<32>));
  out.push_back({32, attr.numRegs, (int64_t)attr.sharedSizeBytes, (int64_t)attr.localSizeBytes});
  CUDA_OK(cudaFuncGetAttributes(&attr, pressure_kernel<96>));
  out.push_back({96, attr.numRegs, (int64_t)attr.sharedSizeBytes, (int64_t)attr.localSizeBytes});
  CUDA_OK(cudaFuncGetAttributes(&attr, pressure_kernel<168>));
  out.push_back({168, attr.numRegs, (int64_t)attr.sharedSizeBytes, (int64_t)attr.localSizeBytes});
  CUDA_OK(cudaFuncGetAttributes(&attr, pressure_kernel<240>));
  out.push_back({240, attr.numRegs, (int64_t)attr.sharedSizeBytes, (int64_t)attr.localSizeBytes});
  return out;
}

void start(int64_t nreg, int64_t blocks_per_sm, int64_t threads) {
  ensure_state();
  TORCH_CHECK(!active, "pressure kernel already running");
  TORCH_CHECK(threads > 0 && threads <= 256, "threads must be in (0, 256]");
  *h_stop = 0;
  __sync_synchronize();
  dim3 grid(num_sms() * (int)blocks_per_sm);
  dim3 block((int)threads);
  switch (nreg) {
    case 32:
      pressure_kernel<32><<<grid, block, 0, pressure_stream>>>(d_stop, d_sink, 0.5f);
      break;
    case 96:
      pressure_kernel<96><<<grid, block, 0, pressure_stream>>>(d_stop, d_sink, 0.5f);
      break;
    case 168:
      pressure_kernel<168><<<grid, block, 0, pressure_stream>>>(d_stop, d_sink, 0.5f);
      break;
    case 240:
      pressure_kernel<240><<<grid, block, 0, pressure_stream>>>(d_stop, d_sink, 0.5f);
      break;
    default:
      TORCH_CHECK(false, "nreg must be one of 32, 96, 168, 240");
  }
  CUDA_OK(cudaGetLastError());
  active = true;
}

void stop() {
  if (!active) return;
  *h_stop = 1;
  __sync_synchronize();
  CUDA_OK(cudaStreamSynchronize(pressure_stream));
  active = false;
}

bool running() { return active; }

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("start", &start, "Launch pressure kernel (nreg, blocks_per_sm, threads)");
  m.def("stop", &stop, "Stop pressure kernel and synchronize");
  m.def("running", &running, "Whether the pressure kernel is active");
  m.def("num_sms", &num_sms, "SM count of device 0");
  m.def("reg_info", &reg_info,
        "Per-level (nreg, regs/thread, smem, local) from cudaFuncGetAttributes");
}
