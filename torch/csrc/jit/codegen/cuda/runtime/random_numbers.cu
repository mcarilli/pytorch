class Philox {
 public:
  __device__ Philox(
      unsigned long long seed,
      unsigned long long subsequence,
      unsigned long long offset) {
    key.x = (unsigned int)seed;
    key.y = (unsigned int)(seed >> 32);
    counter = make_uint4(0, 0, 0, 0);
    counter.z = (unsigned int)(subsequence);
    counter.w = (unsigned int)(subsequence >> 32);
    STATE = 0;
    incr_n(offset / 4);
  }

  __device__ unsigned long operator()() {
    if (STATE == 0) {
      uint4 counter_ = counter;
      uint2 key_ = key;
      for (int i = 0; i < 9; i++) {
        counter_ = single_round(counter_, key_);
        key_.x += (kPhilox10A);
        key_.y += (kPhilox10B);
      }
      output = single_round(counter_, key_);
      incr();
    }
    unsigned long ret = 0;
    switch (STATE) {
      case 0:
        ret = output.x;
        break;
      case 1:
        ret = output.y;
        break;
      case 2:
        ret = output.z;
        break;
      case 3:
        ret = output.w;
        break;
    }
    STATE = (STATE + 1) % 4;
    return ret;
  }

 private:
  __device__ void incr_n(unsigned long long n) {
    unsigned int nlo = (unsigned int)(n);
    unsigned int nhi = (unsigned int)(n >> 32);
    counter.x += nlo;
    if (counter.x < nlo)
      nhi++;
    counter.y += nhi;
    if (nhi <= counter.y)
      return;
    if (++counter.z)
      return;
    ++counter.w;
  }

  __device__ void incr() {
    if (++counter.x)
      return;
    if (++counter.y)
      return;
    if (++counter.z)
      return;
    ++counter.w;
  }

  __device__ unsigned int mulhilo32(
      unsigned int a,
      unsigned int b,
      unsigned int* result_high) {
    *result_high = __umulhi(a, b);
    return a * b;
  }

  __device__ uint4 single_round(uint4 ctr, uint2 key) {
    unsigned int hi0;
    unsigned int hi1;
    unsigned int lo0 = mulhilo32(kPhiloxSA, ctr.x, &hi0);
    unsigned int lo1 = mulhilo32(kPhiloxSB, ctr.z, &hi1);
    uint4 ret = {hi1 ^ ctr.y ^ key.x, lo1, hi0 ^ ctr.w ^ key.y, lo0};
    return ret;
  }

 private:
  static constexpr unsigned long kPhilox10A = 0x9E3779B9;
  static constexpr unsigned long kPhilox10B = 0xBB67AE85;
  static constexpr unsigned long kPhiloxSA = 0xD2511F53;
  static constexpr unsigned long kPhiloxSB = 0xCD9E8D57;

  uint4 counter = {};
  uint4 output = {};
  uint2 key = {};
  unsigned int STATE = 0;
};

__device__ float uniformf(unsigned int x) {
  constexpr float kRanInvM32 = 2.3283064e-10f; // Inverse of 2^32.
  return x * kRanInvM32;
}

__device__ double uniform(unsigned int x, unsigned int y) {
  constexpr double kRan2Pow53Inv = 1.1102230246251565e-16;
  const unsigned long long z =
      (unsigned long long)x ^ ((unsigned long long)y << (53 - 32));
  return z * kRan2Pow53Inv + (kRan2Pow53Inv / 2.0);
}


namespace at {

// WARNING:
// Copy pasted from ATen/CUDAGeneratorImpl.h,
// because we don't want to codegen directly from something in ATen.
// If you change the definition there, you must change the definition here to match.
struct PhiloxCudaState {
  PhiloxCudaState() = default;
  PhiloxCudaState(const PhiloxCudaState&) = default;
  // Called if graph capture is not underway
  PhiloxCudaState(uint64_t seed,
                  uint64_t offset) {
    seed_ = seed;
    offset_.val = offset;
  }
  // Called if graph capture is underway
  PhiloxCudaState(uint64_t seed,
                  int64_t* offset_extragraph,
                  uint32_t offset_intragraph) {
    seed_ = seed;
    offset_.ptr = offset_extragraph;
    offset_intragraph_ = offset_intragraph;
    captured_ = true;
  }

  // Public members, directly accessible by at::cuda::philox::unpack.
  // If we made them private with getters/setters, the getters/setters
  // would have to be __device__, and we can't declare __device__ in ATen.
  union Payload {
    uint64_t val;
    int64_t* ptr;
  };

  uint64_t seed_;
  Payload offset_;
  uint32_t offset_intragraph_;
  bool captured_ = false;
};

namespace cuda {
namespace philox {

// WARNING:
// Copy pasted from ATen/cuda/CudaGraphsUtils.cuh,
// because we don't want to codegen directly from something in ATen.
// If you change the definition there, you must change the definition here to match.
__device__ __forceinline__ std::tuple<uint64_t, uint64_t>
unpack(at::PhiloxCudaState arg) {
  if (arg.captured_) {
    return std::make_tuple(arg.seed_, *(arg.offset_.ptr) + arg.offset_intragraph_);
  } else {
    return std::make_tuple(arg.seed_, arg.offset_.val);
  }
}

} // namespace philox
} // namespace cuda
} // namespace at
