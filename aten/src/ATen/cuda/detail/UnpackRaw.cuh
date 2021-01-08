// In-kernel call to retrieve philox seed and offset from a  PhiloxCudaState instance whether
// that instance was created with graph capture underway or not.
// See Note [CUDA Graph-safe RNG states].
//
// We can't write a __device__ function in CUDAGeneratorImpl.h, because it's in ATen.
// Also, whatever call unpacks PhiloxCudaState in consumer kernels must be inlineable.
// Easiest thing that comes to mind is, define a free function here, in ATen/cuda.
// Any cuda consumer can include this header.
//
// The raw definition lives in its own file so jit codegen can easily copy it.
namespace at {
namespace cuda {
namespace philox {

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
