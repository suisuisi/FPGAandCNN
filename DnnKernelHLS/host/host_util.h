#ifndef DNNKERNEL_HOST_UTIL_H
#define DNNKERNEL_HOST_UTIL_H

#include <cstdlib>
#include <vector>
#include <string>
#include <fstream>
#include <CL/cl2.hpp>
#include <chrono>

namespace dnnk {

class ClHelper {
public:
    ClHelper(const std::string& xclbin_name) {

        cl::Platform::get(&platforms_);
        for (std::size_t i = 0; i < platforms_.size(); i++) {
            cl::Platform& platform = platforms_[i];
            std::string platform_name = platform.getInfo<CL_PLATFORM_NAME>();

            if (platform_name == "Xilinx") {
                platform.getDevices(CL_DEVICE_TYPE_ACCELERATOR, &devices_);
                break;
            }
        }

        cl::Device device = devices_[0];

        context_ = cl::Context(device);

        auto xclbin = read_binary_file(xclbin_name);
        cl::Program::Binaries binaries;
        binaries.push_back(xclbin);

        program_ = cl::Program(context_, devices_, binaries);
    }

    cl::Program& get_program() {
        return program_;
    }

    cl::Context& get_context() {
        return context_;
    }

    cl::Device& get_device() {
        return devices_[0];
    }

private:
    std::vector<unsigned char> read_binary_file(const std::string& filename) {
        std::vector<unsigned char> ret;
        std::ifstream ifs(filename, std::ifstream::binary);

        ifs.seekg(0, ifs.end);
        std::size_t size = ifs.tellg();
        ifs.seekg(0, ifs.beg);

        ret.resize(size);
        ifs.read(reinterpret_cast<char*>(ret.data()), ret.size());

        return ret;
    }

    std::vector<cl::Platform> platforms_;
    std::vector<cl::Device> devices_;
    cl::Context context_;
    cl::Program program_;
};


template <typename T>
class aligned_allocator {
public:
  using value_type = T;

  aligned_allocator() = default;

  template <class U>
  constexpr aligned_allocator(const aligned_allocator<U>&) noexcept {}

  T* allocate(std::size_t size) {
    void* ptr = nullptr;

    if (posix_memalign(&ptr, 4096, size * sizeof(T))) {
      throw std::bad_alloc();
    }

    return reinterpret_cast<T*>(ptr);
  }

  void deallocate(T* ptr, std::size_t size) {
    free(ptr);
  }
};

template <typename T>
using aligned_vector = std::vector<T, aligned_allocator<T>>;


class StopWatch {
public:
  StopWatch() = default;

  void start() {
    tstart_ = clock::now();
  }

  void stop() {
    tstop_ = clock::now();
  }

  double elapsed_time_ms() const {
    auto elapsed_micro = std::chrono::duration_cast<std::chrono::microseconds>(tstop_ - tstart_).count();
    return elapsed_micro / 1000.0;
  }

private:
  using clock = std::chrono::high_resolution_clock;
  using time_point = std::chrono::time_point<clock>;

  time_point tstart_;
  time_point tstop_;
};

}

#endif
