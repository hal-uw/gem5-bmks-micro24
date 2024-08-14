#pragma once

#include <vector>
#include <numeric>
#include <memory>
#include <cstdlib>

// Includes for mmap stuff
#include <cassert>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <hip/hip_runtime_api.h>

//#define DGPU

bool g_tensor_use_mmap { false };
const char *g_tensor_mmap_file;

template <typename T>
class Tensor {
    std::vector<int> dims_;
    int size_;

    struct deleteDevPtr {
        void operator()(T *p) const {
#ifdef DGPU
            hipFree(p);
#else
            free(p);
#endif
        }
    };


public:
    std::shared_ptr<T> ptr_;

    Tensor() {}

    Tensor(std::vector<int> dims) : dims_(dims) {
        T* tmp_ptr;
        size_ = std::accumulate(dims_.begin(), dims_.end(), 1, std::multiplies<int>());
#ifdef DGPU
        hipMalloc(&tmp_ptr, sizeof(T) * size_);
#else
        tmp_ptr = (T*)malloc(sizeof(T) * size_);
#endif

        ptr_.reset(tmp_ptr, deleteDevPtr());
    }

    T* begin() const { return ptr_.get(); }
    T* end()   const { return ptr_.get() + size_; }
    int size() const { return size_; }
    std::vector<int> dims() const { return dims_; }
};

template<typename T>
Tensor<T> fill(std::vector<int> dims, T val) {
    Tensor<T> tensor(dims);
#ifdef DGPU
    std::vector<T> host_ptr(tensor.size());
    std::fill(host_ptr.begin(), host_ptr.end(), val);
    hipMemcpy(tensor.ptr_.get(), host_ptr.data(), tensor.size()*sizeof(T), hipMemcpyHostToDevice);
#else
    std::fill(tensor.begin(), tensor.end(), val);
#endif
    return tensor;
}

template<typename T>
Tensor<T> zeros(std::vector<int> dims)
{
    Tensor<T> tensor(dims);
#ifdef DGPU
    hipMemset(tensor.ptr_.get(), 0, d*sizeof(T));
#else
    memset(tensor.ptr_.get(), 0, tensor.size()*sizeof(T));
#endif
    return tensor;
}

template<typename T>
Tensor<T> rand(std::vector<int> dims)
{
    Tensor<T> tensor(dims);
#ifdef DGPU
    std::vector<T> host_ptr(tensor.size());
    std::srand(std::time(0));
    for(int i=0;i<tensor.size();i++)
    {
      host_ptr[i] = std::rand();
    }
    hipMemcpy(tensor.ptr_.get(), host_ptr.data(), tensor.size()*sizeof(T), hipMemcpyHostToDevice);
#else
    if (g_tensor_use_mmap) {
        T *mmap_ptr;
        int fd = open(g_tensor_mmap_file, O_RDONLY);
        assert (tensor.size()*sizeof(T) <= lseek(fd, 0, SEEK_END));

        mmap_ptr = (T *)mmap(NULL, tensor.size()*sizeof(T), PROT_READ, MAP_SHARED, fd, 0);
        close(fd);

        memcpy(tensor.ptr_.get(), mmap_ptr, tensor.size()*sizeof(T));
        munmap(mmap_ptr, tensor.size()*sizeof(T));
    } else {
        for(int i=0;i<tensor.size();i++)
        {
            tensor.begin()[i] = std::rand();
        }
    }
#endif
    return tensor;
}
