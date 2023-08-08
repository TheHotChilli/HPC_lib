#ifndef HPC_UTILS_BUFFER_HPP
#define HPC_UTILS_BUFFER_HPP

#include <cstddef>
#include <algorithm>
#include <cassert>
#include <new>

namespace hpc { namespace utils { 


template<typename T>
struct Buffer {

    // Variables
    const std::size_t n;                // number of buffer elements
    const std::align_val_t alignment;   // alignment in byte
    T* const ptr;                       // pointer to buffer in memory

    // Constructor
    template<typename... Args>
    Buffer(std::size_t n, std::size_t alignment = alignof(T), Args... args) : 
        n(n), 
        // init alignment with max of the alignment requirements of type T and the provided alignment value 
        alignment(std::align_val_t(std::max(alignof(T), alignment))),
        // allocate memory for n objects of type T and init ptr with Pointer of type T* to it (or nullptr)
	    ptr(n > 0 ?
	        static_cast<T*>(operator new(sizeof(T) * n, this->alignment))
	        :  
	        nullptr)       
    {
        // construct n objects of type T within the allocated memory
        for (std::size_t i = 0; i < n; ++i) {
	        new (ptr + i) T(args...);
        }
    }

    // default Move Constructor 
    Buffer(Buffer&&) = default;

    // delete Copy Constructor -> disable copy construction
    Buffer(const Buffer&) = delete;
    
    // delete Copy Assignment Operator -> disable copy assignment
    Buffer& operator=(const Buffer&) = delete;

    // Destructor
   ~Buffer() 
   {
        for (std::size_t i = 0; i < n; ++i) {
	        ptr[i].~T();
        }
      operator delete(ptr, sizeof(T) * n, alignment);
   }

    // operators
    T& operator[](std::size_t i) {
        assert(i < n);
        return ptr[i];
    }
    const T& operator[](std::size_t i) const {
        assert(i < n);
        return ptr[i];
    }
    T* data() {
        return ptr;
    }
    const T* data() const {
        return ptr;
    }
};


} } // end namespace hpc, utils

#endif // end HPC_UTILS_BUFFER_HPP