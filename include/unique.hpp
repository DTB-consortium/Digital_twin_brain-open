#pragma once

#include <memory>

namespace dtb {

// C++11 doesn't have constexpr std::move / std::forward.
// Implementation taken from libc++.
template<class T>
constexpr inline std::remove_reference<T>&& move(T&& t) noexcept {
  return static_cast<std::remove_reference<T>&&>(t);
}

template <class T>
constexpr inline T&& forward(std::remove_reference<T>& t) noexcept {
    return static_cast<T&&>(t);
}
template <class T>
constexpr inline T&& forward(std::remove_reference<T>&& t) noexcept {
    static_assert(!std::is_lvalue_reference<T>::value,
                  "can not forward an rvalue as an lvalue.");
    return static_cast<T&&>(t);
}


#if __cplusplus >= 201402L 
using std::make_unique;
#else
// Implementation taken from folly
template <typename T, typename... Args>
typename std::enable_if<!std::is_array<T>::value, std::unique_ptr<T>>::type
make_unique(Args&&... args) {
  return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
}
// Allows 'make_unique<T[]>(10)'. (N3690 s20.9.1.4 p3-4)
template <typename T>
typename std::enable_if<std::is_array<T>::value, std::unique_ptr<T>>::type
make_unique(const size_t n) {
  return std::unique_ptr<T>(new typename std::remove_extent<T>::type[n]());
}
// Disallows 'make_unique<T[10]>()'. (N3690 s20.9.1.4 p5)
template <typename T, typename... Args>
typename std::enable_if<std::extent<T>::value != 0, std::unique_ptr<T>>::type
make_unique(Args&&...) = delete;
#endif

}

