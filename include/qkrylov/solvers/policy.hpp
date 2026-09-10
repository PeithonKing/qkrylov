#pragma once

#include <type_traits>

namespace qkrylov {
namespace solvers {
namespace policy {

struct Default {};
struct SinglePass : Default {};
struct TwoPass {};

template <typename T>
struct is_policy : std::false_type {};

template <>
struct is_policy<Default> : std::true_type {};

template <>
struct is_policy<SinglePass> : std::true_type {};

template <>
struct is_policy<TwoPass> : std::true_type {};

template <typename T>
inline constexpr bool is_policy_v = is_policy<T>::value;

} // namespace policy
} // namespace solvers
} // namespace qkrylov
