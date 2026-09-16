//////////////////////////////////////////////////////////////////////////////
// Copyright (c) Lawrence Livermore National Security, LLC and other CARE
// contributors. See the CARE LICENSE and COPYRIGHT files for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////

#ifndef CARE_SORT_H
#define CARE_SORT_H

#include "care/host_device_ptr.h"

#if defined(__CUDACC__)
#include "care/cuda/sort.h"
#elif defined(__HIPCC__)
#include "care/hip/sort.h"
#else
#include "care/host/sort.h"
#endif

namespace care {

/**
 * @brief Sort keys in ascending order.
 * @param keys Keys to sort in place.
 */
template <typename KeyT>
CARE_INLINE void sort(care::host_device_ptr<KeyT>& keys)
{
#if defined(__CUDACC__)
   care::cuda::sort(keys);
#elif defined(__HIPCC__)
   care::hip::sort(keys);
#else
   care::host::sort(keys);
#endif
}

/**
 * @brief Sort keys in ascending order and apply the same permutation to values.
 * @param keys Keys to sort in place.
 * @param values Values associated one-to-one with keys. Must have the same
 * number of entries as keys.
 */
template <typename KeyT, typename ValueT>
CARE_INLINE void paired_sort(care::host_device_ptr<KeyT>& keys,
                             care::host_device_ptr<ValueT>& values)
{
#if defined(__CUDACC__)
   care::cuda::paired_sort(keys, values);
#elif defined(__HIPCC__)
   care::hip::paired_sort(keys, values);
#else
   care::host::paired_sort(keys, values);
#endif
}

/**
 * @brief Sort keys in ascending order independently within each segment.
 * @param keys Keys to sort in place.
 * @param offsets Segment boundaries. For N segments, offsets must contain
 * N + 1 entries: offsets[i] begins segment i, and offsets[N] marks the end of
 * the final segment. Segment i is therefore [offsets[i], offsets[i + 1]).
 * The entries must form a nondecreasing sequence from 0 to keys.size(); that
 * is, offsets[0] must be 0 and offsets[N] must equal keys.size(). Repeated
 * entries denote empty segments.
 */
template <typename KeyT, typename OffsetT>
CARE_INLINE void segmented_sort(care::host_device_ptr<KeyT>& keys,
                                care::host_device_ptr<OffsetT> const& offsets)
{
#if defined(__CUDACC__)
   care::cuda::segmented_sort(keys, offsets);
#elif defined(__HIPCC__)
   care::hip::segmented_sort(keys, offsets);
#else
   care::host::segmented_sort(keys, offsets);
#endif
}

/**
 * @brief Sort keys independently within each segment and apply the same
 * permutation to values.
 * @param keys Keys to sort in place.
 * @param values Values associated one-to-one with keys. Must have the same
 * number of entries as keys.
 * @param offsets Segment boundaries. For N segments, offsets must contain
 * N + 1 entries: offsets[i] begins segment i, and offsets[N] marks the end of
 * the final segment. Segment i is therefore [offsets[i], offsets[i + 1]).
 * The entries must form a nondecreasing sequence from 0 to keys.size(); that
 * is, offsets[0] must be 0 and offsets[N] must equal keys.size(). Repeated
 * entries denote empty segments.
 */
template <typename KeyT, typename ValueT, typename OffsetT>
CARE_INLINE void segmented_paired_sort(
   care::host_device_ptr<KeyT>& keys,
   care::host_device_ptr<ValueT>& values,
   care::host_device_ptr<OffsetT> const& offsets)
{
#if defined(__CUDACC__)
   care::cuda::segmented_paired_sort(keys, values, offsets);
#elif defined(__HIPCC__)
   care::hip::segmented_paired_sort(keys, values, offsets);
#else
   care::host::segmented_paired_sort(keys, values, offsets);
#endif
}

} // namespace care

#endif // CARE_SORT_H
