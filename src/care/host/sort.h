//////////////////////////////////////////////////////////////////////////////
// Copyright (c) Lawrence Livermore National Security, LLC and other CARE
// contributors. See the CARE LICENSE and COPYRIGHT files for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////

#ifndef CARE_HOST_SORT_H
#define CARE_HOST_SORT_H

#include "care/host_device_ptr.h"
#include "care/zip_iterator.h"

#include <algorithm>
#include <cstddef>

namespace care::host {

/**
 * @brief Sort keys in ascending order.
 * @param keys Keys to sort in place.
 */
template <typename KeyT>
void sort(care::host_device_ptr<KeyT>& keys)
{
   if (keys.size() == 0) {
      return;
   }

   std::sort(keys.data(), keys.data() + keys.size());
}

/**
 * @brief Sort keys in ascending order and apply the same permutation to values.
 * @param keys Keys to sort in place.
 * @param values Values associated one-to-one with keys. Must have the same
 * number of entries as keys.
 */
template <typename KeyT, typename ValueT>
void paired_sort(care::host_device_ptr<KeyT>& keys,
                 care::host_device_ptr<ValueT>& values)
{
   auto first = care::zip_iterator<KeyT, ValueT>(keys.data(), values.data());
   std::sort(first, first + keys.size(),
             [](auto const& left, auto const& right) {
                return left.key < right.key;
             });
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
void segmented_sort(care::host_device_ptr<KeyT>& keys,
                    care::host_device_ptr<OffsetT> const& offsets)
{
   const size_t numSegments = offsets.size() > 0 ? offsets.size() - 1 : 0;

   KeyT* rawKeys = keys.data();
   const OffsetT* rawOffsets = offsets.cdata();

   for (size_t segment = 0; segment < numSegments; ++segment) {
      const size_t begin = static_cast<size_t>(rawOffsets[segment]);
      const size_t end = static_cast<size_t>(rawOffsets[segment + 1]);
      std::sort(rawKeys + begin, rawKeys + end);
   }
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
void segmented_paired_sort(care::host_device_ptr<KeyT>& keys,
                           care::host_device_ptr<ValueT>& values,
                           care::host_device_ptr<OffsetT> const& offsets)
{
   const size_t numSegments = offsets.size() > 0 ? offsets.size() - 1 : 0;

   const OffsetT* rawOffsets = offsets.cdata();
   auto first = care::zip_iterator<KeyT, ValueT>(keys.data(), values.data());

   for (size_t segment = 0; segment < numSegments; ++segment) {
      const size_t begin = static_cast<size_t>(rawOffsets[segment]);
      const size_t end = static_cast<size_t>(rawOffsets[segment + 1]);
      std::sort(first + begin, first + end,
                [](auto const& left, auto const& right) {
                   return left.key < right.key;
                });
   }
}

} // namespace care::host

#endif // CARE_HOST_SORT_H
