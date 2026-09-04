//////////////////////////////////////////////////////////////////////////////
// Copyright (c) Lawrence Livermore National Security, LLC and other CARE
// contributors. See the CARE LICENSE and COPYRIGHT files for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////

#ifndef CARE_DETAIL_SEGMENTED_UNIQUE_H
#define CARE_DETAIL_SEGMENTED_UNIQUE_H

#include "care/DefaultMacros.h"
#include "care/host_device_ptr.h"
#include "care/scan.h"

#include <cstddef>
#include <utility>

namespace care::detail {

/**
 * @brief Device implementation shared by the CUDA and HIP front ends.
 */
template <typename KeyT, typename OffsetT>
CARE_INLINE void segmented_unique_device(
   care::host_device_ptr<KeyT> const& keys,
   care::host_device_ptr<OffsetT> const& offsets,
   care::host_device_ptr<KeyT>& uniqueKeys,
   care::host_device_ptr<OffsetT>& uniqueOffsets)
{
   const size_t numItems = keys.size();
   const size_t numSegments = offsets.size() > 0 ? offsets.size() - 1 : 0;

   // One extra entry lets the exclusive scan's final value hold the total
   // number of unique keys.
   care::host_device_ptr<int> positions(numItems + 1);

   CARE_STREAM_LOOP(i, 0, numItems + 1) {
      positions[i] = 0;
   } CARE_STREAM_LOOP_END

   // Mark unique values by comparing adjacent keys. Segment starts are fixed
   // in a separate kernel so equal values on opposite sides of a boundary are
   // both retained.
   CARE_STREAM_LOOP(i, 0, numItems) {
      positions[i] = static_cast<int>(
         i == 0 || keys[i - 1] < keys[i] || keys[i] < keys[i - 1]);
   } CARE_STREAM_LOOP_END

   CARE_STREAM_LOOP(segment, 0, numSegments) {
      const OffsetT begin = offsets[segment];
      if (begin < offsets[segment + 1]) {
         positions[begin] = 1;
      }
   } CARE_STREAM_LOOP_END

   care::exclusive_scan(RAJADeviceExec {}, positions, nullptr,
                        static_cast<int>(numItems + 1), 0, true);

   const int numUnique = positions.pick(numItems);
   care::host_device_ptr<KeyT> result(static_cast<size_t>(numUnique));
   care::host_device_ptr<OffsetT> resultOffsets(offsets.size());

   CARE_STREAM_LOOP(i, 0, numItems) {
      if (positions[i] != positions[i + 1]) {
         result[positions[i]] = keys[i];
      }
   } CARE_STREAM_LOOP_END

   CARE_STREAM_LOOP(segment, 0, offsets.size()) {
      resultOffsets[segment] = static_cast<OffsetT>(positions[offsets[segment]]);
   } CARE_STREAM_LOOP_END

   positions.free();

   uniqueKeys.free();
   uniqueOffsets.free();
   uniqueKeys = std::move(result);
   uniqueOffsets = std::move(resultOffsets);
}

} // namespace care::detail

#endif // CARE_DETAIL_SEGMENTED_UNIQUE_H
