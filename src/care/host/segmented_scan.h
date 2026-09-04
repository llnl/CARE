//////////////////////////////////////////////////////////////////////////////
// Copyright (c) Lawrence Livermore National Security, LLC and other CARE
// contributors. See the CARE LICENSE and COPYRIGHT files for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////

#ifndef CARE_HOST_SEGMENTED_SCAN_H
#define CARE_HOST_SEGMENTED_SCAN_H

#include "care/host_device_ptr.h"

#include <cstddef>
#include <functional>
#include <numeric>

namespace care::host {

/**
 * @brief Perform an in-place exclusive scan independently within each segment.
 * @param values Values to scan and replace with the exclusive scan results.
 * @param offsets Segment boundaries; segment i is [offsets[i], offsets[i + 1]).
 * @param initialValue Initial value assigned to the first item of each segment.
 * @param binaryOp Associative binary operation used to perform the scan.
 */
template <typename ValueT, typename OffsetT, typename BinaryOp>
void segmented_exclusive_scan(care::host_device_ptr<ValueT>& values,
                              care::host_device_ptr<OffsetT> const& offsets,
                              ValueT initialValue,
                              BinaryOp binaryOp)
{
   const size_t numSegments = offsets.size() > 0 ? offsets.size() - 1 : 0;
   ValueT* rawValues = values.data();
   const OffsetT* rawOffsets = offsets.cdata();

   for (size_t segment = 0; segment < numSegments; ++segment) {
      const size_t begin = static_cast<size_t>(rawOffsets[segment]);
      const size_t end = static_cast<size_t>(rawOffsets[segment + 1]);
      std::exclusive_scan(rawValues + begin, rawValues + end,
                          rawValues + begin, initialValue, binaryOp);
   }
}

/**
 * @brief Perform an in-place exclusive sum independently within each segment.
 * @param values Values to scan and replace with the exclusive sums.
 * @param offsets Segment boundaries; segment i is [offsets[i], offsets[i + 1]).
 * @param initialValue Initial value assigned to the first item of each segment.
 */
template <typename ValueT, typename OffsetT>
void segmented_exclusive_scan(care::host_device_ptr<ValueT>& values,
                              care::host_device_ptr<OffsetT> const& offsets,
                              ValueT initialValue)
{
   segmented_exclusive_scan(values, offsets, initialValue,
                            std::plus<ValueT> {});
}

/**
 * @brief Perform an in-place exclusive sum with an initial value of zero
 * independently within each segment.
 * @param values Values to scan and replace with the exclusive sums.
 * @param offsets Segment boundaries; segment i is [offsets[i], offsets[i + 1]).
 */
template <typename ValueT, typename OffsetT>
void segmented_exclusive_scan(care::host_device_ptr<ValueT>& values,
                              care::host_device_ptr<OffsetT> const& offsets)
{
   segmented_exclusive_scan(values, offsets, ValueT {});
}

} // namespace care::host

#endif // CARE_HOST_SEGMENTED_SCAN_H
