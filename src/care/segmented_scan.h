//////////////////////////////////////////////////////////////////////////////
// Copyright (c) Lawrence Livermore National Security, LLC and other CARE
// contributors. See the CARE LICENSE and COPYRIGHT files for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////

#ifndef CARE_SEGMENTED_SCAN_H
#define CARE_SEGMENTED_SCAN_H

#include "care/host_device_ptr.h"

#if defined(__CUDACC__)
#include "care/cuda/segmented_scan.h"
#elif defined(__HIPCC__)
#include "care/hip/segmented_scan.h"
#else
#include "care/host/segmented_scan.h"
#endif

namespace care {

/**
 * @brief Perform an in-place exclusive sum independently within each segment.
 * @param values Values to scan and replace with the exclusive sums.
 * @param offsets Segment boundaries; segment i is [offsets[i], offsets[i + 1]).
 * @param initialValue Initial value assigned to the first item of each segment.
 */
template <typename ValueT, typename OffsetT>
CARE_INLINE void segmented_exclusive_scan(
   care::host_device_ptr<ValueT>& values,
   care::host_device_ptr<OffsetT> const& offsets,
   ValueT initialValue)
{
#if defined(__CUDACC__)
   care::cuda::segmented_exclusive_scan(values, offsets, initialValue);
#elif defined(__HIPCC__)
   care::hip::segmented_exclusive_scan(values, offsets, initialValue);
#else
   care::host::segmented_exclusive_scan(values, offsets, initialValue);
#endif
}

} // namespace care

#endif // CARE_SEGMENTED_SCAN_H
