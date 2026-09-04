//////////////////////////////////////////////////////////////////////////////
// Copyright (c) Lawrence Livermore National Security, LLC and other CARE
// contributors. See the CARE LICENSE and COPYRIGHT files for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////

#ifndef CARE_SEGMENTED_UNIQUE_H
#define CARE_SEGMENTED_UNIQUE_H

#include "care/host_device_ptr.h"

#if defined(__CUDACC__)
#include "care/cuda/segmented_unique.h"
#elif defined(__HIPCC__)
#include "care/hip/segmented_unique.h"
#else
#include "care/host/segmented_unique.h"
#endif

namespace care {

/**
 * @brief Copy the unique values from each sorted segment into a compact array.
 *
 * Equal values in different segments remain distinct. Empty segments are
 * preserved in @p uniqueOffsets.
 *
 * @param keys Sorted keys containing all input segments.
 * @param offsets Input segment boundaries. For N segments, offsets must
 * contain N + 1 entries: offsets[i] begins segment i, and offsets[N] marks
 * the end of the final segment. The entries must form a nondecreasing
 * sequence from 0 to keys.size(). Repeated entries denote empty segments.
 * @param uniqueKeys Compact output containing the unique keys from every
 * segment. This array must not alias @p keys or @p offsets.
 * @param uniqueOffsets Output segment boundaries into @p uniqueKeys. It has
 * the same number of entries as @p offsets and must not alias an input.
 */
template <typename KeyT, typename OffsetT>
CARE_INLINE void segmented_unique(
   care::host_device_ptr<KeyT> const& keys,
   care::host_device_ptr<OffsetT> const& offsets,
   care::host_device_ptr<KeyT>& uniqueKeys,
   care::host_device_ptr<OffsetT>& uniqueOffsets)
{
#if defined(__CUDACC__)
   care::cuda::segmented_unique(keys, offsets, uniqueKeys, uniqueOffsets);
#elif defined(__HIPCC__)
   care::hip::segmented_unique(keys, offsets, uniqueKeys, uniqueOffsets);
#else
   care::host::segmented_unique(keys, offsets, uniqueKeys, uniqueOffsets);
#endif
}

/**
 * @brief Replace sorted keys and their segment offsets with compact unique
 * values and the corresponding new segment boundaries.
 *
 * This overload replaces both allocations, so slices are detached from their
 * original backing allocations. Use the four-argument overload when the input
 * handles must remain unchanged.
 *
 * @param keys Sorted keys containing all input segments. Replaced by the
 * compact unique keys.
 * @param offsets Segment boundaries into @p keys. Replaced by boundaries into
 * the compact result.
 */
template <typename KeyT, typename OffsetT>
CARE_INLINE void segmented_unique(
   care::host_device_ptr<KeyT>& keys,
   care::host_device_ptr<OffsetT>& offsets)
{
   care::host_device_ptr<KeyT> uniqueKeys;
   care::host_device_ptr<OffsetT> uniqueOffsets;

   segmented_unique(
      static_cast<care::host_device_ptr<KeyT> const&>(keys),
      static_cast<care::host_device_ptr<OffsetT> const&>(offsets),
      uniqueKeys,
      uniqueOffsets);

   keys.free();
   offsets.free();
   keys = uniqueKeys;
   offsets = uniqueOffsets;
}

} // namespace care

#endif // CARE_SEGMENTED_UNIQUE_H
