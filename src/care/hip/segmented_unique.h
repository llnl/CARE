//////////////////////////////////////////////////////////////////////////////
// Copyright (c) Lawrence Livermore National Security, LLC and other CARE
// contributors. See the CARE LICENSE and COPYRIGHT files for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////

#ifndef CARE_HIP_SEGMENTED_UNIQUE_H
#define CARE_HIP_SEGMENTED_UNIQUE_H

#include "care/detail/segmented_unique.h"

namespace care::hip {

template <typename KeyT, typename OffsetT>
CARE_INLINE void segmented_unique(
   care::host_device_ptr<KeyT> const& keys,
   care::host_device_ptr<OffsetT> const& offsets,
   care::host_device_ptr<KeyT>& uniqueKeys,
   care::host_device_ptr<OffsetT>& uniqueOffsets)
{
   care::detail::segmented_unique_device(
      keys, offsets, uniqueKeys, uniqueOffsets);
}

} // namespace care::hip

#endif // CARE_HIP_SEGMENTED_UNIQUE_H
