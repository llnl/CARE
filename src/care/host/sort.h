//////////////////////////////////////////////////////////////////////////////
// Copyright (c) Lawrence Livermore National Security, LLC and other CARE
// contributors. See the CARE LICENSE and COPYRIGHT files for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////

#ifndef CARE_HOST_SORT_H
#define CARE_HOST_SORT_H

#include "care/host_device_ptr.h"

#include <algorithm>
#include <cstddef>

namespace care::host {

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

} // namespace care::host

#endif // CARE_HOST_SORT_H
