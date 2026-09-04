//////////////////////////////////////////////////////////////////////////////
// Copyright (c) Lawrence Livermore National Security, LLC and other CARE
// contributors. See the CARE LICENSE and COPYRIGHT files for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////

#ifndef CARE_HOST_SEGMENTED_UNIQUE_H
#define CARE_HOST_SEGMENTED_UNIQUE_H

#include "care/host_device_ptr.h"

#include <cstddef>
#include <utility>

namespace care::host {

/**
 * @brief Copy the unique values from each sorted segment into a compact array.
 */
template <typename KeyT, typename OffsetT>
void segmented_unique(
   care::host_device_ptr<KeyT> const& keys,
   care::host_device_ptr<OffsetT> const& offsets,
   care::host_device_ptr<KeyT>& uniqueKeys,
   care::host_device_ptr<OffsetT>& uniqueOffsets)
{
   const size_t numSegments = offsets.size() > 0 ? offsets.size() - 1 : 0;
   const KeyT* rawKeys = keys.cdata();
   const OffsetT* rawOffsets = offsets.cdata();

   care::host_device_ptr<OffsetT> resultOffsets(offsets.size());
   OffsetT* rawResultOffsets = resultOffsets.data();

   size_t numUnique = 0;
   for (size_t segment = 0; segment < numSegments; ++segment) {
      rawResultOffsets[segment] = static_cast<OffsetT>(numUnique);

      const size_t begin = static_cast<size_t>(rawOffsets[segment]);
      const size_t end = static_cast<size_t>(rawOffsets[segment + 1]);
      for (size_t i = begin; i < end; ++i) {
         if (i == begin || rawKeys[i - 1] < rawKeys[i] || rawKeys[i] < rawKeys[i - 1]) {
            ++numUnique;
         }
      }
   }

   if (offsets.size() > 0) {
      rawResultOffsets[numSegments] = static_cast<OffsetT>(numUnique);
   }

   care::host_device_ptr<KeyT> result(numUnique);
   KeyT* rawResult = result.data();

   size_t output = 0;
   for (size_t segment = 0; segment < numSegments; ++segment) {
      const size_t begin = static_cast<size_t>(rawOffsets[segment]);
      const size_t end = static_cast<size_t>(rawOffsets[segment + 1]);
      for (size_t i = begin; i < end; ++i) {
         if (i == begin || rawKeys[i - 1] < rawKeys[i] || rawKeys[i] < rawKeys[i - 1]) {
            rawResult[output++] = rawKeys[i];
         }
      }
   }

   uniqueKeys.free();
   uniqueOffsets.free();
   uniqueKeys = std::move(result);
   uniqueOffsets = std::move(resultOffsets);
}

} // namespace care::host

#endif // CARE_HOST_SEGMENTED_UNIQUE_H
