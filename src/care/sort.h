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

} // namespace care

#endif // CARE_SORT_H
