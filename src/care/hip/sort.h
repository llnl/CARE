//////////////////////////////////////////////////////////////////////////////
// Copyright (c) Lawrence Livermore National Security, LLC and other CARE
// contributors. See the CARE LICENSE and COPYRIGHT files for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////

#ifndef CARE_HIP_SORT_H
#define CARE_HIP_SORT_H

#include "care/CHAIDataGetter.h"
#include "care/DefaultMacros.h"
#include "care/host_device_ptr.h"

#include <cstddef>
#include <utility>

#include "rocprim/rocprim.hpp"

namespace care::hip {

/**
 * @brief Sort keys in ascending order.
 * @param keys Keys to sort in place.
 */
template <typename KeyT>
CARE_INLINE void sort(care::host_device_ptr<KeyT>& keys)
{
   const size_t numItems = keys.size();

   if (numItems == 0) {
      return;
   }

   CHAIDataGetter<KeyT, RAJADeviceExec> keyGetter {};
   auto* rawKeys = keyGetter.getRawArrayData(keys);

   care::host_device_ptr<KeyT> result(numItems);
   auto* rawResult = keyGetter.getRawArrayData(result);

   size_t tempStorageBytes = 0;
   rocprim::radix_sort_keys(nullptr, tempStorageBytes,
                            rawKeys, rawResult, numItems);

   CHAIDataGetter<char, RAJADeviceExec> charGetter {};
   care::host_device_ptr<char> tempStorage(tempStorageBytes);
   auto* rawTempStorage = charGetter.getRawArrayData(tempStorage);
   rocprim::radix_sort_keys(rawTempStorage, tempStorageBytes,
                            rawKeys, rawResult, numItems);

   tempStorage.free();

   if (keys.isSlice()) {
      care::host_device_ptr<const KeyT> source = result;

      CARE_STREAM_LOOP(i, 0, numItems) {
         keys[i] = source[i];
      } CARE_STREAM_LOOP_END

      result.free();
   } else {
      keys.free();
      keys = std::move(result);
   }
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
   const size_t numItems = keys.size();

   if (numItems == 0) {
      return;
   }

   CHAIDataGetter<KeyT, RAJADeviceExec> keyGetter {};
   auto* rawKeys = keyGetter.getRawArrayData(keys);

   CHAIDataGetter<ValueT, RAJADeviceExec> valueGetter {};
   auto* rawValues = valueGetter.getRawArrayData(values);

   care::host_device_ptr<KeyT> keyResult(numItems);
   auto* rawKeyResult = keyGetter.getRawArrayData(keyResult);

   care::host_device_ptr<ValueT> valueResult(numItems);
   auto* rawValueResult = valueGetter.getRawArrayData(valueResult);

   size_t tempStorageBytes = 0;
   rocprim::radix_sort_pairs(nullptr, tempStorageBytes,
                             rawKeys, rawKeyResult,
                             rawValues, rawValueResult, numItems);

   CHAIDataGetter<char, RAJADeviceExec> charGetter {};
   care::host_device_ptr<char> tempStorage(tempStorageBytes);
   auto* rawTempStorage = charGetter.getRawArrayData(tempStorage);
   rocprim::radix_sort_pairs(rawTempStorage, tempStorageBytes,
                             rawKeys, rawKeyResult,
                             rawValues, rawValueResult, numItems);

   tempStorage.free();

   if (keys.isSlice() || values.isSlice()) {
      care::host_device_ptr<const KeyT> keySource = keyResult;
      care::host_device_ptr<const ValueT> valueSource = valueResult;

      CARE_STREAM_LOOP(i, 0, numItems) {
         keys[i] = keySource[i];
         values[i] = valueSource[i];
      } CARE_STREAM_LOOP_END

      keyResult.free();
      valueResult.free();
   } else {
      keys.free();
      values.free();
      keys = std::move(keyResult);
      values = std::move(valueResult);
   }
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
   const size_t numSegments = offsets.size() > 0 ? offsets.size() - 1 : 0;
   const size_t numItems = keys.size();

   if (numSegments == 0 || numItems == 0) {
      return;
   }

   CHAIDataGetter<KeyT, RAJADeviceExec> keyGetter {};
   auto* rawKeys = keyGetter.getRawArrayData(keys);

   CHAIDataGetter<OffsetT, RAJADeviceExec> offsetGetter {};
   const auto* rawOffsets = offsetGetter.getConstRawArrayData(offsets);

   care::host_device_ptr<KeyT> result(numItems);
   auto* rawResult = keyGetter.getRawArrayData(result);

   size_t tempStorageBytes = 0;
   rocprim::segmented_radix_sort_keys(nullptr, tempStorageBytes,
                                      rawKeys, rawResult, numItems, numSegments,
                                      rawOffsets, rawOffsets + 1);

   CHAIDataGetter<char, RAJADeviceExec> charGetter {};
   care::host_device_ptr<char> tempStorage(tempStorageBytes);
   auto* rawTempStorage = charGetter.getRawArrayData(tempStorage);
   rocprim::segmented_radix_sort_keys(rawTempStorage, tempStorageBytes,
                                      rawKeys, rawResult, numItems, numSegments,
                                      rawOffsets, rawOffsets + 1);

   tempStorage.free();

   if (keys.isSlice()) {
      care::host_device_ptr<const KeyT> source = result;

      CARE_STREAM_LOOP(i, 0, numItems) {
         keys[i] = source[i];
      } CARE_STREAM_LOOP_END

      result.free();
   } else {
      keys.free();
      keys = std::move(result);
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
CARE_INLINE void segmented_paired_sort(
   care::host_device_ptr<KeyT>& keys,
   care::host_device_ptr<ValueT>& values,
   care::host_device_ptr<OffsetT> const& offsets)
{
   const size_t numSegments = offsets.size() > 0 ? offsets.size() - 1 : 0;
   const size_t numItems = keys.size();

   if (numSegments == 0 || numItems == 0) {
      return;
   }

   CHAIDataGetter<KeyT, RAJADeviceExec> keyGetter {};
   auto* rawKeys = keyGetter.getRawArrayData(keys);

   CHAIDataGetter<ValueT, RAJADeviceExec> valueGetter {};
   auto* rawValues = valueGetter.getRawArrayData(values);

   CHAIDataGetter<OffsetT, RAJADeviceExec> offsetGetter {};
   const auto* rawOffsets = offsetGetter.getConstRawArrayData(offsets);

   care::host_device_ptr<KeyT> keyResult(numItems);
   auto* rawKeyResult = keyGetter.getRawArrayData(keyResult);

   care::host_device_ptr<ValueT> valueResult(numItems);
   auto* rawValueResult = valueGetter.getRawArrayData(valueResult);

   size_t tempStorageBytes = 0;
   rocprim::segmented_radix_sort_pairs(
      nullptr, tempStorageBytes, rawKeys, rawKeyResult,
      rawValues, rawValueResult, numItems, numSegments,
      rawOffsets, rawOffsets + 1);

   CHAIDataGetter<char, RAJADeviceExec> charGetter {};
   care::host_device_ptr<char> tempStorage(tempStorageBytes);
   auto* rawTempStorage = charGetter.getRawArrayData(tempStorage);
   rocprim::segmented_radix_sort_pairs(
      rawTempStorage, tempStorageBytes, rawKeys, rawKeyResult,
      rawValues, rawValueResult, numItems, numSegments,
      rawOffsets, rawOffsets + 1);

   tempStorage.free();

   if (keys.isSlice() || values.isSlice()) {
      care::host_device_ptr<const KeyT> keySource = keyResult;
      care::host_device_ptr<const ValueT> valueSource = valueResult;

      CARE_STREAM_LOOP(i, 0, numItems) {
         keys[i] = keySource[i];
         values[i] = valueSource[i];
      } CARE_STREAM_LOOP_END

      keyResult.free();
      valueResult.free();
   } else {
      keys.free();
      values.free();
      keys = std::move(keyResult);
      values = std::move(valueResult);
   }
}

} // namespace care::hip

#endif // CARE_HIP_SORT_H
