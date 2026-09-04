//////////////////////////////////////////////////////////////////////////////
// Copyright (c) Lawrence Livermore National Security, LLC and other CARE
// contributors. See the CARE LICENSE and COPYRIGHT files for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////

#include "care/segmented_unique.h"
#include "care/sort.h"
#include "care/detail/test_utils.h"

#include "gtest/gtest.h"

TEST(segmented_unique, segment_local_and_empty)
{
   care::host_device_ptr<int> keys(10);
   care::host_device_ptr<int> offsets(5);
   care::host_device_ptr<int> uniqueKeys;
   care::host_device_ptr<int> uniqueOffsets;

   const int input[] = {
      1, 1, 2,
      // empty segment
      2, 2, 3, 3,
      3, 3, 4
   };
   const int segmentOffsets[] = {0, 3, 3, 7, 10};
   const int expectedKeys[] = {1, 2, 2, 3, 3, 4};
   const int expectedOffsets[] = {0, 2, 2, 4, 6};

   CARE_SEQUENTIAL_LOOP(i, 0, 10) {
      keys[i] = input[i];
   } CARE_SEQUENTIAL_LOOP_END

   CARE_SEQUENTIAL_LOOP(i, 0, 5) {
      offsets[i] = segmentOffsets[i];
   } CARE_SEQUENTIAL_LOOP_END

   care::segmented_unique(keys, offsets, uniqueKeys, uniqueOffsets);

   ASSERT_EQ(uniqueKeys.size(), 6);
   ASSERT_EQ(uniqueOffsets.size(), 5);
   CARE_SEQUENTIAL_LOOP(i, 0, 6) {
      EXPECT_EQ(uniqueKeys[i], expectedKeys[i]);
   } CARE_SEQUENTIAL_LOOP_END
   CARE_SEQUENTIAL_LOOP(i, 0, 5) {
      EXPECT_EQ(uniqueOffsets[i], expectedOffsets[i]);
   } CARE_SEQUENTIAL_LOOP_END

   uniqueOffsets.free();
   uniqueKeys.free();
   offsets.free();
   keys.free();
}

TEST(segmented_unique, in_place_after_segmented_sort)
{
   care::host_device_ptr<int> keys(9);
   care::host_device_ptr<int> offsets(4);

   const int input[] = {
      4, 1, 4,
      8, 7, 8, 7,
      5, 5
   };
   const int segmentOffsets[] = {0, 3, 7, 9};
   const int expectedKeys[] = {1, 4, 7, 8, 5};
   const int expectedOffsets[] = {0, 2, 4, 5};

   CARE_SEQUENTIAL_LOOP(i, 0, 9) {
      keys[i] = input[i];
   } CARE_SEQUENTIAL_LOOP_END

   CARE_SEQUENTIAL_LOOP(i, 0, 4) {
      offsets[i] = segmentOffsets[i];
   } CARE_SEQUENTIAL_LOOP_END

   care::segmented_sort(keys, offsets);
   care::segmented_unique(keys, offsets);

   ASSERT_EQ(keys.size(), 5);
   ASSERT_EQ(offsets.size(), 4);
   CARE_SEQUENTIAL_LOOP(i, 0, 5) {
      EXPECT_EQ(keys[i], expectedKeys[i]);
   } CARE_SEQUENTIAL_LOOP_END
   CARE_SEQUENTIAL_LOOP(i, 0, 4) {
      EXPECT_EQ(offsets[i], expectedOffsets[i]);
   } CARE_SEQUENTIAL_LOOP_END

   offsets.free();
   keys.free();
}

TEST(segmented_unique, empty_input_and_segments)
{
   care::host_device_ptr<int> keys;
   care::host_device_ptr<int> offsets(4);
   care::host_device_ptr<int> uniqueKeys;
   care::host_device_ptr<int> uniqueOffsets;

   CARE_SEQUENTIAL_LOOP(i, 0, 4) {
      offsets[i] = 0;
   } CARE_SEQUENTIAL_LOOP_END

   care::segmented_unique(keys, offsets, uniqueKeys, uniqueOffsets);

   EXPECT_EQ(uniqueKeys.size(), 0);
   ASSERT_EQ(uniqueOffsets.size(), 4);
   CARE_SEQUENTIAL_LOOP(i, 0, 4) {
      EXPECT_EQ(uniqueOffsets[i], 0);
   } CARE_SEQUENTIAL_LOOP_END

   uniqueOffsets.free();
   offsets.free();
}

TEST(segmented_unique, accepts_input_slices)
{
   care::host_device_ptr<int> keyStorage(8);
   care::host_device_ptr<int> offsetStorage(5);
   care::host_device_ptr<int> keys = keyStorage.slice(1, 6);
   care::host_device_ptr<int> offsets = offsetStorage.slice(1, 3);
   care::host_device_ptr<int> uniqueKeys;
   care::host_device_ptr<int> uniqueOffsets;

   const int input[] = {-1, 1, 1, 2, 2, 2, 3, -2};
   const int segmentOffsets[] = {-1, 0, 3, 6, -2};
   const int expectedKeys[] = {1, 2, 2, 3};
   const int expectedOffsets[] = {0, 2, 4};

   CARE_SEQUENTIAL_LOOP(i, 0, 8) {
      keyStorage[i] = input[i];
   } CARE_SEQUENTIAL_LOOP_END
   CARE_SEQUENTIAL_LOOP(i, 0, 5) {
      offsetStorage[i] = segmentOffsets[i];
   } CARE_SEQUENTIAL_LOOP_END

   care::segmented_unique(keys, offsets, uniqueKeys, uniqueOffsets);

   ASSERT_EQ(uniqueKeys.size(), 4);
   ASSERT_EQ(uniqueOffsets.size(), 3);
   CARE_SEQUENTIAL_LOOP(i, 0, 4) {
      EXPECT_EQ(uniqueKeys[i], expectedKeys[i]);
   } CARE_SEQUENTIAL_LOOP_END
   CARE_SEQUENTIAL_LOOP(i, 0, 3) {
      EXPECT_EQ(uniqueOffsets[i], expectedOffsets[i]);
   } CARE_SEQUENTIAL_LOOP_END

   // The out-of-place overload leaves both backing allocations untouched.
   CARE_SEQUENTIAL_LOOP(i, 0, 8) {
      EXPECT_EQ(keyStorage[i], input[i]);
   } CARE_SEQUENTIAL_LOOP_END
   CARE_SEQUENTIAL_LOOP(i, 0, 5) {
      EXPECT_EQ(offsetStorage[i], segmentOffsets[i]);
   } CARE_SEQUENTIAL_LOOP_END

   uniqueOffsets.free();
   uniqueKeys.free();
   offsetStorage.free();
   keyStorage.free();
}

int main(int argc, char** argv)
{
   testing::InitGoogleTest(&argc, argv);

#ifdef CARE_GPUCC
   init_care_for_testing();
#endif

   return RUN_ALL_TESTS();
}
