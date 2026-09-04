//////////////////////////////////////////////////////////////////////////////
// Copyright (c) Lawrence Livermore National Security, LLC and other CARE
// contributors. See the CARE LICENSE and COPYRIGHT files for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////

#include "care/sort.h"
#include "care/detail/test_utils.h"

#include "gtest/gtest.h"

TEST(segmented_sort, segment_local_and_empty)
{
   care::host_device_ptr<int> keys(8);
   care::host_device_ptr<int> offsets(5);

   const int input[] = {
      5, 1, 4,
      // empty segment
      9, 3, 8,
      7, 2
   };
   const int segmentOffsets[] = {0, 3, 3, 6, 8};
   const int expected[] = {
      1, 4, 5,
      // empty segment
      3, 8, 9,
      2, 7
   };

   CARE_SEQUENTIAL_LOOP(i, 0, 8) {
      keys[i] = input[i];
   } CARE_SEQUENTIAL_LOOP_END

   CARE_SEQUENTIAL_LOOP(i, 0, 5) {
      offsets[i] = segmentOffsets[i];
   } CARE_SEQUENTIAL_LOOP_END

   care::segmented_sort(keys, offsets);

   CARE_SEQUENTIAL_LOOP(i, 0, 8) {
      EXPECT_EQ(keys[i], expected[i]);
   } CARE_SEQUENTIAL_LOOP_END

   offsets.free();
   keys.free();
}

TEST(segmented_sort, preserves_gaps_before_and_after_segments)
{
   care::host_device_ptr<int> keys(9);
   care::host_device_ptr<int> offsets(3);

   const int input[] = {
      -2, -1, // gap before first segment
      5, 1, 4,
      9, 3,
      -4, -3 // gap after final segment
   };
   const int segmentOffsets[] = {2, 5, 7};
   const int expected[] = {
      -2, -1, // gap before first segment
      1, 4, 5,
      3, 9,
      -4, -3 // gap after final segment
   };

   CARE_SEQUENTIAL_LOOP(i, 0, 9) {
      keys[i] = input[i];
   } CARE_SEQUENTIAL_LOOP_END

   CARE_SEQUENTIAL_LOOP(i, 0, 3) {
      offsets[i] = segmentOffsets[i];
   } CARE_SEQUENTIAL_LOOP_END

   care::segmented_sort(keys, offsets);

   CARE_SEQUENTIAL_LOOP(i, 0, 9) {
      EXPECT_EQ(keys[i], expected[i]);
   } CARE_SEQUENTIAL_LOOP_END

   offsets.free();
   keys.free();
}

TEST(segmented_sort, preserves_slice)
{
   care::host_device_ptr<int> storage(6);
   care::host_device_ptr<int> keys = storage.slice(1, 4);
   care::host_device_ptr<int> offsets(3);

   const int input[] = {
      -1, // before slice
      4, 2,
      5, 3,
      -2 // after slice
   };
   const int segmentOffsets[] = {0, 2, 4};
   const int expected[] = {
      -1, // before slice
      2, 4,
      3, 5,
      -2 // after slice
   };

   CARE_SEQUENTIAL_LOOP(i, 0, 6) {
      storage[i] = input[i];
   } CARE_SEQUENTIAL_LOOP_END

   CARE_SEQUENTIAL_LOOP(i, 0, 3) {
      offsets[i] = segmentOffsets[i];
   } CARE_SEQUENTIAL_LOOP_END

   care::segmented_sort(keys, offsets);

   CARE_SEQUENTIAL_LOOP(i, 0, 6) {
      EXPECT_EQ(storage[i], expected[i]);
   } CARE_SEQUENTIAL_LOOP_END

   offsets.free();
   storage.free();
}

TEST(segmented_sort, empty_input)
{
   care::host_device_ptr<int> keys;
   care::host_device_ptr<int> offsets(1);

   CARE_SEQUENTIAL_LOOP(i, 0, 1) {
      offsets[i] = 0;
   } CARE_SEQUENTIAL_LOOP_END

   care::segmented_sort(keys, offsets);
   offsets.free();

   EXPECT_EQ(keys.size(), 0);
   EXPECT_EQ(keys.data(), nullptr);
}

int main(int argc, char** argv)
{
   testing::InitGoogleTest(&argc, argv);

#ifdef CARE_GPUCC
   init_care_for_testing();
#endif

   return RUN_ALL_TESTS();
}
