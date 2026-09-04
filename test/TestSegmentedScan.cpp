//////////////////////////////////////////////////////////////////////////////
// Copyright (c) Lawrence Livermore National Security, LLC and other CARE
// contributors. See the CARE LICENSE and COPYRIGHT files for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////

#include "care/segmented_scan.h"
#include "care/detail/test_utils.h"

#include "gtest/gtest.h"

TEST(segmented_exclusive_scan, segment_local_empty_and_nonzero_initial_value)
{
   care::host_device_ptr<int> values(8);
   care::host_device_ptr<int> offsets(5);

   const int input[] = {
      5, 1, 4,
      // empty segment
      9, 3, 8,
      7, 2
   };
   const int segmentOffsets[] = {0, 3, 3, 6, 8};
   const int expected[] = {
      10, 15, 16,
      // empty segment
      10, 19, 22,
      10, 17
   };

   CARE_SEQUENTIAL_LOOP(i, 0, 8) {
      values[i] = input[i];
   } CARE_SEQUENTIAL_LOOP_END

   CARE_SEQUENTIAL_LOOP(i, 0, 5) {
      offsets[i] = segmentOffsets[i];
   } CARE_SEQUENTIAL_LOOP_END

   care::segmented_exclusive_scan(values, offsets, 10);

   CARE_SEQUENTIAL_LOOP(i, 0, 8) {
      EXPECT_EQ(values[i], expected[i]);
   } CARE_SEQUENTIAL_LOOP_END

   offsets.free();
   values.free();
}

TEST(segmented_exclusive_scan, preserves_slice)
{
   care::host_device_ptr<int> storage(6);
   care::host_device_ptr<int> values = storage.slice(1, 4);
   care::host_device_ptr<int> offsets(3);

   const int input[] = {-1, 2, 3, 4, 5, -2};
   const int segmentOffsets[] = {0, 2, 4};
   const int expected[] = {-1, 1, 3, 1, 5, -2};

   CARE_SEQUENTIAL_LOOP(i, 0, 6) {
      storage[i] = input[i];
   } CARE_SEQUENTIAL_LOOP_END

   CARE_SEQUENTIAL_LOOP(i, 0, 3) {
      offsets[i] = segmentOffsets[i];
   } CARE_SEQUENTIAL_LOOP_END

   care::segmented_exclusive_scan(values, offsets, 1);

   CARE_SEQUENTIAL_LOOP(i, 0, 6) {
      EXPECT_EQ(storage[i], expected[i]);
   } CARE_SEQUENTIAL_LOOP_END

   offsets.free();
   storage.free();
}

TEST(segmented_exclusive_scan, empty_input)
{
   care::host_device_ptr<int> values;
   care::host_device_ptr<int> offsets(1);

   CARE_SEQUENTIAL_LOOP(i, 0, 1) {
      offsets[i] = 0;
   } CARE_SEQUENTIAL_LOOP_END

   care::segmented_exclusive_scan(values, offsets, 7);
   offsets.free();

   EXPECT_EQ(values.size(), 0);
   EXPECT_EQ(values.data(), nullptr);
}

int main(int argc, char** argv)
{
   testing::InitGoogleTest(&argc, argv);

#ifdef CARE_GPUCC
   init_care_for_testing();
#endif

   return RUN_ALL_TESTS();
}
