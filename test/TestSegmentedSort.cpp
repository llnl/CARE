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
#ifdef CARE_GPUCC
   init_care_for_testing();
#endif

   care::host_device_ptr<int> keys(8);
   care::host_device_ptr<int> offsets(5);

   const int input[] = {5, 1, 4, 9, 3, 8, 7, 2};
   const int segmentOffsets[] = {0, 3, 3, 6, 8};
   const int expected[] = {1, 4, 5, 3, 8, 9, 2, 7};

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

TEST(segmented_sort, empty_input)
{
#ifdef CARE_GPUCC
   init_care_for_testing();
#endif

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
