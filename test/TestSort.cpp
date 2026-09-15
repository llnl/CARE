//////////////////////////////////////////////////////////////////////////////
// Copyright (c) Lawrence Livermore National Security, LLC and other CARE
// contributors. See the CARE LICENSE and COPYRIGHT files for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////

#include "care/sort.h"
#include "care/detail/test_utils.h"

#include "gtest/gtest.h"

TEST(sort, ascending)
{
   care::host_device_ptr<int> keys(6);
   const int input[] = {4, 1, 6, 2, 5, 3};
   const int expected[] = {1, 2, 3, 4, 5, 6};

   CARE_SEQUENTIAL_LOOP(i, 0, 6) {
      keys[i] = input[i];
   } CARE_SEQUENTIAL_LOOP_END

   care::sort(keys);

   CARE_SEQUENTIAL_LOOP(i, 0, 6) {
      EXPECT_EQ(keys[i], expected[i]);
   } CARE_SEQUENTIAL_LOOP_END

   keys.free();
}

TEST(sort, preserves_slice)
{
   care::host_device_ptr<int> storage(6);
   care::host_device_ptr<int> keys = storage.slice(1, 4);
   const int input[] = {-1, 4, 2, 3, 1, -2};
   const int expected[] = {-1, 1, 2, 3, 4, -2};

   CARE_SEQUENTIAL_LOOP(i, 0, 6) {
      storage[i] = input[i];
   } CARE_SEQUENTIAL_LOOP_END

   care::sort(keys);

   CARE_SEQUENTIAL_LOOP(i, 0, 6) {
      EXPECT_EQ(storage[i], expected[i]);
   } CARE_SEQUENTIAL_LOOP_END

   storage.free();
}

TEST(paired_sort, preserves_key_value_association)
{
   care::host_device_ptr<int> keys(6);
   care::host_device_ptr<char> values(6);
   const int inputKeys[] = {4, 1, 6, 2, 5, 3};
   const char inputValues[] = {'d', 'a', 'f', 'b', 'e', 'c'};
   const int expectedKeys[] = {1, 2, 3, 4, 5, 6};
   const char expectedValues[] = {'a', 'b', 'c', 'd', 'e', 'f'};

   CARE_SEQUENTIAL_LOOP(i, 0, 6) {
      keys[i] = inputKeys[i];
      values[i] = inputValues[i];
   } CARE_SEQUENTIAL_LOOP_END

   care::paired_sort(keys, values);

   CARE_SEQUENTIAL_LOOP(i, 0, 6) {
      EXPECT_EQ(keys[i], expectedKeys[i]);
      EXPECT_EQ(values[i], expectedValues[i]);
   } CARE_SEQUENTIAL_LOOP_END

   values.free();
   keys.free();
}

TEST(paired_sort, preserves_slices)
{
   care::host_device_ptr<int> keyStorage(6);
   care::host_device_ptr<char> valueStorage(6);
   care::host_device_ptr<int> keys = keyStorage.slice(1, 4);
   care::host_device_ptr<char> values = valueStorage.slice(1, 4);
   const int inputKeys[] = {-1, 4, 2, 3, 1, -2};
   const char inputValues[] = {'x', 'd', 'b', 'c', 'a', 'y'};
   const int expectedKeys[] = {-1, 1, 2, 3, 4, -2};
   const char expectedValues[] = {'x', 'a', 'b', 'c', 'd', 'y'};

   CARE_SEQUENTIAL_LOOP(i, 0, 6) {
      keyStorage[i] = inputKeys[i];
      valueStorage[i] = inputValues[i];
   } CARE_SEQUENTIAL_LOOP_END

   care::paired_sort(keys, values);

   CARE_SEQUENTIAL_LOOP(i, 0, 6) {
      EXPECT_EQ(keyStorage[i], expectedKeys[i]);
      EXPECT_EQ(valueStorage[i], expectedValues[i]);
   } CARE_SEQUENTIAL_LOOP_END

   valueStorage.free();
   keyStorage.free();
}

// Verify that each segment is sorted independently and empty segments are
// accepted without affecting adjacent segments.
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

// Verify that sorting an empty key array is a no-op.
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

// Verify that sorting a slice updates its backing storage without replacing
// the slice or modifying values outside it.
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

TEST(segmented_paired_sort, segment_local_and_empty)
{
   care::host_device_ptr<int> keys(8);
   care::host_device_ptr<char> values(8);
   care::host_device_ptr<int> offsets(5);

   const int inputKeys[] = {5, 1, 4, 9, 3, 8, 7, 2};
   const char inputValues[] = {'e', 'a', 'd', 'i', 'c', 'h', 'g', 'b'};
   const int segmentOffsets[] = {0, 3, 3, 6, 8};
   const int expectedKeys[] = {1, 4, 5, 3, 8, 9, 2, 7};
   const char expectedValues[] = {'a', 'd', 'e', 'c', 'h', 'i', 'b', 'g'};

   CARE_SEQUENTIAL_LOOP(i, 0, 8) {
      keys[i] = inputKeys[i];
      values[i] = inputValues[i];
   } CARE_SEQUENTIAL_LOOP_END

   CARE_SEQUENTIAL_LOOP(i, 0, 5) {
      offsets[i] = segmentOffsets[i];
   } CARE_SEQUENTIAL_LOOP_END

   care::segmented_paired_sort(keys, values, offsets);

   CARE_SEQUENTIAL_LOOP(i, 0, 8) {
      EXPECT_EQ(keys[i], expectedKeys[i]);
      EXPECT_EQ(values[i], expectedValues[i]);
   } CARE_SEQUENTIAL_LOOP_END

   offsets.free();
   values.free();
   keys.free();
}

TEST(segmented_paired_sort, preserves_slices)
{
   care::host_device_ptr<int> keyStorage(6);
   care::host_device_ptr<char> valueStorage(6);
   care::host_device_ptr<int> keys = keyStorage.slice(1, 4);
   care::host_device_ptr<char> values = valueStorage.slice(1, 4);
   care::host_device_ptr<int> offsets(3);

   const int inputKeys[] = {-1, 4, 2, 5, 3, -2};
   const char inputValues[] = {'x', 'd', 'b', 'e', 'c', 'y'};
   const int segmentOffsets[] = {0, 2, 4};
   const int expectedKeys[] = {-1, 2, 4, 3, 5, -2};
   const char expectedValues[] = {'x', 'b', 'd', 'c', 'e', 'y'};

   CARE_SEQUENTIAL_LOOP(i, 0, 6) {
      keyStorage[i] = inputKeys[i];
      valueStorage[i] = inputValues[i];
   } CARE_SEQUENTIAL_LOOP_END

   CARE_SEQUENTIAL_LOOP(i, 0, 3) {
      offsets[i] = segmentOffsets[i];
   } CARE_SEQUENTIAL_LOOP_END

   care::segmented_paired_sort(keys, values, offsets);

   CARE_SEQUENTIAL_LOOP(i, 0, 6) {
      EXPECT_EQ(keyStorage[i], expectedKeys[i]);
      EXPECT_EQ(valueStorage[i], expectedValues[i]);
   } CARE_SEQUENTIAL_LOOP_END

   offsets.free();
   valueStorage.free();
   keyStorage.free();
}

TEST(sort, empty_inputs)
{
   care::host_device_ptr<int> keys;
   care::host_device_ptr<char> values;
   care::host_device_ptr<int> offsets(1);
   offsets[0] = 0;

   care::sort(keys);
   care::paired_sort(keys, values);
   care::segmented_paired_sort(keys, values, offsets);

   EXPECT_EQ(keys.size(), 0);
   EXPECT_EQ(values.size(), 0);

   offsets.free();
}

int main(int argc, char** argv)
{
   testing::InitGoogleTest(&argc, argv);

#ifdef CARE_GPUCC
   init_care_for_testing();
#endif

   return RUN_ALL_TESTS();
}
