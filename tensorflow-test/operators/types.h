#pragma once
#include <cstddef>
#include "tensorflow/lite/c/common.h"

struct TensorData {
  const unsigned char* bytes;
  size_t size;
  TfLiteType type;
  const int* shape;
  int rank;
};
struct TestCase {
  const char* id;
  const unsigned char* model;
  const TensorData* inputs;
  size_t input_count;
  const TensorData* expected;
  size_t output_count;
  float atol, rtol;
  bool negative_control_target;
  unsigned npu_subgraphs;
};
