#include <cmath>
#include <cstdio>
#include <cstring>
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "test_data.h"

namespace {
alignas(16) unsigned char arena[32768];
bool tensor_ok(const TfLiteTensor* t) {
  return t && t->type == kTfLiteFloat32 && t->dims && t->dims->size == 2 &&
         t->dims->data[0] == 1 && t->dims->data[1] == 4 && t->bytes == 4 * sizeof(float);
}
int fail(const char* reason) {
  printf("ADD_REASON:%s\nADD_CASE:add_float32_reference:FAIL\nADD_SUMMARY:0/1\n", reason);
  return 1;
}
}

extern "C" int app_main(void) {
  const auto* model = tflite::GetModel(kModel);
  if (model->version() != TFLITE_SCHEMA_VERSION) return fail("schema");
  tflite::MicroMutableOpResolver<1> resolver;
  if (resolver.AddAdd() != kTfLiteOk) return fail("registration");
  tflite::MicroInterpreter interpreter(model, resolver, arena, sizeof(arena));
  if (interpreter.AllocateTensors() != kTfLiteOk) return fail("allocation");
  if (interpreter.inputs_size() != 2 || interpreter.outputs_size() != 1)
    return fail("tensor count");
  if (!tensor_ok(interpreter.input(0)) || !tensor_ok(interpreter.input(1)))
    return fail("input metadata");
  memcpy(interpreter.input(0)->data.f, kInput0, sizeof(kInput0));
  memcpy(interpreter.input(1)->data.f, kInput1, sizeof(kInput1));
  if (interpreter.Invoke() != kTfLiteOk) return fail("invoke");
  const auto* output = interpreter.output(0);
  if (!tensor_ok(output)) return fail("output metadata");
  bool pass = true;
  for (int i = 0; i < 4; ++i) {
    const float actual = output->data.f[i];
    const float expected = kExpected[i];
    printf("ADD_VALUE:%d:actual=%.9g:expected=%.9g\n", i, actual, expected);
    if (!std::isfinite(actual) || !std::isfinite(expected) ||
        std::fabs(actual - expected) > kAtol + kRtol * std::fabs(expected)) pass = false;
  }
  if (!pass) return fail("numerical mismatch");
  printf("ADD_CASE:add_float32_reference:PASS\nADD_SUMMARY:1/1\n");
  return 0;
}
