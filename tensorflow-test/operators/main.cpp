#include <cmath>
#include <cstdio>
#include <cstring>
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "test_data.h"
#include "resolver.h"
#include "control.h"
#ifndef OPS_CASE_FILTER
#define OPS_CASE_FILTER ""
#endif
#ifdef OPS_USE_NPU
extern "C" unsigned ops_npu_irq_count(void);
#endif

namespace {
alignas(16) unsigned char arena[131072];
Resolver resolver;

bool metadata(const TfLiteTensor* t, const TensorData& d) {
  if (!t || !t->dims || t->type != d.type || t->bytes != d.size || t->dims->size != d.rank)
    return false;
  for (int i = 0; i < d.rank; ++i) if (t->dims->data[i] != d.shape[i]) return false;
  return true;
}

bool run(const TestCase& c, size_t& checks, float& max_error, const char*& reason) {
  memset(arena, 0, sizeof(arena));
  const auto* model = tflite::GetModel(c.model);
  if (model->version() != TFLITE_SCHEMA_VERSION) { reason = "schema"; return false; }
  tflite::MicroInterpreter interpreter(model, resolver, arena, sizeof(arena));
  if (interpreter.AllocateTensors() != kTfLiteOk) { reason = "allocation"; return false; }
  if (interpreter.inputs_size() != c.input_count || interpreter.outputs_size() != c.output_count) {
    reason = "tensor-count"; return false;
  }
  for (size_t i = 0; i < c.input_count; ++i) {
    if (!metadata(interpreter.input(i), c.inputs[i])) { reason = "input-metadata"; return false; }
    memcpy(interpreter.input(i)->data.raw, c.inputs[i].bytes, c.inputs[i].size);
  }
#ifdef OPS_USE_NPU
  const unsigned before = ops_npu_irq_count();
#endif
  if (interpreter.Invoke() != kTfLiteOk) { reason = "invoke"; return false; }
#ifdef OPS_USE_NPU
  const unsigned interrupts = ops_npu_irq_count() - before;
  printf("OPS_NPU:%s:subgraphs=%u:irqs=%u\n", c.id, c.npu_subgraphs, interrupts);
  if ((c.npu_subgraphs && interrupts < c.npu_subgraphs) || (!c.npu_subgraphs && interrupts)) {
    reason = "npu-evidence"; return false;
  }
#endif
  bool pass = true;
  for (size_t i = 0; i < c.output_count; ++i) {
    auto* t = interpreter.output(i);
    const auto& d = c.expected[i];
    if (!metadata(t, d)) { reason = "output-metadata"; return false; }
    size_t elements = 1;
    for (int j = 0; j < d.rank; ++j) elements *= d.shape[j];
    checks += elements;
    if (t->type == kTfLiteFloat32) {
      for (size_t j = 0; j < elements; ++j) {
        float expected;
        memcpy(&expected, d.bytes + j * sizeof(float), sizeof(float));
        if (OPS_NEGATIVE_CONTROL && c.negative_control_target && i == 0 && j == 0)
          expected += 100.f;
        const float actual = t->data.f[j];
        const float error = std::fabs(actual - expected);
        if (!std::isfinite(actual) || !std::isfinite(expected) || !std::isfinite(error)) {
          pass = false; max_error = INFINITY;
        } else {
          if (error > max_error) max_error = error;
          if (error > c.atol + c.rtol * std::fabs(expected)) pass = false;
        }
      }
    } else if (t->type == kTfLiteInt8) {
      for (size_t j = 0; j < elements; ++j) {
        int expected = static_cast<signed char>(d.bytes[j]);
        if (OPS_NEGATIVE_CONTROL && c.negative_control_target && i == 0 && j == 0)
          expected += expected > 0 ? -100 : 100;
        int delta = int(t->data.int8[j]) - expected;
        if (delta < 0) delta = -delta;
        if (delta > max_error) max_error = float(delta);
        if (delta != 0) {
          if (pass) printf("OPS_MISMATCH:%s:output=%u:index=%u:expected=%d:actual=%d\n",
                           c.id, unsigned(i), unsigned(j), expected, int(t->data.int8[j]));
          pass = false;
        }
      }
    } else if (memcmp(t->data.raw, d.bytes, d.size) != 0) {
      pass = false;
    }
  }
  if (!pass) reason = "numerical";
  if (!pass) {
    for (size_t i = 0; i < c.output_count; ++i) {
      auto* t = interpreter.output(i);
      if (t->type != kTfLiteInt8) continue;
      printf("OPS_OUTPUT:%s:output=%u:actual=", c.id, unsigned(i));
      for (size_t j = 0; j < c.expected[i].size; ++j)
        printf("%s%d", j ? "," : "", int(t->data.int8[j]));
      printf("\n");
    }
  }
  return pass;
}
}

extern "C" int app_main(void) {
  printf("OPS_MODE:negative_control=%u:case_filter=%s\n", unsigned(OPS_NEGATIVE_CONTROL), OPS_CASE_FILTER);
  if (!register_ops(resolver)) { printf("OPS_FATAL:registration\n"); return 1; }
  size_t passed = 0, total = 0;
  for (const auto& c : kCases) {
    if (OPS_CASE_FILTER[0] && strcmp(c.id, OPS_CASE_FILTER) != 0) continue;
    ++total;
    size_t checks = 0;
    float max_error = 0.f;
    const char* reason = "none";
    const bool ok = run(c, checks, max_error, reason);
    passed += ok;
    printf("OPS_CASE:%s:%s:checks=%u:max_abs=%.9g:reason=%s\n",
           c.id, ok ? "PASS" : "FAIL", unsigned(checks), max_error, reason);
  }
  printf("OPS_SUMMARY:%u/%u\n", unsigned(passed), unsigned(total));
  return total && passed == total ? 0 : 1;
}
