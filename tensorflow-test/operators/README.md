# Generated operator suite — Reference CPU

This extends the original ADD test into a manifest-accounted suite for the
**1.26.8-rc4** candidate. Its resolver header contains **100 built-in** and
**18 custom** registration helpers. The current profile has **92 built-in
operator recipes** and **26 explicit skips** (8 built-in, 18 custom).
These are one reviewed case per executed operator, not exhaustive operator
version, dtype, shape, quantization or hardware coverage.

## What is checked

1. `inventory.py` reads the actual candidate archive's resolver header, including
   custom helpers. New or removed registrations, overlapping recipes/skips,
   missing recipes and empty skip reasons fail reconciliation.
2. `recipes.py` describes fixed inputs and TensorFlow reference calculations.
   TensorFlow 2.18.0 exports most models. Explicit single-op FlatBuffers are
   used where optimization would replace/fold the intended operator (including
   SHAPE, SQUEEZE, FILL and BROADCAST_TO), and for explicit conversion/custom
   fixture needs such as QUANTIZE/DEQUANTIZE. These models still run through the
   pinned host TFLite interpreter and are cross-checked against TF calculations.
   Here “custom fixture” does not mean a TFLite custom operator.
3. Every final model must contain its intended built-in operator. Actual
   operator versions, model hashes, runtime input/output types and shapes,
   quantization metadata, static arguments, expected values and fixed tolerances
   are recorded in `manifest.json`. Conversion failures fail the workflow;
   they never become automatic skips. In particular, static shape parameters
   required by TFLM Prepare are embedded as constant tensors, while the main
   data operand remains a runtime input.
4. Generated `resolver.h` registers all built-ins actually present in the
   exported graphs, including any auxiliary operators. The consumer builds
   with the installed pack's **Reference** component, GCC and the pinned
   Corstone-320 board/startup/stdio environment. The NPU driver component is not
   selected. Shared consumer setup is in `../common/consumer.py`.
5. The C++ runner creates a fresh interpreter/allocator for each case using a
   reusable 128 KiB arena. It checks input/output count, dtype, dimensions and
   byte count. Floating-point outputs must be finite and within the recipe's
   fixed absolute/relative tolerances; integer and boolean outputs are exact.
   Multiple outputs and scalar tensors are supported. Every output element is
   counted, and per-case maximum absolute float error is recorded.
6. The host reconciles every case ID and comparison count against the manifest.
   Duplicate/unknown/missing records, zero cases, inconsistent summaries,
   malformed results, non-finite errors, unexpected failures, process failure
   and timeout fail the run. The FVP has a 30-second simulation limit and a
   300-second wall timeout. This is simulator testing, not a physical-board run.
7. A separate rebuild enables deliberate corruption of ADD's first expected
   value. Exactly ADD must fail numerically, while every other case still passes.
   This control does not turn arbitrary failures into success and its report
   never claims a numerical pass.

## Coverage boundaries

The suite includes arithmetic, comparisons, logical operations, activations,
reductions, pooling, convolution, fully connected and batch matrix operations,
shape/layout/indexing operations, and float/int8 quantize/dequantize conversions.
It does **not** qualify int8 convolution or all dtype combinations simply because
QUANTIZE and DEQUANTIZE have cases.

The eight deferred built-ins require additional test machinery:

| Operators | Required extension |
| --- | --- |
| ASSIGN_VARIABLE, READ_VARIABLE, VAR_HANDLE, CALL_ONCE | Shared resource variables, initialization and state checks across invocations/subgraphs |
| IF, WHILE | Multi-subgraph fixtures, both branch outcomes, loop termination and iteration cases |
| SVDF, UNIDIRECTIONAL_SEQUENCE_LSTM | Recurrent-state fixtures with controlled weights and per-step/reset goldens |

All 18 custom entries remain explicit skips. These include the 14 `Signal*`
helpers (signal-specific scaling/state/reference fixtures), CIRCULAR_BUFFER,
TFLM_DECODE and TFLite_Detection_PostProcess (custom protocols/options), and the
Ethos-U entry (requires Vela and proven NPU execution). The complete individual
reasons and next steps live in `recipes.py` and appear in `coverage.json` and
JUnit. A skip is a reported coverage gap, not a passing test or a claim that
the runtime does not support the operator.

The pack's broad Reference component is compiled, but unreferenced sections can
be removed by the linker. Do not interpret that compilation as build/link or
runtime coverage for every skipped operator.

## Run

Select **Generated operator suite** on branch
`test-infrastructure-cmsis-ethos-u`, revision `26.08`, candidate `rc4`.
The existing Release CI cache must contain the exact candidate artifacts.
Workflow: `.github/workflows/tests_operators.yml`. The existing ADD and
CMSIS-LiteRT integration workflows remain separate regression tests.

Local execution uses the same prerequisites and disposable pinned LiteRT
checkout as [the ADD test](../add/README.md). After candidate preparation and
pack installation, replace its generation/run commands with:

```sh
python "$PACK_REPO/tensorflow-test/operators/generate.py" --checkout . --packs "$PACKS"
cbuild cmsis-litert.csolution.yml --active SSE-320-CPU --toolchain GCC --packs --update-rte
python "$PACK_REPO/.github/scripts/litert_candidate.py" verify --checkout .
python "$PACK_REPO/tensorflow-test/operators/run_fvp.py" --checkout .
cp OperatorTest/control.h OperatorTest/control.positive.h
python "$PACK_REPO/tensorflow-test/operators/negative_control.py" --header OperatorTest/control.h
cbuild cmsis-litert.csolution.yml --active SSE-320-CPU --toolchain GCC --update-rte
python "$PACK_REPO/tensorflow-test/operators/run_fvp.py" --checkout . --name operators-negative --negative
```

Use `../add/requirements.txt` for the pinned host environment. The generator
overwrites solution/project/board files **inside the disposable checkout**.
The currently supported profile is fixed to `1.26.8-rc4`; Release CI versioning
is unchanged. It does not qualify the earlier `1.26.5_rc3` target.

Artifacts include `OperatorTest/` with inventory, coverage, recipes' generated
models, manifest, C++ data/resolver/runner, and generation errors; pack evidence;
generated project/solution and resolved pack locks; build/FVP logs; separate
positive and negative JSON/JUnit reports; both ELF images and their hashes.
The final `control.h` enables the negative control; `control.positive.h` preserves
the pristine switch. All model/reference data remain unchanged by the control.

Host bookkeeping regression tests:

```sh
python -m unittest discover -s tensorflow-test/operators -p 'test_*.py'
```
