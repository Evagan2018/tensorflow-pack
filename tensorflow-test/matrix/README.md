# Full-int8 compiler/core matrix

The requested matrix is recorded in `requested-matrix.json`. Two CPU targets are implemented with GCC 15.3.2, AC6 6.24.0 and Arm LLVM 23.1.0 (six jobs). The exact **Cortex-M55 + Ethos-U85** target is pending user identification; its three cells are **not executed**, not passing. Corstone-320's standard M85 + U85 FVP is not a substitute.

## Version and dependency boundaries

Release versioning is unchanged. The generator currently qualifies the existing 26.08 / rc4 candidate, whose pack metadata version is `1.26.8-rc4`. All five TensorFlow candidate pack versions/hashes are checked; the independent driver remains `ARM::CMSIS-Ethos-U@1.0.0`. CPU jobs do not initialize the NPU.

Compiler package pins live in `vcpkg-configuration.json`: GCC **15.3.2**, AC6 **6.24.0**, LLVM **23.1.0**. The GCC package identifies itself as **Arm GNU Toolchain 15.3.Rel2**, while `arm-none-eabi-gcc --version` reports **15.3.1**. Both package and executable identities are retained as artifacts. CMSIS-NN is pinned to **7.0.0**, matching the TensorFlow candidate's 7.x dependency (8.0.0 is incompatible with this candidate's included headers). Other pins are CMSIS 6.4.0, Cortex_DFP 1.2.0 and CMSIS-Compiler 2.1.0.

The M4 simulation enables its optional single-precision FPU to match Cortex_DFP's default compiler configuration; CMSIS-NN int8 computation uses the M4/DSP path. The M55 simulation explicitly configures integer/float MVE, 2 MiB ITCM and 256 KiB DTCM with bit-28 secure aliases. This accommodates the combined suite and is not a claim about a particular physical board's memory capacity.

## What each job does

1. Restore the exact candidate cache and verify the pack identities.
2. Generate the **39 explicitly selected operator fixtures** from `profile.py`. Deterministic calibration converts the recipes to full-integer TFLite models. Reject float tensors, missing intended operations and non-int8 model inputs/outputs. Shape/control constants may be int32.
3. Quantize the test inputs with each model's scale and zero point. Run host TFLite with delegates disabled to obtain the expected **int8** output values. These are integer-domain goldens, not the original float outputs. Save tensor values, scales, zero points, model hashes and operator versions in the manifest.
4. Generate a Cortex_DFP consumer selecting the TensorFlow **CMSIS-NN** kernel variant. Operators without a CMSIS-NN optimized implementation may use the variant's fallback code; this is not a claim that every operator is accelerated.
5. Build with the pinned compiler. Inspect the linked executable for CMSIS-NN symbols. M55 additionally requires integer Helium instructions in CMSIS-NN functions; M4 rejects those instructions. Retain disassembly and audit results.
6. Execute on the matching FVP. Firmware reads CPUID and reports its compiled MVE feature. The host rejects a missing/incorrect core or MVE identity. Compare every output against its host golden with **exact equality, zero LSB tolerance**. Retain per-case errors, image hash, command, JSON and JUnit.
7. Rebuild with one intentionally incorrect ADD golden. Only ADD may fail, specifically for a numerical mismatch; the remaining 38 cases must still pass. This checks that the comparison actually detects errors.

Static disassembly proves which instructions are linked; it is not an instruction trace or a performance measurement. Runtime identity, invocation and numerical output checks provide separate evidence.

## Coverage is explicit

The unchanged Reference suite covers 92 built-in operators. The new full-int8 profile covers 39 selected operators; the remaining **79 resolver entries** have explicit exclusions, including the previous custom/stateful/control-flow gaps. These are profile limitations, not claims that the runtime lacks int8 support. A conversion/build/runtime failure in a selected case fails the job and is never converted into a skip. There is one small fixture per selected operator, not exhaustive shape, saturation or quantization coverage.

Run `.github/workflows/tests_matrix.yml` on this branch, or locally after preparing candidate packs:

```sh
python tensorflow-test/operators/generate.py --checkout /path/to/litert --packs /path/to/packs --target M55-CMSISNN --compiler GCC
```

The workflow supplies the complete build, binary audit, FVP run and negative control commands. No legacy `test_avh.yml` dependency is introduced.

## Qualification finding: RELU6

The first full-int8 execution exposed a candidate runtime issue that the float Reference tests do not exercise. TensorFlow 2.18 generates a RELU6 model with different input/output quantization. The candidate's generic `tensorflow/lite/micro/kernels/activations.cpp` / `activations_common.cpp` clamp in the input encoding without rescaling to the output encoding. For the recorded fixture, input scale/zero point are 0.019607843831181526 / -1, output scale/zero point are 0.009803921915590763 / -128, and the maximum observed integer discrepancy is 127.

This fixture is retained as a failing regression. The infrastructure does not modify the runtime pack, relax equality, rewrite the model's quantization or skip RELU6. A failed qualification run is the correct outcome until that runtime behavior is fixed or its supported contract is explicitly revised. Negative-control builds still run after a numerical failure: ADD must be detected, and the unrelated RELU6 failure must remain visible and fail the overall gate.

AC6's ELF entry is explicitly set to `Reset_Handler`, so FVP executes the CMSIS startup and `SystemInit` before the C runtime. Without that setting, AC6's default `__main` entry bypasses FPU initialization and faults before main. Startup trace evidence was collected in Actions run 40699489726.

## Recorded execution

[Run 40700594420](https://github.com/Evagan2018/tensorflow-pack/actions/runs/40700594420), implementation commit `1291dee`:

- All six requested CPU/compiler combinations built, passed pack-resolution and binary audits, and executed all 39 cases on the matching core.
- Each positive run passed **38/39**; only `relu6_int8` failed, with maximum integer error **127**. Across the matrix: **228 passes, six failures**.
- Each negative run passed **37/39**, detected the deliberate ADD error of **100**, and retained the RELU6 failure. Its overall acceptance gate correctly remains failed.
- M55 binaries contain integer Helium instructions inside linked CMSIS-NN functions (AC6: 11 functions, LLVM: 14, GCC: 42); runtime core/MVE identity passed. These counts are static evidence, not performance comparisons.
- Generated model hashes, input values and host goldens are identical across all six jobs.
- All three M55 + U85 cells are **not executed**, pending the exact target.

Generate a browser-readable report from the downloaded Actions artifact directories:

```sh
python tensorflow-test/matrix/report.py --artifacts /path/to/artifacts --run-url https://github.com/Evagan2018/tensorflow-pack/actions/runs/40700594420 --output matrix.html
```
