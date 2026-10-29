# CMSIS-Ethos-U migration

The release producer now builds five TensorFlow candidate packs. The existing
`revision` and manually entered lowercase `candidate_rev` inputs, defaults,
SemVer normalization and cache key are unchanged. For example, `26.08` + `rc4`
continues to use the existing version-generation code; no counter is introduced.
The earlier `1.26.5_rc3` test-generator profile is a separate design scope, not a
new default for either existing workflow.

## Dependency selection

The runtime PDSC requires `ARM::CMSIS-Ethos-U@1.0.0` independently of the
TensorFlow version. The release and nightly wrappers no longer stage or package
the legacy driver. The retired packaging directory and its nested Git link are
removed. Historical release metadata in `.pack/tensorflow.pidx` is retained:
renaming that entry would falsely describe the already published 1.26.5 packs.

The CMSIS-LiteRT workflow retains its pinned external source commit. It downloads
the official driver release, checks its committed SHA256, and patches the
solution/board pack selections in the temporary checkout. It preserves the driver
component identifiers, `RTE_ETHOS_U_CORE_DRIVER` and `ethosu_driver.h`.
The board explicitly supplies `ETHOSU_MACS: 256`; the new component supplies
`ETHOSU85`. The existing initialization API is retained.

The preparation helper requires exactly five equally versioned TensorFlow packs
plus the independently pinned driver. It rejects duplicates, legacy driver
archives, substituted driver bytes, and runtime PDSCs requiring the old driver.
It checks all edits before writing. Pack resolution is checked both before and
after generation/build, including rejection of a transitive legacy driver.

## Run

1. Select this branch in **Release CI**. Enter the intended `revision` and
   `candidate_rev` manually. A previously built candidate that depends on the
   legacy driver is not compatible with this migration; rebuild it on this branch.
2. Select this branch in **CMSIS-LiteRT build and execution test**, with exactly
   the same input pair. No fallback cache is accepted.
3. Inspect AC6, GCC and CLANG diagnostics. Each run must build and execute on
   Corstone-320 U85, report 256 MACs/cycle, execute Ethos-U inference and report
   the existing numerical success marker without the failure marker.

Local helper regression checks (Python 3.11 or newer):

```sh
python -m pip install PyYAML==6.0.2
python -m unittest discover -s .github/scripts -p 'test_*.py'
python .github/scripts/litert_candidate.py fetch-driver --packs out
```

`tests_pack.yml` runs helper tests, shell syntax checks and validation of the
published driver download. This workflow does not claim target execution.
`tests_avh.yml` and its external MLEK matrix are removed; the independent
CMSIS-LiteRT FVP tests and their shared tools remain.

## Scope

The driver replacement and AVH retirement are implemented. The first operator
test is now implemented separately in `tensorflow-test/add`: a generated
float32 ADD model using the Reference CPU kernel, with a numerical verdict and
a deliberately incorrect golden value as a negative control. See its
[run instructions](../tensorflow-test/add/README.md) and
[illustrated explanation](add-test/add-test-explained.html).
The generated Reference CPU suite now covers 92 built-in operator cases and
accounts for 26 explicit skips. See [operator-suite instructions and boundaries](../tensorflow-test/operators/README.md).
Stateful/control-flow, custom signal and NPU qualification remain future work. The existing LiteRT
AI-layer generator and integration example remain in use. A build success alone
does not qualify numerical correctness or NPU execution.

## Validation on 2026-10-27

- [Infrastructure checks](https://github.com/Evagan2018/tensorflow-pack/actions/runs/40510581809):
  11 regression tests, shell syntax and official driver checksum passed.
- [Release CI, 26.08 / rc4](https://github.com/Evagan2018/tensorflow-pack/actions/runs/40510583183):
  passed; produced exactly five TensorFlow packs with PDSC version `1.26.8-rc4`.
- [CMSIS-LiteRT matrix](https://github.com/Evagan2018/tensorflow-pack/actions/runs/40511435981):
  AC6, GCC and CLANG built and passed Corstone-320 U85 execution, including CPU
  float inference and Ethos-U quantized inference. Logs reported 256 MACs/cycle
  and `~~~ALL TESTS PASSED~~~`. Tested implementation commit: `1b6613a`.

The attempted [26.05 / rc3 rebuild](https://github.com/Evagan2018/tensorflow-pack/actions/runs/40509920771)
failed in the existing third-party `packchk` release-history validation (M396):
`1.26.5-rc3` was placed before the already published `1.26.5` entry. Versioning
and history generation were deliberately left unchanged. Consequently the
successful runtime result above qualifies only the tested 26.08 / rc4 candidate,
not the earlier 1.26.5_rc3 generator profile or untested NPU variants.

## First generated operator test (2026-10-28)

The [ADD workflow](https://github.com/Evagan2018/tensorflow-pack/actions/runs/40668174880) passed generation, Reference CPU compilation,
pack-resolution verification and FVP numerical execution using `1.26.8-rc4`.
The deliberately incorrect expected-value control was also detected. This is
one float32 ADD case with shape `[1, 4]`, not full operator coverage.
See the [explanation and SVG](add-test/add-test-explained.html).

## Expanded Reference CPU suite (2026-10-28)

[Workflow 40675554477](https://github.com/Evagan2018/tensorflow-pack/actions/runs/40675554477) passed all **92 generated operator cases**
using 1.26.8-rc4, GCC and Corstone-320. The negative-control rebuild changed only
ADD's first expected value: ADD failed as intended and the other 91 cases passed.
JSON and JUnit reports retain all 26 explicit skips, distinct from passing cases.
The resolver inventory has 118 entries; no claim is made that every shape, type
or execution backend is covered. Implementation tested: `7a912f9`.


## Full-int8 compiler/core matrix (2026-10-28)

The [matrix implementation](../tensorflow-test/matrix/README.md) adds 39 full-int8
fixtures across Cortex-M4 CMSIS-NN and Cortex-M55 CMSIS-NN with Helium, using
GCC package 15.3.2, AC6 6.24 and LLVM 23.1.0. On 29 October the user authorized
replacing the pending M55 + Ethos-U85 target with Corstone-320 M85 + Ethos-U85.
That path compiles the preserved fixtures with Vela 5.1.0, retains the original
goldens, and checks per-case NPU completion interrupts and explicit CPU fallback.
It verifies correctness and execution routing, not hardware speedup. The requested
nine cells are retained in `tensorflow-test/matrix/requested-matrix.json`.

The [qualification run](https://github.com/Evagan2018/tensorflow-pack/actions/runs/40700594420)
exposes a RELU6 int8 quantization discrepancy in the candidate's generic kernel.
The host golden is preserved and the failing regression remains enabled.
This is not a passing int8 pack qualification. The generator rejects floating
fallback tensors and verifies the intended operator; firmware checks the core,
while a binary audit checks CMSIS-NN symbols and M55 integer Helium instructions.

The [Reference suite regression](https://github.com/Evagan2018/tensorflow-pack/actions/runs/40700594503)
still passes all 92 cases and its negative control after these shared-runner changes.
Release versioning and candidate pack contents remain unchanged.

Recorded outcome: **38/39 cases passed in each of six CPU/compiler cells**;
only RELU6 failed (maximum integer error 127). ADD corruption was detected in
all six negative runs. All 39 model hashes and goldens match across the matrix.
