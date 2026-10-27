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

This change implements the driver replacement and AVH retirement. It does not
implement the separately proposed operator-model generator or claim new
operator/shape/dtype coverage. The existing LiteRT AI-layer generator remains
in use. A build success alone does not qualify the new driver at runtime.
