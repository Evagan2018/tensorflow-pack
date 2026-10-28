# Generated ADD operator test

This first test profile accepts TensorFlow candidate **1.26.8-rc4** only. It tests
one ADD version 1 model, float32, two inputs of shape `[1, 4]`, using the pack's
Reference CPU kernel on Corstone-320 with GCC. It does not qualify other
operators, shapes, quantized kernels, CMSIS-NN or NPU execution.

The existing Release CI and its version-generation logic are unchanged.
`tests_add.yml` uses revision `26.08` and candidate `rc4` for this fixed profile.
The earlier `1.26.5_rc3` target needs its own artifacts/profile and is not
implicitly replaced or qualified by these results.

## Run in GitHub Actions

Select **Generated ADD operator test**, branch
`test-infrastructure-cmsis-ethos-u`, revision `26.08`, candidate `rc4`.
The exact candidate cache must exist from Release CI on the same branch or an
accessible base branch. There are no fallback cache keys. Changes to the ADD
implementation also run this profile on the implementation branch.

The workflow:

1. Checks out the pinned CMSIS-LiteRT board/tool environment, installs the
   five selected TensorFlow packs and the independently pinned driver artifact.
2. Generates a `.tflite` containing exactly one ADD version 1 operation. Both
   operands remain runtime inputs. TensorFlow 2.18.0's host interpreter computes
   the expected result; NumPy addition independently cross-checks these four
   simple values. No expected output is calculated by the device under test.
3. Generates a CPU-only consumer in that disposable checkout. The selected
   kernel component is Reference; the board driver component and NPU setup are
   removed. Pack dependency verification still includes CMSIS-Ethos-U.
4. Uses CMSIS-Toolbox and GCC to compile installed pack sources. The embedded
   C++ settings disable exceptions and RTTI. Other Reference kernels compile as
   part of the component, but only ADD is registered/executed by this test.
5. Runs the image on the FVP and requires a numerical pass, correct output
   count/type/shape, finite values, four value records and one exact case result.
6. Replaces expected output element zero with 99 instead of -2, rebuilds and
   runs again. This negative control must report a numerical mismatch. Crashes,
   missing output or timeouts do not satisfy it.

The workflow artifact `add-reference-cpu` contains the model, manifest, pack
identities/hashes, generated C++/YAML, build logs, positive/negative FVP logs,
JSON verdicts and both executable images with hashes. The final generated header
contains the deliberately wrong value; `test_data.positive.h` preserves the
original golden data. `negative.json` must have `numerical_pass: false` and
`negative_control_detected: true`.

## Local use

Use Python 3.11, the dependencies in `requirements.txt`, and the compiler/FVP
tools specified by the pinned LiteRT checkout's `vcpkg-configuration.json`.
`generate.py` **overwrites** solution/project/board selections in its `--checkout`;
use a disposable checkout at commit
`84bdb98f2045ce299686364518615177ecb0fe25`, not your working application.

From that scratch checkout, with `PACK_REPO` pointing to this repository and
`PACKS` to the selected pack directory:

```sh
python -m pip install -r "$PACK_REPO/tensorflow-test/add/requirements.txt"
python "$PACK_REPO/.github/scripts/litert_candidate.py" fetch-driver --packs "$PACKS"
python "$PACK_REPO/.github/scripts/litert_candidate.py" prepare --packs "$PACKS" --checkout .
cpackget add "$PACKS"/*.pack -a
python "$PACK_REPO/tensorflow-test/add/generate.py" --checkout .
cbuild cmsis-litert.csolution.yml --active SSE-320-CPU --toolchain GCC --packs --update-rte
python "$PACK_REPO/.github/scripts/litert_candidate.py" verify --checkout .
python "$PACK_REPO/tensorflow-test/add/run_fvp.py" --checkout . --name positive
cp AddTest/test_data.h AddTest/test_data.positive.h
python "$PACK_REPO/tensorflow-test/add/negative_control.py" --header AddTest/test_data.h
cbuild cmsis-litert.csolution.yml --active SSE-320-CPU --toolchain GCC --update-rte
python "$PACK_REPO/tensorflow-test/add/run_fvp.py" --checkout . --name negative --expect-numerical-failure
```

Run host verdict tests with:

```sh
python -m unittest discover -s tensorflow-test/add -p 'test_*.py'
```

The FVP has a 10-second simulated-time limit and a 120-second host wall timeout.
A positive result requires process exit code zero and complete numerical evidence.
The negative control permits exit code zero or one, but only with the exact
numerical-failure evidence. All other terminations fail the workflow.

See [the browser explanation](../../docs/add-test/add-test-explained.html) and
[the SVG flow diagram](../../docs/add-test/add-test-flow.svg).

## Verified result (2026-10-28)

[Workflow 40668174880](https://github.com/Evagan2018/tensorflow-pack/actions/runs/40668174880) passed at implementation commit
`e5ab844`. The normal run returned `[-2, -2, 0.75, 8]`, reported `1/1`
and exited with code 0. The negative control reported a numerical mismatch
for actual -2 versus expected 99 and `0/1`; its host control check passed.
Both runs completed without a wall timeout. The artifact contains both ELF
images and their SHA256 hashes.
