"""Run a single ADD image with a wall timeout and strict, case-specific verdict."""
import argparse
import hashlib
import json
import re
import shutil
import subprocess
from pathlib import Path


def verdict(log, returncode, timed_out=False):
    cases = re.findall(r'^ADD_CASE:([^\r\n]+)', log, re.M)
    summaries = re.findall(r'^ADD_SUMMARY:([^\r\n]+)', log, re.M)
    values = re.findall(r'^ADD_VALUE:(\d+):', log, re.M)
    if timed_out or returncode != 0:
        return False
    return (cases == ['add_float32_reference:PASS'] and summaries == ['1/1']
            and values == ['0', '1', '2', '3'] and 'ADD_REASON:' not in log)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkout', type=Path, required=True)
    p.add_argument('--name', default='positive')
    p.add_argument('--expect-numerical-failure', action='store_true')
    args = p.parse_args()
    root = args.checkout.resolve()
    images = sorted(set(root.glob('out/**/cmsis-litert.elf')) | set(root.glob('out/**/cmsis-litert.axf')))
    if len(images) != 1:
        p.error(f'Expected exactly one image, found {len(images)}')
    command = ['FVP_Corstone_SSE-320', '-f', str(root / 'board/Corstone-320/fvp_config.txt'),
               '-a', str(images[0]), '--simlimit', '10', '--stat']
    timed_out = False
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=120)
        log, code = result.stdout + result.stderr, result.returncode
    except subprocess.TimeoutExpired as error:
        timed_out = True
        def decode(data):
            return data.decode(errors='replace') if isinstance(data, bytes) else (data or '')
        log, code = decode(error.stdout) + decode(error.stderr), None
    passed = verdict(log, code, timed_out)
    # A negative control must execute all four comparisons and fail numerically.
    # A crash, timeout, malformed log or missing result never counts as proof.
    negative_detected = (not timed_out and code in (0, 1) and not passed
        and re.findall(r'^ADD_CASE:([^\r\n]+)', log, re.M) == ['add_float32_reference:FAIL']
        and re.findall(r'^ADD_SUMMARY:([^\r\n]+)', log, re.M) == ['0/1']
        and re.findall(r'^ADD_VALUE:(\d+):', log, re.M) == ['0', '1', '2', '3']
        and re.findall(r'^ADD_REASON:([^\r\n]+)', log, re.M) == ['numerical mismatch'])
    report = dict(case='add_float32_reference', numerical_pass=passed,
                  negative_control=args.expect_numerical_failure,
                  negative_control_detected=negative_detected,
                  returncode=code, timed_out=timed_out, command=command,
                  image_sha256=hashlib.sha256(images[0].read_bytes()).hexdigest())
    shutil.copyfile(images[0], root / f'{args.name}.elf')
    (root / f'{args.name}.log').write_text(log)
    (root / f'{args.name}.json').write_text(json.dumps(report, indent=2))
    print(log)
    print(json.dumps(report, indent=2))
    return 0 if (negative_detected if args.expect_numerical_failure else passed) else 1


if __name__ == '__main__':
    raise SystemExit(main())
