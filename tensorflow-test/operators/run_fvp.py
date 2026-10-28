"""Run the suite, reconcile every manifest case, and emit JSON/JUnit coverage evidence."""
import argparse
import hashlib
import json
import math
import re
import shutil
import subprocess
from pathlib import Path
from xml.etree.ElementTree import Element, SubElement, ElementTree


def evaluate(manifest, log, returncode, timed_out=False, negative=False):
    expected = {c['id']: c for c in manifest['cases']}
    if not expected or len(expected) != len(manifest['cases']):
        raise ValueError('Empty/duplicate manifest cases')
    pattern = r'OPS_CASE:([^:]+):(PASS|FAIL):checks=(\d+):max_abs=([^:]+):reason=([^:]+)'
    actual, errors = {}, []
    if 'matrix' in manifest:
        matrix = manifest['matrix']
        identities = re.findall(r'^MATRIX_TARGET:part=([0-9a-f]+):mve=(\d+)$', log, re.M)
        if len(identities) != 1 or int(identities[0][0], 16) != matrix['part'] or bool(int(identities[0][1]) & 1) != matrix['mve']:
            errors.append('Missing or incorrect runtime core/Helium identity')
    for line in log.splitlines():
        if line.startswith('OPS_FATAL:'):
            errors.append(line)
        if not line.startswith('OPS_CASE:'):
            continue
        m = re.fullmatch(pattern, line)
        if not m or m[1] in actual or m[1] not in expected:
            errors.append('Malformed, duplicate or unknown case: ' + line)
            continue
        case = expected[m[1]]
        count = sum(math.prod(t['shape']) for t in case['expected'])
        try:
            delta = float(m[4])
        except ValueError:
            delta = float('nan')
        actual[m[1]] = dict(status=m[2], checks=int(m[3]), max_abs=delta, reason=m[5])
        if int(m[3]) != count or not math.isfinite(delta) or delta < 0:
            errors.append('Incomplete/invalid comparison: ' + m[1])
        want_fail = negative and case['operator'] == 'ADD'
        if (m[2], m[5]) != (('FAIL', 'numerical') if want_fail else ('PASS', 'none')):
            errors.append('Unexpected verdict: ' + m[1])
    if set(actual) != set(expected):
        errors.append('Missing cases: ' + ','.join(sorted(set(expected) - set(actual))))
    total = len(expected)
    passed = sum(r['status'] == 'PASS' for r in actual.values())
    summaries = [line for line in log.splitlines() if line.startswith('OPS_SUMMARY:')]
    if summaries != [f'OPS_SUMMARY:{passed}/{total}']:
        errors.append('Missing, duplicate or inconsistent summary')
    if negative and sum(c['operator'] == 'ADD' for c in expected.values()) != 1:
        errors.append('Negative control requires exactly one ADD case')
    if timed_out or returncode not in ((0, 1) if negative else (0,)):
        errors.append('Process failed or timed out')
    return dict(accepted=not errors, numerical_pass=not negative and not errors,
                negative_control=negative, errors=errors, cases=actual,
                passed=passed, expected_count=total, returncode=returncode, timed_out=timed_out,
                coverage=manifest['coverage'])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkout', type=Path, required=True)
    p.add_argument('--name', default='operators-positive')
    p.add_argument('--negative', action='store_true')
    args = p.parse_args()
    root = args.checkout.resolve()
    manifest = json.loads((root / 'OperatorTest/manifest.json').read_text())
    images = sorted(set(root.glob('out/**/cmsis-litert.elf')) | set(root.glob('out/**/cmsis-litert.axf')))
    if len(images) != 1:
        p.error(f'Expected one executable, found {len(images)}')
    fvp = manifest.get('fvp', dict(executable='FVP_Corstone_SSE-320', config='board/Corstone-320/fvp_config.txt'))
    command = [fvp['executable'], '-f', str(root / fvp['config']),
               '-a', str(images[0]), '--simlimit', '30', '--stat']
    timed_out = False
    try:
        result = subprocess.run(command, text=True, capture_output=True, timeout=300)
        log, code = result.stdout + result.stderr, result.returncode
    except subprocess.TimeoutExpired as error:
        timed_out = True
        def decode(data):
            return data.decode(errors='replace') if isinstance(data, bytes) else (data or '')
        log, code = decode(error.stdout) + decode(error.stderr), None
    report = evaluate(manifest, log, code, timed_out, args.negative)
    report.update(matrix=manifest.get('matrix'), image_sha256=hashlib.sha256(images[0].read_bytes()).hexdigest(), command=command)
    for row in report['coverage']:
        if row['status'] == 'planned':
            found = report['cases'].get(row['operator'].lower() + '_reference')
            row['status'] = found['status'].lower() if found else 'missing'
    (root / f'{args.name}.log').write_text(log)
    # Non-finite deltas are represented as null in the JSON evidence.
    for case in report['cases'].values():
        if not math.isfinite(case['max_abs']): case['max_abs'] = None
    (root / f'{args.name}.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    shutil.copyfile(images[0], root / f'{args.name}.elf')
    suite = Element('testsuite', name='Reference CPU operators' if not args.negative else 'Negative control')
    if args.negative:
        control = SubElement(suite, 'testcase', name='ADD corruption detected; other cases remain passing')
        if not report['accepted']:
            SubElement(control, 'failure').text = '\n'.join(report['errors'])
    else:
        for row in report['coverage']:
            case = SubElement(suite, 'testcase', name=row['operator'])
            if row['status'] == 'skipped':
                SubElement(case, 'skipped', message=row['reason'])
            elif row['status'] != 'pass':
                SubElement(case, 'failure', message=row['status'])
        if report['errors']:
            SubElement(SubElement(suite, 'testcase', name='suite-integrity'), 'failure').text = '\n'.join(report['errors'])
    suite.set('tests', str(len(suite.findall('testcase'))))
    suite.set('failures', str(len(suite.findall('testcase/failure'))))
    suite.set('skipped', str(len(suite.findall('testcase/skipped'))))
    ElementTree(suite).write(root / f'{args.name}.junit.xml', encoding='utf-8', xml_declaration=True)
    print(log)
    print(json.dumps({k: v for k, v in report.items() if k not in ('cases', 'coverage')}, indent=2))
    return 0 if report['accepted'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
