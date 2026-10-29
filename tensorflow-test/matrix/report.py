"""Render all requested matrix cells, including targets not yet executable."""
import argparse
import html
import json
from pathlib import Path


def render(root, run_url):
    config = json.loads(Path(__file__).with_name('requested-matrix.json').read_text())
    rows, details = [], []
    for target in config['targets']:
        for compiler, version in config['toolchains'].items():
            folder = root / f"int8-{target['id']}-{compiler}"
            status, positive, negative = 'NOT EXECUTED', '—', '—'
            if target['readiness'] == 'awaiting-user-target':
                note = target['reason']
            else:
                note = 'No complete evidence downloaded.'
                pos, neg = folder / 'operators-positive.json', folder / 'operators-negative.json'
                if pos.exists():
                    result = json.loads(pos.read_text())
                    positive = f"{result['passed']}/{result['expected_count']}"
                    control = json.loads(neg.read_text()) if neg.exists() else {}
                    add = control.get('cases', {}).get('add_int8', {})
                    detected = add.get('status') == 'FAIL' and add.get('reason') == 'numerical'
                    negative = 'detected; suite accepted' if control.get('accepted') else ('ADD detected; other failure' if detected else 'not verified')
                    audit = folder / 'binary-audit.json'
                    status = 'PASS' if result['numerical_pass'] and control.get('accepted') and audit.exists() else ('FAIL' if neg.exists() and audit.exists() else 'INCOMPLETE')
                    note = '; '.join(result['errors']) or 'Exact int8 equality; runtime CPU check.'
                    if audit.exists():
                        code = json.loads(audit.read_text())
                        note += f" {len(code['cmsis_nn_symbols'])} linked CMSIS-NN symbols; {len(code['mve_instructions'])} functions with integer Helium instructions."
                    manifest = json.loads((folder / 'OperatorTest/manifest.json').read_text())
                    details.append((target['id'], compiler, manifest, result))
            cells = [target['core'] + ' / ' + target['backend'], compiler + ' ' + version, status, positive, negative, note]
            rows.append('<tr>' + ''.join('<td>' + html.escape(c) + '</td>' for c in cells) + '</tr>')
    operators = ''
    if details:
        manifest = details[0][2]
        operators = '<p><strong>' + str(len(manifest['cases'])) + ' int8 fixtures:</strong> ' + ', '.join(html.escape(c['operator']) for c in manifest['cases']) + '</p>'
        operators += '<details><summary>All coverage exclusions (' + str(sum(r['status']=='skipped' for r in manifest['coverage'])) + ')</summary><ul>'
        operators += ''.join('<li><code>' + html.escape(r['operator']) + '</code>: ' + html.escape(r['reason']) + '</li>' for r in manifest['coverage'] if r['status']=='skipped') + '</ul></details>'
    return """<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>TensorFlow int8 test matrix</title><style>body{font:16px/1.5 system-ui,sans-serif;max-width:1350px;margin:30px auto;padding:0 24px;background:#f5f8fc;color:#183247}table{border-collapse:collapse;background:white;width:100%;font-size:14px}th,td{border:1px solid #cad8e3;padding:10px;text-align:left;vertical-align:top}th{background:#e1edf9}.note{padding:16px;background:#e1edf9;border-radius:9px}code{font-size:13px}a{color:#006a96}</style>
<h1>TensorFlow pack — int8 test matrix</h1>
<p class="note">Baseline: <strong>1.26.8-rc4</strong> (Release CI inputs 26.08 / rc4). Existing versioning remains unchanged. The M55 + Ethos-U85 target is awaiting your identification; M85 + U85 has not been substituted.</p>
<table><thead><tr><th>Core / backend</th><th>Compiler package</th><th>Status</th><th>Numerical passes</th><th>Deliberate ADD error detected?</th><th>Evidence / limitation</th></tr></thead><tbody>""" + ''.join(rows) + """</tbody></table>
<p><a href=""" + '"' + html.escape(run_url, quote=True) + '"' + """>GitHub Actions run and downloadable evidence</a></p>
<h2>Candidate issue discovered by this matrix</h2>
<p><strong>RELU6 with different input/output quantization fails.</strong> The TensorFlow 2.18 converter emits input scale 0.0196078438 / zero point -1 and output scale 0.0098039219 / zero point -128. The candidate's RELU6 implementation clamps directly in the input encoding and does not rescale to the output encoding. The host golden is retained, and the matrix stays failed when the target disagrees. No tolerance increase, skip, quantization rewrite or runtime-pack modification conceals this finding.</p>
<p>The negative-control run still corrupts ADD. It must detect ADD, but its overall gate also remains failed while the unrelated RELU6 regression exists. See the per-job JSON and logs in the Actions artifacts.</p>
<h2>What happens in each test?</h2>
<ol><li><strong>Make a tiny model:</strong> the recipe performs one selected operation. Calibration chooses its quantization scales.</li>
<li><strong>Use signed 8-bit numbers:</strong> inputs and outputs are int8. Float tensors or a missing intended operator make generation fail.</li>
<li><strong>Calculate the expected answer:</strong> host TFLite runs the quantized model. Its exact integer outputs become the goldens.</li>
<li><strong>Build for the selected core:</strong> GCC, AC6 or LLVM builds the CMSIS-NN consumer. The binary audit checks optimized kernel symbols and M55 Helium instructions.</li>
<li><strong>Run and compare:</strong> the matching FVP executes the firmware, checks CPU identity and compares every output. No integer difference is allowed.</li>
<li><strong>Check the checker:</strong> intentionally change one ADD golden. Only ADD must fail; every other case must still pass.</li></ol>
<h2>Scope and versions</h2><p>The original 92-operator Reference suite remains separate. The int8 fixtures below are not exhaustive operator, shape or quantization coverage. CMSIS-NN's kernel variant includes fallback implementations; this does not imply every operator uses an optimized CMSIS-NN routine. The disassembly check is static code evidence, not a runtime instruction trace or performance measurement.</p>
<p>GCC package <strong>15.3.2</strong> is Arm GNU Toolchain <strong>15.3.Rel2</strong>; its executable reports <strong>15.3.1</strong>. AC6 is <strong>6.24</strong>, LLVM is <strong>23.1.0</strong>. The artifacts record both package selections and executable version output. CMSIS-NN <strong>7.0.0</strong> matches this TensorFlow candidate's dependency; CMSIS 6.4.0, CMSIS-Compiler 2.1.0 and Cortex_DFP 1.2.0 are pinned.</p>""" + operators + '</html>'


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--artifacts', type=Path, required=True)
    p.add_argument('--run-url', required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render(args.artifacts, args.run_url), encoding='utf-8')
