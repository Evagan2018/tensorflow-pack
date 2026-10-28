"""Create a standalone browser-readable coverage report from recorded suite evidence."""
import argparse
import html
import json
from pathlib import Path


def render(manifest, result):
    if result['negative_control']:
        raise ValueError('The coverage overview must use the normal run, not the negative control')
    esc = lambda value: html.escape(str(value))
    cases = {c['operator']: c for c in manifest['cases']}
    rows = []
    for entry in result['coverage']:
        case = cases.get(entry['operator'])
        details = entry.get('reason', '')
        error = '—'
        if case:
            details = '; '.join(f"{t['dtype']} {t['shape']}" for t in case['inputs'])
            details += ' → ' + '; '.join(f"{t['dtype']} {t['shape']}" for t in case['expected'])
            details += '; ' + case['generator']
            recorded = result['cases'].get(case['id'], {})
            error = recorded.get('max_abs', '—')
            if recorded.get('reason') not in (None, 'none'):
                details += '; ' + recorded['reason']
        status = entry['status']
        rows.append(f'<tr data-status="{esc(status)}"><td><code>{esc(entry["operator"])}</code></td>'
                    f'<td>{esc(entry["kind"])}</td><td class="{esc(status)}">{esc(status.upper())}</td>'
                    f'<td>{esc(details)}</td><td>{esc(error)}</td></tr>')
    skips = sum(row['status'] == 'skipped' for row in result['coverage'])
    diagram = '''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1100 190" role="img"
      aria-label="Selected pack to inventory, recipes and skips, generated models, firmware, FVP numerical results">
      <defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0 0L10 5L0 10" fill="#526e82"/></marker></defs>
      <style>text{font:16px system-ui,sans-serif;fill:#183247}.line{stroke:#526e82;stroke-width:2;marker-end:url(#arrow)}</style>
      <rect width="1100" height="190" fill="#eef4fa" rx="12"/>
      <g fill="#dcebf7" stroke="#8bacbf"><rect x="15" y="35" width="190" height="85" rx="8"/><rect x="235" y="35" width="190" height="85" rx="8"/><rect x="455" y="35" width="190" height="85" rx="8"/><rect x="675" y="35" width="190" height="85" rx="8"/><rect x="895" y="35" width="190" height="85" rx="8"/></g>
      <text x="30" y="65">1. Pack inventory</text><text x="30" y="94">Every resolver entry</text>
      <text x="250" y="65">2. Recipe or skip</text><text x="250" y="94">No silent omissions</text>
      <text x="470" y="65">3. Models + goldens</text><text x="470" y="94">Verify intended op</text>
      <text x="690" y="65">4. Consumer build</text><text x="690" y="94">Reference CPU / GCC</text>
      <text x="910" y="65">5. FVP comparison</text><text x="910" y="94">Every expected case</text>
      <path class="line" d="M205 77H235"/><path class="line" d="M425 77H455"/><path class="line" d="M645 77H675"/><path class="line" d="M865 77H895"/>
      <text x="30" y="160">Separate negative control: corrupt only ADD’s golden → ADD must fail; all other cases must still pass.</text></svg>'''
    errors = '<br>'.join(esc(e) for e in result['errors']) or 'No suite-integrity errors.'
    page = f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>TensorFlow operator coverage</title><style>
body{{font:17px/1.55 system-ui,sans-serif;color:#183247;background:#f5f8fc;max-width:1250px;margin:32px auto;padding:0 24px}}
h1{{font-size:30px}}h2{{font-size:23px}}.note{{background:#e1edf9;padding:16px;border-radius:8px}}
svg{{width:100%;height:auto}}table{{border-collapse:collapse;width:100%;background:white;font-size:15px}}
th,td{{text-align:left;border:1px solid #cad8e3;padding:10px;vertical-align:top}}th{{background:#e1edf9}}
.pass{{color:#17683a;font-weight:bold}}.fail,.missing,.invalid{{color:#a11f25;font-weight:bold}}.skipped{{color:#7b570c;font-weight:bold}}
code{{font-size:13px;overflow-wrap:anywhere}}input,select{{font:inherit;padding:9px;margin:8px}}a{{color:#006a96}}
</style></head><body><h1>TensorFlow operator coverage — Reference CPU</h1>
<p><strong>{result['passed']} / {result['expected_count']} executed cases passed; {skips} entries explicitly skipped.</strong>
Overall run accepted: <strong>{esc(result['accepted'])}</strong>.</p>
<p class="note">Profile: {esc(manifest['profile'])}. This tests one reviewed case per executed operator, not every dtype, shape or operator version.
This is Reference CPU execution; it does not establish CMSIS-NN or NPU coverage. A skipped entry is not a pass.</p>
{diagram}<h2>What the table means</h2><p>PASS means that the recorded normal run completed this case's type, shape and numerical comparisons.
SKIPPED means the case was deliberately excluded for the stated reason. The last column is maximum absolute error for float outputs;
integer and boolean outputs are checked exactly. Static model parameters may not appear in the runtime-input column.</p>
<p>{errors}</p><p>Converter: {esc(manifest['converter'])}; NumPy: {esc(manifest['numpy'])}; arena: {manifest['arena_bytes']} bytes.
Executable SHA256: <code>{esc(result['image_sha256'])}</code>.</p>
<label>Search <input id="search" type="search" placeholder="Operator, dtype or reason" oninput="filterRows()"></label>
<label>Status <select id="status" onchange="filterRows()"><option value="">All</option><option value="pass">Passed</option><option value="skipped">Skipped</option><option value="fail">Failed</option></select></label>
<table><thead><tr><th>Operator</th><th>Kind</th><th>Status</th><th>Runtime inputs → outputs / skip reason</th><th>Max float error</th></tr></thead><tbody>{''.join(rows)}</tbody></table>
<h2>Evidence and reproducibility</h2><p>The workflow artifact contains the exact model files, manifest, inventory, pack hashes,
resolved dependencies, source, executable images, FVP logs, JSON verdicts and JUnit results.
<a href="https://github.com/Evagan2018/tensorflow-pack/tree/test-infrastructure-cmsis-ethos-u/tensorflow-test/operators">Source and run instructions</a>.</p>
<script>function filterRows(){{const q=document.getElementById('search').value.toLowerCase();const s=document.getElementById('status').value;
for(const r of document.querySelectorAll('tbody tr'))r.hidden=(!r.textContent.toLowerCase().includes(q))||(s&&r.dataset.status!==s);}}</script></body></html>'''
    return page, diagram


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--results', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    page, diagram = render(json.loads(args.manifest.read_text()), json.loads(args.results.read_text()))
    args.output.write_text(page, encoding='utf-8')
    args.output.with_suffix('.svg').write_text(diagram, encoding='utf-8')
