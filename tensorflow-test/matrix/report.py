"""Render all requested matrix cells, including targets not yet executable."""
import argparse
import html
import json
from pathlib import Path


def render(root, run_url):
    config = json.loads(Path(__file__).with_name('requested-matrix.json').read_text())
    rows, details, routes = [], [], []
    for target in config['targets']:
        for compiler, version in config['toolchains'].items():
            folder = root / f"int8-{target['id']}-{compiler}"
            status, positive, negative = 'NOT EXECUTED', '—', '—'
            execution = 'CPU' if target['id'] != 'M85-U85' else 'Not verified'
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
                        note = note.rstrip('.') + f". Static audit: {len(code['cmsis_nn_symbols'])} linked CMSIS-NN symbols; {len(code['mve_instructions'])} functions with integer Helium instructions."
                        if code.get('ethosu_invoke_symbols'):
                            note += ' Ethos-U invocation code linked. These CPU fallbacks do not require optimized CMSIS-NN kernels.'
                    manifest = json.loads((folder / 'OperatorTest/manifest.json').read_text())
                    if target['id'] == 'M85-U85':
                        delegated = sum(c['vela']['npu_subgraphs'] > 0 for c in manifest['cases'])
                        observed = sum(r['completion_interrupts'] > 0 for r in result.get('npu_evidence', {}).values())
                        execution = f'{delegated} NPU-planned; {observed} with completion interrupts; {len(manifest["cases"]) - delegated} CPU-only'
                        if not routes:
                            routes = [(c['operator'], c['vela']['route'], ', '.join(c['vela']['cpu_operators']) or '—') for c in manifest['cases']]
                    details.append((target['id'], compiler, manifest, result))
            cells = [target['core'] + ' / ' + target['backend'], compiler + ' ' + version, status, positive, negative, execution, note]
            rows.append('<tr>' + ''.join('<td>' + html.escape(c) + '</td>' for c in cells) + '</tr>')
    operators = ''
    if details:
        manifest = details[0][2]
        operators = '<p><strong>' + str(len(manifest['cases'])) + ' int8 fixtures:</strong> ' + ', '.join(html.escape(c['operator']) for c in manifest['cases']) + '</p>'
        operators += '<details><summary>All coverage exclusions (' + str(sum(r['status']=='skipped' for r in manifest['coverage'])) + ')</summary><ul>'
        operators += ''.join('<li><code>' + html.escape(r['operator']) + '</code>: ' + html.escape(r['reason']) + '</li>' for r in manifest['coverage'] if r['status']=='skipped') + '</ul></details>'
    route_table = ''
    if routes:
        route_table = '<h2>Where the M85 + U85 fixtures execute</h2><p>Vela partitions the original model. The runtime interrupt evidence above independently checks that delegated work completed. CPU fallback is retained, tested and identified.</p><details><summary>Execution route for all 39 fixtures</summary><table><tr><th>Fixture operator</th><th>Route</th><th>Remaining CPU operators</th></tr>'
        route_table += ''.join('<tr>' + ''.join('<td>' + html.escape(c) + '</td>' for c in r) + '</tr>' for r in routes) + '</table></details>'
    return """<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>TensorFlow int8 test matrix</title><style>body{font:16px/1.5 system-ui,sans-serif;max-width:1350px;margin:30px auto;padding:0 24px;background:#f5f8fc;color:#183247}table{border-collapse:collapse;background:white;width:100%;font-size:14px}th,td{border:1px solid #cad8e3;padding:10px;text-align:left;vertical-align:top}th{background:#e1edf9}.note{padding:16px;background:#e1edf9;border-radius:9px}code{font-size:13px}a{color:#006a96}</style>
<h1>TensorFlow pack — int8 test matrix</h1>
<p class="note">Baseline: <strong>1.26.8-rc4</strong> (Release CI inputs 26.08 / rc4). Existing versioning remains unchanged. As requested on 29 October 2026, <strong>Cortex-M85 + Ethos-U85 replaces the previously unexecuted M55 + U85 target</strong>. Corstone-320 supplies the matching FVP. This is a correctness and execution-route report; no hardware speedup has been measured.</p>
<table><thead><tr><th>Core / backend</th><th>Compiler package</th><th>Status</th><th>Numerical passes</th><th>Deliberate ADD error detected?</th><th>Execution route</th><th>Evidence / limitation</th></tr></thead><tbody>""" + ''.join(rows) + """</tbody></table>
<p><a href=""" + '"' + html.escape(run_url, quote=True) + '"' + """>GitHub Actions run and downloadable evidence</a></p>
""" + route_table + """
<h2>Why the new target makes sense</h2>
<p>The former M55 + U85 row had no execution results; it did not demonstrate equal performance. M55 with CMSIS-NN runs inference on the CPU, using Helium where supported. M85 + U85 uses the M85 to manage inference and run CPU fallbacks, while Vela-compiled regions execute on the separate U85 NPU through ARM::CMSIS-Ethos-U. CMSIS-NN itself executes on the CPU, not on the NPU. <a href="https://support.arm.com/compute-ip/corstone-320">Arm Corstone-320 specification</a>; <a href="https://arm-software.github.io/CMSIS-Ethos-U/main/vela/index.html">Vela compilation and CPU fallback</a>.</p>
<h2>Correct answers and faster execution are different measurements</h2>
<table><tr><th>Question</th><th>Evidence needed</th><th>What this report provides</th></tr>
<tr><td>Is the answer correct?</td><td>Compare every output with the preserved host golden.</td><td>Numerical passes and exact integer errors.</td></tr>
<tr><td>Did the NPU do work?</td><td>Compiled Ethos-U regions and runtime completion evidence.</td><td>Vela routing plus per-case NPU interrupts.</td></tr>
<tr><td>How much faster is it?</td><td>Controlled latency measurements on hardware or a validated timing model.</td><td>No measured speedup; FVP host run time is not hardware inference latency.</td></tr></table>
<p>M85 is a higher-performance CPU, but the gain depends on the workload, compiler, clocks, caches and memory. Both M55 and M85 support dual-beat Helium; beat width alone does not explain their performance difference. FPU improvements do not directly accelerate int8 arithmetic, and TrustZone/PACBTI are security features rather than evidence of faster inference. <a href="https://documentation-service.arm.com/static/655e085f2c8b3557fee7048f">Arm Cortex-M comparison</a>.</p>
<p>A U85 can accelerate suitable computation-heavy models, but these very small fixtures may be dominated by dispatch and memory overhead. Keeping the same U85 while changing M55 to M85 primarily changes CPU work and system integration; it does not automatically increase NPU throughput.</p>
<p>The FVP log's wall time, MIPS and performance index describe simulator execution, not physical CPU/NPU speedup. Vela's bundled reference configuration supplies cost-model assumptions such as a 1000 MHz accelerator clock; this is not a measured board clock or a hardware benchmark.</p>
<h2>How to demonstrate performance fairly</h2>
<ol><li>Use the same original int8 model, inputs and expected outputs in every comparison. Record Vela version, partitioning, NPU MAC count and memory configuration.</li>
<li>Compare <strong>M55 CPU-only versus M85 CPU-only</strong> to investigate the CPU change. Compare <strong>M85 CPU-only versus M85 + U85</strong> to isolate the benefit of NPU offload. The additional M85 CPU-only benchmark is a proposed control, not a measured matrix result.</li>
<li>Use representative convolution/fully-connected workloads and an end-to-end model, in addition to small correctness fixtures. Fix clocks, compiler optimization, memory placement and warm/cold-cache conditions.</li>
<li>Measure inference latency over repeated runs after warm-up, separately from model initialization and logging. Report median and spread, CPU fallback time, and <code>speedup = CPU-only latency / CPU+NPU latency</code>.</li>
<li>Use physical hardware or a validated timing model. Arm states that FVP processor models are not cycle accurate; an NPU timing estimate does not make a mixed CPU/NPU workload a hardware benchmark. <a href="https://documentation-service.arm.com/static/6821edae8f79851ff2c3e485">Arm ML Developers Guide, page 36</a>.</li></ol>
<h2>Candidate issue discovered by the CPU matrix</h2>
<p><strong>RELU6 with different input/output quantization fails.</strong> The TensorFlow 2.18 converter emits input scale 0.0196078438 / zero point -1 and output scale 0.0098039219 / zero point -128. The candidate's RELU6 implementation clamps directly in the input encoding and does not rescale to the output encoding. The host golden is retained, and the matrix stays failed when the target disagrees. No tolerance increase, skip, quantization rewrite or runtime-pack modification conceals this finding.</p>
<p>The negative-control run still corrupts ADD. It must detect ADD, but its overall gate also remains failed while the unrelated RELU6 regression exists. See the per-job JSON and logs in the Actions artifacts.</p>
<h2>Separate finding on the M85 + U85 path</h2>
<p>The U85 qualification exposes a <strong>MIRROR_PAD numerical discrepancy</strong>: the first output element is expected to be <code>-27</code> but is <code>-103</code>; the maximum error across the fixture is <code>101</code> integer steps. The model was delegated and an NPU completion interrupt was observed. This identifies a failure in the compiled-model/NPU execution path; it does not yet isolate the root cause to Vela, the runtime, driver or simulator. The golden and strict comparison remain unchanged.</p>
<p><strong>RELU6 passes on the U85 path.</strong> Therefore a 38/39 total on both CPU and NPU paths does not mean the same behavior or performance. CPU jobs fail RELU6; U85 jobs fail MIRROR_PAD. The U85 negative control also detects the deliberately corrupted ADD value, while retaining MIRROR_PAD as an unrelated failure. Neither path is a clean qualification pass.</p>
<h2>What happens in each test?</h2>
<ol><li><strong>Make a tiny model:</strong> take an operation and example arrays from <code>recipes.py</code>. Convert the computation to int8 using the calibration policy in <code>quantize.py</code>. <a href="#tiny-model-inputs">See the ADD example and exact file locations below.</a></li>
<li><strong>Use signed 8-bit numbers:</strong> inputs and outputs are int8. Float tensors or a missing intended operator make generation fail.</li>
<li><strong>Calculate the expected answer:</strong> host TFLite runs the quantized model. Its exact integer outputs become the goldens.</li>
<li><strong>Choose the execution path:</strong> CPU targets retain the original model. The U85 target runs Vela 5.1.0 for <code>ethos-u85-256</code>, preserves the original model and goldens, and verifies that ordered input/output shapes, types and quantization are unchanged.</li>
<li><strong>Build for the selected core:</strong> GCC, AC6 or LLVM builds the consumer. CPU audits require CMSIS-NN symbols and, for M55, Helium instructions. U85 audits require the Ethos-U invocation code; CMSIS-NN functions may be absent if remaining CPU operators use generic kernels.</li>
<li><strong>Run and compare:</strong> the matching FVP executes the firmware, checks CPU identity and compares every output. No integer difference is allowed.</li>
<li><strong>Check the checker:</strong> intentionally change one ADD golden. Only ADD must fail; every other case must still pass.</li></ol>
<section id="tiny-model-inputs"><h3>What do I supply to make a tiny model?</h3>
<p><strong>A tiny model is a small computation graph made for testing.</strong> For example, the ADD model accepts two arrays and adds them. It is not a trained application such as an image classifier. You do not need to supply new files to reproduce our existing tests: the recipes and example numbers are already in the repository.</p>
<table><thead><tr><th>Required information</th><th>ADD example</th><th>Where it comes from</th></tr></thead><tbody>
<tr><td>Calculation</td><td><code>tf.add</code>: add two input tensors.</td><td><a href="https://github.com/Evagan2018/tensorflow-pack/blob/test-infrastructure-cmsis-ethos-u/tensorflow-test/operators/recipes.py">recipes.py — operation, arrays and parameters</a></td></tr>
<tr><td>Example input values, shape and type</td><td>Two arrays, each shape <code>[1, 4]</code>, initially <code>float32</code>.</td><td>The NumPy arrays <code>x</code> and <code>y</code> in the recipe.</td></tr>
<tr><td>Operation settings or constant weights, if needed</td><td>ADD has no weights. Convolution recipes provide fixed filters, stride and padding.</td><td>Arguments and constants inside each recipe function.</td></tr>
<tr><td>Calibration examples and conversion settings</td><td>Six input sets made by multiplying the recipe arrays by <code>0, 0.25, 0.5, 0.75, 1, 1.25</code>. Require int8 model inputs and outputs.</td><td><a href="https://github.com/Evagan2018/tensorflow-pack/blob/test-infrastructure-cmsis-ethos-u/tensorflow-test/matrix/quantize.py">quantize.py — conversion and calibration</a></td></tr>
</tbody></table>
<p>Calibration runs representative examples through the computation so the converter can choose quantization scales and zero points. It does <strong>not</strong> train weights. These deliberately small calibration sets serve the fixtures; a production model would need calibration data representative of its actual use.</p>
<h3>Concrete ADD recipe</h3><p>This simplified excerpt shows the same values and operation selected in <code>recipes.py</code>. The source file registers ADD through a loop.</p>
<pre>x = np.array([[-2.0, -0.5,  0.25, 2.0]], np.float32)
y = np.array([[ 1.0,  2.0, -0.5,  0.5]], np.float32)
recipe = Recipe(tf.add, [x, y])</pre>
<p>The arrays supply the input shapes and initial example values. <code>quantize.py</code> wraps <code>recipe.fn</code> in a TensorFlow function and converts it to an int8 <code>.tflite</code> model. Host TFLite then runs that quantized model to calculate the expected integer answers. Those answers are generated; you do not type them into the recipe. The generator checks that the intended operator survived conversion; a selected operation is not a guarantee that every resulting graph contains exactly one node.</p>
<h3>Where can I find the inputs and generated files?</h3><ul>
<li><strong>Start with the recipe:</strong> <a href="https://github.com/Evagan2018/tensorflow-pack/blob/test-infrastructure-cmsis-ethos-u/tensorflow-test/operators/recipes.py">recipes.py — operation, arrays and parameters</a>. Search for <code>'ADD': tf.add</code> and the definitions of <code>x</code> and <code>y</code>.</li>
<li><strong>Selected test scope:</strong> <a href="https://github.com/Evagan2018/tensorflow-pack/blob/test-infrastructure-cmsis-ethos-u/tensorflow-test/matrix/profile.py">profile.py — selected operators and targets</a>. The <code>OPERATORS</code> set selects the 39 int8 recipes.</li>
<li><strong>Conversion/calibration:</strong> <a href="https://github.com/Evagan2018/tensorflow-pack/blob/test-infrastructure-cmsis-ethos-u/tensorflow-test/matrix/quantize.py">quantize.py — conversion and calibration</a>; <strong>generation entry point:</strong> <a href="https://github.com/Evagan2018/tensorflow-pack/blob/test-infrastructure-cmsis-ethos-u/tensorflow-test/operators/generate.py">generate.py — assemble and save the fixture</a>. The pinned environment uses TensorFlow 2.18.0 and NumPy 2.0.2.</li>
<li><strong>Generated evidence:</strong> open <a href="https://github.com/Evagan2018/tensorflow-pack/actions/runs/40891992958">workflow run 40891992958</a>, download artifact <code>int8-M85-U85-GCC</code>, and unzip it.</li></ul>
<pre>OperatorTest/
  pre-vela/add_int8.tflite             Original quantized model
  vela/add_int8/add_int8_vela.tflite   Vela-compiled model
  vela/add_int8/compile.log           Vela compilation report
  add_int8.tflite                     Model embedded for this target
  manifest.json                      Find cases[] entry id = add_int8
  test_data.h                        Embedded models, inputs and goldens</pre>
<p>In the manifest, <code>inputs</code> contains the actual ordered int8 test values, <code>expected</code> contains the host goldens, and <code>quantization</code> describes their encoding. For CPU artifacts, <code>OperatorTest/add_int8.tflite</code> is the original model; the Vela folders exist only for the U85 path. Generated files are Actions artifacts, not files you should expect to find committed alongside the recipes.</p></section>
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
