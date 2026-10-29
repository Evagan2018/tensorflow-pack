"""Compile preserved int8 fixtures; never replace the host reference with NPU output."""
import hashlib
import importlib.metadata
import subprocess
import sys
from tensorflow.lite.python import schema_py_generated as schema


def interface(model):
    graph = schema.Model.GetRootAsModel(model, 0).Subgraphs(0)
    result = []
    for indices in [graph.InputsAsNumpy(), graph.OutputsAsNumpy()]:
        tensors = []
        for index in indices:
            t = graph.Tensors(int(index))
            q = t.Quantization()
            tensors.append((t.Type(), tuple(t.ShapeAsNumpy()),
                tuple(q.ScaleAsNumpy()) if q and q.ScaleLength() else (),
                tuple(q.ZeroPointAsNumpy()) if q and q.ZeroPointLength() else ()))
        result.append(tensors)
    return result


def compile_model(model, output, case_id):
    version = importlib.metadata.version('ethos-u-vela')
    if version != '5.1.0': raise ValueError('Use ethos-u-vela==5.1.0')
    source = output / 'pre-vela' / (case_id + '.tflite')
    source.parent.mkdir(exist_ok=True)
    source.write_bytes(model)
    destination = output / 'vela' / case_id
    destination.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, '-m', 'ethosu.vela', str(source),
        '--output-dir', str(destination), '--accelerator-config', 'ethos-u85-256',
        '--system-config', 'Ethos_U85_SYS_DRAM_Mid', '--memory-mode', 'Shared_Sram']
    result = subprocess.run(command, text=True, capture_output=True, timeout=180)
    (destination / 'compile.log').write_text(result.stdout + result.stderr)
    if result.returncode: raise ValueError('Vela compilation failed; see ' + str(destination / 'compile.log'))
    compiled = (destination / (case_id + '_vela.tflite')).read_bytes()
    if interface(compiled) != interface(model):
        raise ValueError('Vela changed the ordered input/output interface or quantization')
    parsed = schema.Model.GetRootAsModel(compiled, 0)
    if parsed.SubgraphsLength() != 1: raise ValueError('Expected one executable TFLite subgraph')
    graph = parsed.Subgraphs(0)
    builtin_names = {v: k for k, v in vars(schema.BuiltinOperator).items() if isinstance(v, int)}
    operators = []
    for i in range(graph.OperatorsLength()):
        code = parsed.OperatorCodes(graph.Operators(i).OpcodeIndex())
        name = builtin_names[code.BuiltinCode()]
        if name == 'CUSTOM':
            if code.CustomCode() != b'ethos-u': raise ValueError('Unexpected custom operator')
            name = 'ETHOS_U'
        operators.append(dict(operator=name, version=code.Version()))
    npu = sum(op['operator'] == 'ETHOS_U' for op in operators)
    return compiled, operators, dict(version=version, accelerator='ethos-u85-256',
        system='Ethos_U85_SYS_DRAM_Mid', memory='Shared_Sram',
        source_sha256=hashlib.sha256(model).hexdigest(), npu_subgraphs=npu,
        cpu_operators=[o['operator'] for o in operators if o['operator'] != 'ETHOS_U'],
        route='NPU + CPU fallback' if npu and len(operators) > npu else ('NPU' if npu else 'CPU only'))
