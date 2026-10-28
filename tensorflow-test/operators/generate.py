"""Generate a manifest-accounted Reference CPU operator suite from selected pack artifacts."""
import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import flatbuffers
import numpy as np
import tensorflow as tf
from tensorflow.lite.python import schema_py_generated as schema

from inventory import inventory, reconcile
from recipes import recipes, SKIPS
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'common'))
from consumer import write_consumer

TYPES = {'float32': ('FLOAT32', 'kTfLiteFloat32'), 'int32': ('INT32', 'kTfLiteInt32'),
         'int64': ('INT64', 'kTfLiteInt64'), 'bool': ('BOOL', 'kTfLiteBool'),
         'int8': ('INT8', 'kTfLiteInt8'), 'uint8': ('UINT8', 'kTfLiteUInt8')}


def leaves(value):
    return list(value) if isinstance(value, (tuple, list)) else [value]


def direct_model(name, inputs, expected, constants=()):
    """Explicit single-op model for operations otherwise folded/re-written by conversion."""
    model = schema.ModelT()
    model.version = 3
    model.description = b'Reviewed single-op Reference CPU fixture'
    model.buffers = [schema.BufferT()]
    code = schema.OperatorCodeT()
    code.builtinCode = getattr(schema.BuiltinOperator, name)
    code.deprecatedBuiltinCode = min(code.builtinCode, 127)
    # The host resolver starts BROADCAST_TO at v2 (v1 was rolled back).
    code.version = 2 if name == 'BROADCAST_TO' else 1
    model.operatorCodes = [code]
    graph = schema.SubGraphT()
    graph.name = b'main'
    graph.tensors = []
    for i, data in enumerate(inputs + expected):
        tensor = schema.TensorT()
        tensor.name = f't{i}'.encode()
        tensor.shape = list(data.shape)
        tensor.type = getattr(schema.TensorType, TYPES[str(data.dtype)][0])
        tensor.buffer = 0
        if i < len(inputs) and i in constants:
            buffer = schema.BufferT()
            buffer.data = np.frombuffer(data.tobytes(), dtype=np.uint8)
            tensor.buffer = len(model.buffers)
            model.buffers.append(buffer)
        if data.dtype == np.int8 and name in ('QUANTIZE', 'DEQUANTIZE'):
            tensor.quantization = schema.QuantizationParametersT()
            tensor.quantization.scale = [.25]
            tensor.quantization.zeroPoint = [0]
        graph.tensors.append(tensor)
    graph.inputs = [i for i in range(len(inputs)) if i not in constants]
    graph.outputs = list(range(len(inputs), len(inputs) + len(expected)))
    op = schema.OperatorT()
    op.opcodeIndex = 0
    op.inputs, op.outputs = list(range(len(inputs))), graph.outputs
    option = {'SHAPE': 'ShapeOptions', 'ADD_N': 'AddNOptions',
              'FILL': 'FillOptions', 'ZEROS_LIKE': 'ZerosLikeOptions',
              'SQUEEZE': 'SqueezeOptions', 'EXPAND_DIMS': 'ExpandDimsOptions',
              'BROADCAST_TO': 'BroadcastToOptions', 'SELECT_V2': 'SelectV2Options',
              'QUANTIZE': 'QuantizeOptions', 'DEQUANTIZE': 'DequantizeOptions',
              'L2_POOL_2D': 'Pool2DOptions'}.get(name)
    if option:
        op.builtinOptionsType = getattr(schema.BuiltinOptions, option)
        op.builtinOptions = getattr(schema, option + 'T')()
        if name == 'SHAPE':
            op.builtinOptions.outType = schema.TensorType.INT32
        if name == 'SQUEEZE':
            op.builtinOptions.squeezeDims = [0]
        if name == 'L2_POOL_2D':
            op.builtinOptions.padding = schema.Padding.VALID
            op.builtinOptions.strideW = op.builtinOptions.strideH = 2
            op.builtinOptions.filterWidth = op.builtinOptions.filterHeight = 2
    graph.operators = [op]
    model.subgraphs = [graph]
    builder = flatbuffers.Builder(1024)
    builder.Finish(model.Pack(builder), file_identifier=b'TFL3')
    return bytes(builder.Output())


def make_model(name, recipe):
    inputs = [np.array(x) for x in recipe.inputs]
    eager = [np.array(x) for x in leaves(recipe.fn(*[tf.constant(x) for x in inputs]))]
    if recipe.direct:
        model = direct_model(name, inputs, eager, recipe.constants)
    else:
        class Module(tf.Module):
            @tf.function(input_signature=[tf.TensorSpec(x.shape, tf.as_dtype(x.dtype), name=f'x{i}')
                                           for i, x in enumerate(inputs)])
            def infer(self, *args):
                return {f'o{i}': value for i, value in enumerate(leaves(recipe.fn(*args)))}
        module = Module()
        concrete = module.infer.get_concrete_function()
        converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete], module)
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
        model = converter.convert()
    parsed = schema.Model.GetRootAsModel(model, 0)
    if parsed.SubgraphsLength() != 1:
        raise ValueError('Profile currently requires a single subgraph')
    builtin_names = {v: k for k, v in vars(schema.BuiltinOperator).items() if isinstance(v, int)}
    graph = parsed.Subgraphs(0)
    operators = []
    for i in range(graph.OperatorsLength()):
        code = parsed.OperatorCodes(graph.Operators(i).OpcodeIndex())
        operators.append(dict(operator=builtin_names[code.BuiltinCode()], version=code.Version()))
    if name not in {op['operator'] for op in operators}:
        raise ValueError(f'{name} was removed/replaced during conversion: {operators}')
    host = tf.lite.Interpreter(model_content=model, num_threads=1,
        experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES)
    host.allocate_tensors()
    ins, outs = host.get_input_details(), host.get_output_details()
    if recipe.direct:
        ordered_inputs = [x for i, x in enumerate(inputs) if i not in recipe.constants]
        ordered_eager = eager
    else:
        runner = host.get_signature_runner()
        in_map = {detail['index']: inputs[int(key[1:])] for key, detail in runner.get_input_details().items()}
        out_map = {detail['index']: eager[int(key[1:])] for key, detail in runner.get_output_details().items()}
        del runner  # Release its native interpreter reference before host.invoke().
        ordered_inputs = [in_map[d['index']] for d in ins]
        ordered_eager = [out_map[d['index']] for d in outs]
    if len(ins) != len(inputs) - len(recipe.constants) or len(outs) != len(eager):
        raise ValueError('Input/output count changed')
    for detail, data in zip(ins, ordered_inputs):
        if detail['dtype'] != data.dtype or list(detail['shape']) != list(data.shape):
            raise ValueError('Input metadata changed')
        host.set_tensor(detail['index'], data)
    host.invoke()
    expected = [host.get_tensor(d['index']) for d in outs]
    for actual, reference in zip(expected, ordered_eager):
        if actual.shape != reference.shape or actual.dtype != reference.dtype:
            raise ValueError('Output metadata changed')
        if not np.all(np.isfinite(actual)) or not np.all(np.isfinite(reference)):
            raise ValueError('Non-finite golden is not admitted by this profile')
        if actual.dtype.kind == 'f':
            np.testing.assert_allclose(actual, reference, atol=recipe.atol, rtol=recipe.rtol)
        else:
            np.testing.assert_array_equal(actual, reference)
    return model, ordered_inputs, expected, operators


def emit(cases, methods, output):
    code = ['#pragma once', '#include "types.h"']
    records = []
    for i, case in enumerate(cases):
        model, inputs, expected = case.pop('_data')
        code.append(f'alignas(16) const unsigned char model_{i}[] = {{{",".join(map(str, model))}}};')
        for group, arrays in [('inputs', inputs), ('expected', expected)]:
            tensors = []
            for j, data in enumerate(arrays):
                prefix = f'{group}_{i}_{j}'
                raw = ','.join(map(str, data.tobytes()))
                code.append(f'alignas(16) const unsigned char {prefix}_bytes[] = {{{raw}}};')
                shape = ','.join(map(str, data.shape)) or '0'
                code.append(f'const int {prefix}_shape[] = {{{shape}}};')
                tensors.append(f'{{{prefix}_bytes, {data.nbytes}, {TYPES[str(data.dtype)][1]}, '
                               f'{prefix}_shape, {data.ndim}}}')
            code.append(f'const TensorData {group}_{i}[] = {{{",".join(tensors)}}};')
        # Deliberately corrupt ADD only in a separate negative-control build.
        records.append(f'{{"{case["id"]}", model_{i}, inputs_{i}, {len(inputs)}, expected_{i}, '
                       f'{len(expected)}, {case["atol"]}f, {case["rtol"]}f, '
                       f'{str(case["operator"] == "ADD").lower()}}}')
    code.append('const TestCase kCases[] = {' + ',\n'.join(records) + '};')
    code.append('constexpr size_t kCaseCount = sizeof(kCases)/sizeof(kCases[0]);')
    (output / 'test_data.h').write_text('\n'.join(code) + '\n')
    resolver = '#pragma once\n#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"\n'
    resolver += f'using Resolver = tflite::MicroMutableOpResolver<{len(methods)}>;\n'
    resolver += 'inline bool register_ops(Resolver& r) {\n'
    resolver += ''.join(f'  if (r.{method}() != kTfLiteOk) return false;\n' for method in sorted(methods))
    resolver += '  return true;\n}\n'
    (output / 'resolver.h').write_text(resolver)


def quantization(model_bytes):
    graph = schema.Model.GetRootAsModel(model_bytes, 0).Subgraphs(0)
    result = {}
    for name, indices in [('inputs', graph.InputsAsNumpy()), ('outputs', graph.OutputsAsNumpy())]:
        result[name] = []
        for index in indices:
            q = graph.Tensors(int(index)).Quantization()
            result[name].append(dict(
                scales=[q.Scale(i) for i in range(q.ScaleLength())] if q else [],
                zero_points=[q.ZeroPoint(i) for i in range(q.ZeroPointLength())] if q else [],
                dimension=q.QuantizedDimension() if q else 0))
    return result


def generate(checkout, packs, target=None, compiler="GCC"):
    if tf.__version__ != '2.18.0' or np.__version__ != '2.0.2':
        raise ValueError('Use the pinned TensorFlow/NumPy environment')
    evidence = json.loads((checkout / 'candidate-packs.json').read_text())
    if {e['version'] for e in evidence if e['pack'].startswith('tensorflow::')} != {'1.26.8-rc4'}:
        raise ValueError('Only the 1.26.8-rc4 profile is implemented')
    entry = next(e for e in evidence if e['pack'] == 'tensorflow::tensorflow-lite-micro')
    archive = packs / entry['file']
    if hashlib.sha256(archive.read_bytes()).hexdigest() != entry['sha256']:
        raise ValueError('Candidate archive changed after preparation')
    rows = inventory(archive)
    registry = recipes()
    skips = dict(SKIPS)
    model_builder = make_model
    if target:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'matrix'))
        from profile import OPERATORS, TARGETS, COMPILERS
        from quantize import make_model as model_builder
        if target not in TARGETS or compiler not in COMPILERS:
            raise ValueError('Unknown target/compiler')
        if not OPERATORS <= set(registry): raise ValueError('Unknown int8 recipe')
        skips.update({n: 'Outside the reviewed full-int8 fixture scope; existing Reference coverage is separate.'
                      for n in registry if n not in OPERATORS})
        registry = {n: r for n, r in registry.items() if n in OPERATORS}
    coverage = reconcile(rows, registry, skips)
    output = checkout / 'OperatorTest'
    output.mkdir(exist_ok=True)
    # Existing output is not used as input: only cases generated successfully
    # in this invocation can enter the manifest/header.
    (output / 'coverage.json').write_text(json.dumps(coverage, indent=2))
    (output / 'inventory.json').write_text(json.dumps(rows, indent=2))
    cases, errors, used = [], [], set()
    for name, recipe in sorted(registry.items()):
        try:
            model, inputs, expected, operators = model_builder(name, recipe)
            case_id = name.lower() + '_reference'
            (output / f'{case_id}.tflite').write_bytes(model)
            cases.append(dict(id=case_id, operator=name, operators=operators,
                generator='explicit-flatbuffer' if recipe.direct else 'tensorflow-converter',
                constant_arguments=[dict(index=i, values=np.array(recipe.inputs[i]).tolist())
                                    for i in recipe.constants],
                model_sha256=hashlib.sha256(model).hexdigest(),
                quantization=quantization(model),
                inputs=[dict(shape=list(a.shape), dtype=str(a.dtype), values=a.tolist()) for a in inputs],
                expected=[dict(shape=list(a.shape), dtype=str(a.dtype), values=a.tolist()) for a in expected],
                atol=0.0 if target else recipe.atol, rtol=0.0 if target else recipe.rtol, _data=(model, inputs, expected)))
            used.update(op['operator'] for op in operators)
            print(f'GENERATED {name}', flush=True)
        except Exception as error:
            errors.append(dict(operator=name, error=str(error)))
            print(f'ERROR {name}: {error}', flush=True)
    (output / 'generation-errors.json').write_text(json.dumps(errors, indent=2))
    if errors:
        raise ValueError(f'{len(errors)} generation errors; no suite can run')
    methods = {r['operator']: r['method'] for r in rows if r['kind'] == 'builtin'}
    if not used <= set(methods):
        raise ValueError(f'Model uses unregistered operations: {used-set(methods)}')
    emit(cases, {methods[n] for n in used}, output)
    for filename in ['main.cpp', 'types.h', 'control.h']:
        shutil.copyfile(Path(__file__).with_name(filename), output / filename)
    (output / 'manifest.json').write_text(json.dumps(dict(profile='1.26.8-rc4-int8' if target else '1.26.8-rc4-reference',
        converter=tf.__version__, numpy=np.__version__, cases=cases, coverage=coverage, packs=evidence,
        arena_bytes=131072, scope='One reviewed case per executed operator; not exhaustive dtype/shape coverage'), indent=2))
    write_consumer(checkout, evidence)
    project = checkout / 'cmsis-litert.cproject.yml'
    project.write_text(project.read_text().replace('Generated float32 ADD test', 'Generated operator suite')
                       .replace('ADD test', 'Operator suite').replace('./AddTest/main.cpp', './OperatorTest/main.cpp'))
    solution = checkout / 'cmsis-litert.csolution.yml'
    solution.write_text(solution.read_text().replace('optimize: debug', 'optimize: size'))
    if target:
        from consumer_matrix import configure
        configure(checkout, evidence, target, compiler)
    print(f'Generated {len(cases)} cases; {len(skips)} explicit skips; {len(rows)} inventory entries')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkout', type=Path, required=True)
    parser.add_argument('--packs', type=Path, required=True)
    parser.add_argument('--target', choices=['M4-CMSISNN', 'M55-CMSISNN'])
    parser.add_argument('--compiler', choices=['GCC', 'AC6', 'CLANG'], default='GCC')
    args = parser.parse_args()
    generate(args.checkout, args.packs, args.target, args.compiler)
