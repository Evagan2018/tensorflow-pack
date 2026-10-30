"""Audit emitted bytes, original host inference, and independent numerical checks."""
import argparse
import hashlib
import json
import re
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow.lite.python import schema_py_generated as schema


def audit(root):
    folder = root / 'OperatorTest'
    manifest = json.loads((folder / 'manifest.json').read_text())
    header = (folder / 'test_data.h').read_text()
    arrays = {name: bytes(map(int, raw.split(','))) for name, raw in re.findall(
        r'const unsigned char (\w+)\[\] = \{([0-9,]+)\};', header)}
    results = []
    for index, case in enumerate(manifest['cases']):
        model = (folder / (case['id'] + '.tflite')).read_bytes()
        assert hashlib.sha256(model).hexdigest() == case['model_sha256']
        assert arrays[f'model_{index}'] == model
        for group in ['inputs', 'expected']:
            for j, tensor in enumerate(case[group]):
                array = np.asarray(tensor['values'], dtype=tensor['dtype'])
                assert list(array.shape) == tensor['shape']
                assert arrays[f'{group}_{index}_{j}_bytes'] == array.tobytes()
        original = (folder / 'pre-vela' / (case['id'] + '.tflite')).read_bytes() if 'vela' in case else model
        if 'vela' in case:
            assert hashlib.sha256(original).hexdigest() == case['vela']['source_sha256']
        host = tf.lite.Interpreter(model_content=original,
            experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES)
        host.allocate_tensors()
        for detail, tensor in zip(host.get_input_details(), case['inputs']):
            host.set_tensor(detail['index'], np.asarray(tensor['values'], dtype=np.int8))
        host.invoke()
        for detail, tensor in zip(host.get_output_details(), case['expected']):
            np.testing.assert_array_equal(host.get_tensor(detail['index']), tensor['values'])
        row = dict(case=case['id'], emitted_bytes='match', host_replay='exact', independent_math=None)
        # Independent math for the two reported failures, plus SUM (whose host
        # BUILTIN_REF implementation differs from the optimized host implementation).
        if case['operator'] in ['MIRROR_PAD', 'RELU6', 'SUM']:
            x = np.asarray(case['inputs'][0]['values'], np.int32)
            qi, qo = case['quantization']['inputs'][0], case['quantization']['outputs'][0]
            if case['operator'] == 'MIRROR_PAD':
                parsed = schema.Model.GetRootAsModel(original, 0)
                graph = parsed.Subgraphs(0)
                assert graph.OperatorsLength() == 1
                op = graph.Operators(0)
                options = schema.MirrorPadOptions()
                options.Init(op.BuiltinOptions().Bytes, op.BuiltinOptions().Pos)
                assert options.Mode() == schema.MirrorPadMode.REFLECT
                pads = graph.Tensors(op.Inputs(1))
                padding = np.frombuffer(parsed.Buffers(pads.Buffer()).DataAsNumpy().tobytes(), dtype='<i4').reshape(-1, 2)
                np.testing.assert_array_equal(padding, [[0, 0], [1, 1]])
                assert qi == qo
                answer = np.pad(x, padding, mode='reflect')
                row['independent_math'] = 'NumPy reflect padding; FlatBuffer mode/paddings checked'
            else:
                real = (x.astype(np.float64) - qi['zero_points'][0]) * qi['scales'][0]
                real = np.clip(real, 0, 6) if case['operator'] == 'RELU6' else real.sum(axis=1, keepdims=True)
                answer = np.clip(np.rint(real / qo['scales'][0]) + qo['zero_points'][0], -128, 127).astype(np.int8)
                row['independent_math'] = 'NumPy dequantize, mathematical operation, requantize'
            np.testing.assert_array_equal(answer, case['expected'][0]['values'])
        try:
            reference = tf.lite.Interpreter(model_content=original,
                experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_REF)
            reference.allocate_tensors()
        except ValueError as error:
            # Record the explicit secondary-oracle limitation; primary replay above is mandatory.
            if case['operator'] not in ['DEPTH_TO_SPACE', 'REVERSE_V2']: raise
            row['host_reference'] = 'unavailable for this operator version'
            row['host_reference_detail'] = str(error)
        else:
            for detail, tensor in zip(reference.get_input_details(), case['inputs']):
                reference.set_tensor(detail['index'], np.asarray(tensor['values'], np.int8))
            reference.invoke()
            differences = [int(np.max(np.abs(reference.get_tensor(d['index']).astype(np.int32) -
                np.asarray(t['values'], np.int32)))) for d, t in zip(reference.get_output_details(), case['expected'])]
            row['host_reference_max_abs'] = differences
            if any(differences):
                if case['operator'] != 'SUM' or row['independent_math'] is None:
                    raise ValueError('Unresolved secondary-oracle disagreement: ' + case['id'])
                row['host_reference'] = 'SUM disagreement; independent math confirms primary golden'
            else: row['host_reference'] = 'exact'
        results.append(row)
    report = dict(accepted=True, cases=results, scope='Byte integrity and host oracle audit; not proof that all software is defect-free')
    (folder / 'fixture-audit.json').write_text(json.dumps(report, indent=2))
    print('Fixture audit passed for', len(results), 'cases')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkout', type=Path, required=True)
    audit(p.parse_args().checkout)
