"""Generate one float32 ADD model and a Reference CPU consumer in a scratch LiteRT checkout."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import tensorflow as tf
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
from consumer import write_consumer
from tensorflow.lite.python import schema_py_generated as schema


def generate(checkout):
    if tf.__version__ != '2.18.0':
        raise ValueError('Use the pinned TensorFlow 2.18.0 converter')
    evidence = json.loads((checkout / 'candidate-packs.json').read_text())
    versions = {p['version'] for p in evidence if p['pack'].startswith('tensorflow::')}
    if versions != {'1.26.8-rc4'}:
        raise ValueError('This first ADD profile supports only the verified 1.26.8-rc4 artifacts')

    @tf.function(input_signature=[tf.TensorSpec([1, 4], tf.float32, name='a'),
                                  tf.TensorSpec([1, 4], tf.float32, name='b')])
    def add(a, b):
        return tf.add(a, b)

    converter = tf.lite.TFLiteConverter.from_concrete_functions([add.get_concrete_function()])
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    model = converter.convert()
    parsed = schema.Model.GetRootAsModel(model, 0)
    graph = parsed.Subgraphs(0)
    if parsed.SubgraphsLength() != 1 or graph.OperatorsLength() != 1:
        raise ValueError('Expected exactly one subgraph and one operator')
    opcode = parsed.OperatorCodes(graph.Operators(0).OpcodeIndex())
    if opcode.BuiltinCode() != schema.BuiltinOperator.ADD or opcode.Version() != 1:
        raise ValueError('Expected ADD version 1; do not silently test another operation')
    interpreter = tf.lite.Interpreter(model_content=model, num_threads=1,
        experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES)
    interpreter.allocate_tensors()
    inputs = [np.array([[-3., 0., .5, 16.]], dtype=np.float32),
              np.array([[1., -2., .25, -8.]], dtype=np.float32)]
    details = interpreter.get_input_details()
    if len(details) != 2 or len(interpreter.get_output_details()) != 1:
        raise ValueError('Expected two runtime inputs and one output')
    for detail, data in zip(details, inputs):
        if detail['dtype'] != np.float32 or list(detail['shape']) != [1, 4]:
            raise ValueError('Unexpected input metadata')
        interpreter.set_tensor(detail['index'], data)
    interpreter.invoke()
    expected = interpreter.get_tensor(interpreter.get_output_details()[0]['index'])
    np.testing.assert_array_equal(expected, inputs[0] + inputs[1])
    output = checkout / 'AddTest'
    output.mkdir(exist_ok=True)
    (output / 'add.tflite').write_bytes(model)
    def floats(data):
        return ', '.join(f'{float(v):.9f}f' for v in data.flatten())
    header = '#pragma once\n'
    header += 'alignas(16) const unsigned char kModel[] = {' + ','.join(map(str, model)) + '};\n'
    for name, data in [('kInput0', inputs[0]), ('kInput1', inputs[1]), ('kExpected', expected)]:
        header += f'const float {name}[] = {{{floats(data)}}};\n'
    header += 'constexpr float kAtol = 0.000001f;\nconstexpr float kRtol = 0.000001f;\n'
    (output / 'test_data.h').write_text(header)
    shutil.copyfile(Path(__file__).with_name('main.cpp'), output / 'main.cpp')
    manifest = dict(case='add_float32_reference', pack_version='1.26.8-rc4',
                    converter=tf.__version__, numpy=np.__version__, operator='ADD', operator_version=1,
                    shape=[1, 4], dtype='float32', kernel='Reference', execution='CPU',
                    model_sha256=hashlib.sha256(model).hexdigest(),
                    inputs=[v.tolist() for v in inputs], expected=expected.tolist(),
                    atol=1e-6, rtol=1e-6, packs=evidence)
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')

    write_consumer(checkout, evidence)
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkout', type=Path, required=True,
                        help='Disposable pinned CMSIS-LiteRT checkout, after candidate prepare')
    generate(parser.parse_args().checkout)
