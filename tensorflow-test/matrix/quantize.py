"""Full-integer conversion with deterministic calibration and a host TFLite oracle."""
import numpy as np
import tensorflow as tf
from tensorflow.lite.python import schema_py_generated as schema


def calibration_samples(name, recipe):
    for factor in [0., .25, .5, .75, 1., 1.25]:
        yield {f'x{i}': np.asarray(x * factor, np.float32) for i, x in enumerate(recipe.inputs)}
    if name in ['MAXIMUM', 'MINIMUM']:
        # These operations use a common input/output encoding. The paired recipe
        # inputs alone can hide extremes from the output during calibration.
        # Exercise both common-range endpoints so the original inputs fit too.
        for value in [min(float(x.min()) for x in recipe.inputs) * 1.25,
                      max(float(x.max()) for x in recipe.inputs) * 1.25]:
            yield {f'x{i}': np.full(x.shape, value, np.float32) for i, x in enumerate(recipe.inputs)}


def make_model(name, recipe):
    if recipe.direct or any(x.dtype != np.float32 for x in recipe.inputs):
        raise ValueError('This profile requires float source recipes and integer model inputs')
    class Module(tf.Module):
        @tf.function(input_signature=[tf.TensorSpec(x.shape, tf.float32, name=f'x{i}')
                                       for i, x in enumerate(recipe.inputs)])
        def infer(self, *args):
            result = recipe.fn(*args)
            if not isinstance(result, (list, tuple)): result = [result]
            return {f'o{i}': value for i, value in enumerate(result)}
    module = Module()
    converter = tf.lite.TFLiteConverter.from_concrete_functions([module.infer.get_concrete_function()], module)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    # Fixed data, no random seed or calibration-dependent acceptance tolerance.
    # Signature names are stable; converter tensor order may differ from x0,x1.
    converter.representative_dataset = lambda: calibration_samples(name, recipe)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    model = converter.convert()
    parsed = schema.Model.GetRootAsModel(model, 0)
    if parsed.SubgraphsLength() != 1: raise ValueError('Expected one subgraph')
    graph = parsed.Subgraphs(0)
    names = {v: k for k, v in vars(schema.BuiltinOperator).items() if isinstance(v, int)}
    operators = []
    for i in range(graph.OperatorsLength()):
        code = parsed.OperatorCodes(graph.Operators(i).OpcodeIndex())
        operators.append(dict(operator=names[code.BuiltinCode()], version=code.Version()))
    if name not in {o['operator'] for o in operators}: raise ValueError(f'{name} disappeared: {operators}')
    for i in range(graph.TensorsLength()):
        if graph.Tensors(i).Type() in (schema.TensorType.FLOAT32, schema.TensorType.FLOAT16):
            raise ValueError('Float tensor/fallback in full-integer model')
    host = tf.lite.Interpreter(model_content=model, num_threads=1,
        experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES)
    host.allocate_tensors()
    runner = host.get_signature_runner()
    source = {v['index']: recipe.inputs[int(k[1:])] for k,v in runner.get_input_details().items()}
    del runner
    inputs = []
    for detail in host.get_input_details():
        if detail['dtype'] != np.int8: raise ValueError('Non-int8 input')
        scale, zero = detail['quantization']
        if not scale > 0: raise ValueError('Invalid input scale')
        values = source[detail['index']]
        encoded = np.rint(values / scale) + zero
        if not np.all(np.isfinite(encoded)) or np.any(encoded < -128) or np.any(encoded > 127):
            raise ValueError('Recipe input clips in calibrated range; check named calibration mapping')
        data = encoded.astype(np.int8)
        if not np.all(np.abs((data.astype(np.float64) - zero) * scale - values) <= scale * .501):
            raise ValueError('Input quantization round-trip exceeds half an integer step')
        host.set_tensor(detail['index'], data)
        inputs.append(data)
    host.invoke()
    expected = [host.get_tensor(d['index']) for d in host.get_output_details()]
    if any(x.dtype != np.int8 for x in expected): raise ValueError('Non-int8 output')
    return model, inputs, expected, operators
