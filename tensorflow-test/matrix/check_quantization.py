"""Integration regression for named calibration; requires the pinned TF environment."""
import numpy as np
import tensorflow as tf
from types import SimpleNamespace
from quantize import make_model


def main():
    # Different ranges expose positional calibration after converter input reorder.
    x = np.array([[-2., -.5, .25, 2.]], np.float32)
    y = np.array([[1., 2., -.5, .5]], np.float32)
    recipe = SimpleNamespace(fn=tf.subtract, inputs=[x, y], direct=False)
    model, inputs, expected, _ = make_model('SUB', recipe)
    host = tf.lite.Interpreter(model_content=model)
    runner = host.get_signature_runner()
    details = runner.get_input_details()
    del runner
    for name, original in [('x0', x), ('x1', y)]:
        detail = details[name]
        scale, zero = detail['quantization']
        index = next(i for i, d in enumerate(host.get_input_details()) if d['index'] == detail['index'])
        decoded = (inputs[index].astype(np.float64) - zero) * scale
        np.testing.assert_allclose(decoded, original, atol=scale * .501, rtol=0)
    # Noncommutative calculation checks that input identities were not swapped.
    scale, zero = host.get_output_details()[0]['quantization']
    decoded = (expected[0].astype(np.float64) - zero) * scale
    np.testing.assert_allclose(decoded, x - y, atol=.05, rtol=0)
    for name, function in [('MAXIMUM', tf.maximum), ('MINIMUM', tf.minimum)]:
        make_model(name, SimpleNamespace(fn=function, inputs=[x, y], direct=False))
    print('Named calibration regression passed: input ranges and SUB identity preserved')


if __name__ == '__main__': main()
