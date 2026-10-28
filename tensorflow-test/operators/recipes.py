"""Reviewed small Reference-CPU cases. Conversion failures are errors, never skips."""
from dataclasses import dataclass
import numpy as np
import tensorflow as tf


@dataclass
class Recipe:
    fn: object
    inputs: list
    atol: float = 1e-5
    rtol: float = 1e-5
    direct: bool = False
    constants: tuple = ()  # static control operands required during TFLM Prepare


def recipes():
    r = {}
    x = np.array([[-2., -.5, .25, 2.]], np.float32)
    y = np.array([[1., 2., -.5, .5]], np.float32)
    positive = np.array([[.25, .5, 1., 4.]], np.float32)
    for op, fn in {
        'ABS': tf.abs, 'CEIL': tf.math.ceil, 'COS': tf.cos, 'ELU': tf.nn.elu,
        'EXP': tf.exp, 'FLOOR': tf.floor, 'LEAKY_RELU': lambda a: tf.nn.leaky_relu(a, alpha=.2),
        'LOGISTIC': tf.sigmoid, 'LOG_SOFTMAX': tf.nn.log_softmax, 'NEG': tf.negative,
        'RELU': tf.nn.relu, 'RELU6': tf.nn.relu6, 'ROUND': tf.round, 'SIN': tf.sin,
        'SOFTMAX': tf.nn.softmax, 'SQUARE': tf.square, 'TANH': tf.tanh,
        'HARD_SWISH': lambda a: a * tf.nn.relu6(a + 3.) / 6.,
    }.items():
        r[op] = Recipe(fn, [x])
    for op, fn in {'LOG': tf.math.log, 'SQRT': tf.sqrt, 'RSQRT': tf.math.rsqrt}.items():
        r[op] = Recipe(fn, [positive])
    for op, fn in {
        'ADD': tf.add, 'SUB': tf.subtract, 'MUL': tf.multiply, 'DIV': tf.divide,
        'FLOOR_DIV': tf.math.floordiv, 'FLOOR_MOD': tf.math.floormod,
        'MAXIMUM': tf.maximum, 'MINIMUM': tf.minimum,
        'SQUARED_DIFFERENCE': tf.math.squared_difference,
        'EQUAL': tf.equal, 'NOT_EQUAL': tf.not_equal, 'GREATER': tf.greater,
        'GREATER_EQUAL': tf.greater_equal, 'LESS': tf.less, 'LESS_EQUAL': tf.less_equal,
    }.items():
        r[op] = Recipe(fn, [x, y])
    b = np.array([[True, False, True, False]], np.bool_)
    r['LOGICAL_NOT'] = Recipe(tf.logical_not, [b])
    for op, fn in {'LOGICAL_AND': tf.logical_and, 'LOGICAL_OR': tf.logical_or}.items():
        r[op] = Recipe(fn, [b, ~b])
    r['SELECT_V2'] = Recipe(tf.where, [b, x, y], direct=True)
    r['REDUCE_ALL'] = Recipe(lambda a: tf.reduce_all(a, axis=1), [b])
    for op, fn in {'SUM': tf.reduce_sum, 'MEAN': tf.reduce_mean,
                   'REDUCE_MAX': tf.reduce_max, 'REDUCE_MIN': tf.reduce_min}.items():
        r[op] = Recipe(lambda a, f=fn: f(a, axis=1, keepdims=True), [x])
    r['ARG_MAX'] = Recipe(lambda a: tf.argmax(a, axis=1, output_type=tf.int32), [x])
    r['ARG_MIN'] = Recipe(lambda a: tf.argmin(a, axis=1, output_type=tf.int32), [x])
    r['CAST'] = Recipe(lambda a: tf.cast(a, tf.int32), [x])
    r['CONCATENATION'] = Recipe(lambda a, b: tf.concat([a, b], axis=1), [x, y])
    r['PACK'] = Recipe(lambda a, b: tf.stack([a, b], axis=1), [x, y])
    r['UNPACK'] = Recipe(lambda a: tf.unstack(a, axis=1), [x])
    r['SPLIT'] = Recipe(lambda a: tf.split(a, 2, axis=1), [x])
    r['SPLIT_V'] = Recipe(lambda a: tf.split(a, [1, 3], axis=1), [x])
    r['RESHAPE'] = Recipe(lambda a: tf.reshape(a, [2, 2]), [x])
    r['EXPAND_DIMS'] = Recipe(tf.expand_dims, [x, np.array(2, np.int32)], direct=True, constants=(1,))
    r['SQUEEZE'] = Recipe(lambda a: tf.squeeze(a, axis=0), [x], direct=True)
    r['TRANSPOSE'] = Recipe(lambda a: tf.transpose(a, [1, 0]), [np.arange(6, dtype=np.float32).reshape(2, 3)])
    r['REVERSE_V2'] = Recipe(lambda a: tf.reverse(a, [1]), [x])
    r['SLICE'] = Recipe(lambda a: tf.slice(a, [0, 1], [1, 2]), [x])
    r['STRIDED_SLICE'] = Recipe(lambda a: a[:, ::2], [x])
    r['PAD'] = Recipe(lambda a: tf.pad(a, [[0, 0], [1, 2]]), [x])
    r['PADV2'] = Recipe(lambda a: tf.pad(a, [[0, 0], [1, 2]], constant_values=2.), [x])
    r['MIRROR_PAD'] = Recipe(lambda a: tf.pad(a, [[0, 0], [1, 1]], mode='REFLECT'), [x])
    r['GATHER'] = Recipe(lambda a, b: tf.gather(a, b, axis=1), [x, np.array([3, 1], np.int32)])
    r['GATHER_ND'] = Recipe(tf.gather_nd, [x, np.array([[0, 1], [0, 3]], np.int32)])
    r['BROADCAST_TO'] = Recipe(tf.broadcast_to, [x, np.array([3, 4], np.int32)], direct=True, constants=(1,))
    r['CUMSUM'] = Recipe(lambda a: tf.cumsum(a, axis=1), [x])
    r['L2_NORMALIZATION'] = Recipe(lambda a: tf.math.l2_normalize(a, axis=1), [x])
    # Constants are deliberately tiny; the data operand is always a runtime input.
    weights = tf.constant(np.arange(12, dtype=np.float32).reshape(4, 3) / 16)
    r['FULLY_CONNECTED'] = Recipe(lambda a: tf.matmul(a, weights), [x])
    r['BATCH_MATMUL'] = Recipe(tf.matmul, [np.arange(6, dtype=np.float32).reshape(1, 2, 3),
                                        np.arange(6, dtype=np.float32).reshape(1, 3, 2) / 4])
    image = np.arange(16, dtype=np.float32).reshape(1, 4, 4, 1) / 8 - 1
    filt = tf.constant(np.array([[[[.25]], [[-.5]]], [[[.75]], [[1.]]]], np.float32))
    r['CONV_2D'] = Recipe(lambda a: tf.nn.conv2d(a, filt, strides=1, padding='VALID'), [image])
    r['DEPTHWISE_CONV_2D'] = Recipe(lambda a: tf.nn.depthwise_conv2d(a, filt, [1, 1, 1, 1], 'VALID'), [image])
    r['TRANSPOSE_CONV'] = Recipe(lambda a: tf.nn.conv2d_transpose(a, filt, [1, 5, 5, 1], strides=1, padding='VALID'), [image])
    r['AVERAGE_POOL_2D'] = Recipe(lambda a: tf.nn.avg_pool2d(a, 2, 2, 'VALID'), [image])
    r['MAX_POOL_2D'] = Recipe(lambda a: tf.nn.max_pool2d(a, 2, 2, 'VALID'), [image])
    r['L2_POOL_2D'] = Recipe(lambda a: tf.sqrt(tf.nn.avg_pool2d(tf.square(a), 2, 2, 'VALID')), [image], direct=True)
    r['PRELU'] = Recipe(lambda a, alpha: tf.maximum(a, 0.) + alpha * tf.minimum(a, 0.),
                        [x, np.array([.1, .2, .3, .4], np.float32)], direct=True)
    r['QUANTIZE'] = Recipe(lambda a: tf.cast(tf.clip_by_value(tf.round(a / .25), -128., 127.), tf.int8),
                           [np.array([[-64., -.5, .25, 64.]], np.float32)], direct=True)
    r['DEQUANTIZE'] = Recipe(lambda a: tf.cast(a, tf.float32) * .25,
                             [np.array([[-128, -2, 1, 127]], np.int8)], direct=True)
    r['DYNAMIC_UPDATE_SLICE'] = Recipe(
        lambda a, update, start: tf.tensor_scatter_nd_update(
            a, tf.stack([start, start + tf.constant([0, 1], tf.int32)]), tf.reshape(update, [2])),
        [np.array([[1., 2., 3., 4.]], np.float32), np.array([[8., 9.]], np.float32),
         np.array([0, 1], np.int32)], direct=True)
    r['RESIZE_BILINEAR'] = Recipe(lambda a: tf.image.resize(a, [2, 2], method='bilinear'), [image])
    r['RESIZE_NEAREST_NEIGHBOR'] = Recipe(lambda a: tf.image.resize(a, [2, 2], method='nearest'), [image])
    r['SPACE_TO_DEPTH'] = Recipe(lambda a: tf.nn.space_to_depth(a, 2), [image])
    r['DEPTH_TO_SPACE'] = Recipe(lambda a: tf.nn.depth_to_space(a, 2), [image.reshape(1, 2, 2, 4)])
    r['SPACE_TO_BATCH_ND'] = Recipe(lambda a: tf.space_to_batch_nd(a, [2, 2], [[0, 0], [0, 0]]), [image])
    r['BATCH_TO_SPACE_ND'] = Recipe(lambda a: tf.batch_to_space(a, [2, 2], [[0, 0], [0, 0]]), [image.reshape(4, 2, 2, 1)])
    # These explicit single-op flatbuffers prevent converter folding/rewriting
    # of the operation being qualified. Host TFLite and TF eager still verify it.
    for op, fn, inputs in [
        ('ADD_N', lambda a, b, c: tf.add_n([a, b, c]), [x, y, x]),
        ('SHAPE', lambda a: tf.shape(a), [x]),
        ('ZEROS_LIKE', tf.zeros_like, [x]),
        ('BROADCAST_ARGS', tf.broadcast_dynamic_shape, [np.array([1, 4], np.int32), np.array([3, 1], np.int32)]),
        ('FILL', tf.fill, [np.array([2, 3], np.int32), np.array(.75, np.float32)]),
        ('EMBEDDING_LOOKUP', lambda ids, table: tf.nn.embedding_lookup(table, ids),
         [np.array([2, 0], np.int32), np.arange(12, dtype=np.float32).reshape(3, 4)]),
    ]:
        r[op] = Recipe(fn, inputs, direct=True, constants=(0,) if op == 'FILL' else ())
    return r


# These are explicit limitations of this first Reference-CPU profile, not
# declarations that the runtime lacks these operators. Every skip has a next step.
SKIPS = {
    'ASSIGN_VARIABLE': 'Needs a shared resource-variable fixture and state checks across invocations; add alongside READ_VARIABLE/VAR_HANDLE/CALL_ONCE.',
    'READ_VARIABLE': 'Needs a shared resource-variable fixture and state checks across invocations; add alongside ASSIGN_VARIABLE/VAR_HANDLE/CALL_ONCE.',
    'VAR_HANDLE': 'Resource handles are not numeric outputs; add a composed variable-lifecycle test.',
    'CALL_ONCE': 'Needs a multi-subgraph initialization fixture with a proven one-time side effect.',
    'IF': 'Needs a multi-subgraph fixture exercising both branches; the current runner profile admits a single subgraph.',
    'WHILE': 'Needs multi-subgraph state/termination fixtures including zero and multiple iterations.',
    'SVDF': 'Needs a stateful sequence fixture with controlled feature/time weights and per-step expected outputs.',
    'UNIDIRECTIONAL_SEQUENCE_LSTM': 'Needs recurrent-state and gate-weight fixtures; add sequence and state-reset checks.',
    'CIRCULAR_BUFFER': 'Custom stateful buffer protocol needs a multi-invocation fixture and independent reference.',
    'TFLM_DECODE': 'Custom compressed-data decoder requires format-specific payload fixtures and decoder reference.',
    'TFLite_Detection_PostProcess': 'Custom detection options and multi-output box/score fixtures require dedicated reference validation.',
    'ETHOS_U': 'This Reference CPU profile does not use Vela or execute the NPU; qualify separately with proven delegation and numerical results.',
}
for name, detail in {
    'SignalDelay': 'delay-line state', 'SignalEnergy': 'energy scaling',
    'SignalFftAutoScale': 'FFT scaling', 'SignalFilterBank': 'filter coefficients',
    'SignalFilterBankLog': 'log scaling', 'SignalFilterBankSquareRoot': 'fixed-point square-root scaling',
    'SignalFilterBankSpectralSubtraction': 'noise-estimate state', 'SignalFramer': 'frame overlap/state',
    'SignalIrfft': 'complex layout and inverse scaling', 'SignalOverlapAdd': 'overlap accumulator state',
    'SignalPCAN': 'gain-control state', 'SignalRfft': 'complex layout and forward scaling',
    'SignalStacker': 'stack history', 'SignalWindow': 'window coefficients',
}.items():
    SKIPS[name] = f'Signal-library fixture needed for {detail}; keep signal validation separate from this built-in CPU profile.'
