"""Explicit int8 scope; exclusions are planned, never inferred from failed conversion."""
OPERATORS = frozenset("""ADD SUB MUL ABS RELU RELU6 LEAKY_RELU LOGISTIC TANH SOFTMAX
MAXIMUM MINIMUM CONCATENATION PACK UNPACK SPLIT SPLIT_V RESHAPE TRANSPOSE
REVERSE_V2 SLICE STRIDED_SLICE PAD PADV2 MIRROR_PAD MEAN SUM REDUCE_MAX REDUCE_MIN
FULLY_CONNECTED CONV_2D DEPTHWISE_CONV_2D AVERAGE_POOL_2D MAX_POOL_2D
RESIZE_NEAREST_NEIGHBOR SPACE_TO_DEPTH DEPTH_TO_SPACE SPACE_TO_BATCH_ND BATCH_TO_SPACE_ND""".split())
TARGETS = {
    'M4-CMSISNN': dict(core='Cortex-M4', device='ARMCM4', part=0xc24, mve=False,
                       model='FVP_MPS2_Cortex-M4', folder='CM4'),
    'M55-CMSISNN': dict(core='Cortex-M55', device='ARMCM55', part=0xd22, mve=True,
                        model='FVP_MPS2_Cortex-M55', folder='CM55'),
    'M85-U85': dict(core='Cortex-M85', device='SSE-320-FVP', part=0xd23, mve=True,
                    model='FVP_Corstone_SSE-320', npu='Ethos-U85', macs=256),
}
COMPILERS = {'GCC': '15.3.2', 'AC6': '6.24.0', 'CLANG': '23.1.0'}
