"""Shared generated Reference-CPU consumer using the pinned LiteRT board."""
import yaml

def write_consumer(checkout, evidence, use_npu=False):
    # Reuse the pinned board's startup and semihosting, without NPU initialization.
    board_path = checkout / 'board/Corstone-320/Board-U85.clayer.yml'
    board = yaml.safe_load(board_path.read_text())
    layer = board['layer']
    layer['description'] = 'Corstone-320 CPU-only ADD test; NPU not selected'
    layer['define'] = [d for d in layer['define'] if not
                       (isinstance(d, dict) and 'ETHOSU_MACS' in d) and d != 'ETHOSU85']
    layer['components'] = [c for c in layer['components'] if 'Ethos-U Driver' not in c['component']]
    for group in layer['groups']:
        group['files'] = [f for f in group['files'] if not f['file'].endswith('ethos_setup.c')]
    if not use_npu:
        board_path.write_text(yaml.safe_dump(board, sort_keys=False))
    selections = '\n'.join(f"    - pack: {e['pack']}@{e['version']}" for e in evidence)
    (checkout / 'cmsis-litert.csolution.yml').write_text('''solution:
  created-for: CMSIS-Toolbox@2.14.1
  compiler: GCC
  packs:
''' + selections + '''
    - pack: ARM::CMSIS
    - pack: ARM::CMSIS-Compiler@^2.1.0
    - pack: ARM::SSE_320_BSP@1.0.0
  target-types:
    - type: SSE-320-CPU
      board: SSE-320
      device: SSE-320-FVP
  build-types:
    - type: Debug
      debug: on
      optimize: debug
  projects:
    - project: ./cmsis-litert.cproject.yml
''')
    (checkout / 'cmsis-litert.cproject.yml').write_text('''project:
  description: Generated float32 ADD test using Reference kernels
  connections:
    - connect: ADD test
      consumes:
        - STDOUT
        - Heap: 65536
  misc:
    - for-compiler: GCC
      CPP:
        - -fno-exceptions
        - -fno-rtti
      Link:
        - -Wl,-u,_printf_float
  components:
    - component: tensorflow::Data Exchange:Serialization:flatbuffers
    - component: tensorflow::Data Processing:Math:gemmlowp fixed-point
    - component: tensorflow::Data Processing:Math:kissfft
    - component: tensorflow::Data Processing:Math:ruy
    - component: tensorflow::Machine Learning:TensorFlow:Kernel&Reference
    - component: tensorflow::Machine Learning:TensorFlow:Kernel Utils
  groups:
    - group: Generated ADD test
      files:
        - file: ./AddTest/main.cpp
  layers:
    - layer: ./board/Corstone-320/Board-U85.clayer.yml
      type: Board
  output:
    type: [elf, map]
''')
