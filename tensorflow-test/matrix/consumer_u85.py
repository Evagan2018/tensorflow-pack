"""Corstone-320 consumer: Ethos-U kernels with CMSIS-NN CPU fallback."""
import json
import yaml
from profile import TARGETS, COMPILERS


def configure(root, evidence, compiler):
    spec = TARGETS['M85-U85']
    path = root / 'cmsis-litert.csolution.yml'
    document = yaml.safe_load(path.read_text())
    sol = document['solution']
    sol['compiler'] = compiler + '@' + COMPILERS[compiler]
    sol['packs'] = [dict(pack=e['pack'] + '@' + e['version']) for e in evidence] + [
        dict(pack='ARM::CMSIS@6.4.0'), dict(pack='ARM::CMSIS-Compiler@2.1.0'),
        dict(pack='ARM::CMSIS-NN@7.0.0'), dict(pack='ARM::SSE_320_BSP@1.0.0')]
    sol['target-types'] = [dict(type='M85-U85', board='SSE-320', device=spec['device'])]
    sol['build-types'] = [dict(type='Release', optimize='size', debug='on')]
    path.write_text(yaml.safe_dump(document, sort_keys=False))
    path = root / 'cmsis-litert.cproject.yml'
    document = yaml.safe_load(path.read_text())
    proj = document['project']
    proj['description'] = 'Int8 fixtures compiled for Ethos-U85 with CMSIS-NN CPU fallback'
    for component in proj['components']:
        component['component'] = component['component'].replace('Kernel&Reference', 'Kernel&Ethos-U')
    proj['components'].append(dict(component='ARM::CMSIS:NN Lib'))
    proj['define'] = ['OPS_USE_NPU']
    proj['misc'] += [{'for-compiler': c, 'CPP': ['-fno-exceptions', '-fno-rtti', '-fno-fast-math']}
                     for c in ['AC6', 'CLANG']]
    next(m for m in proj['misc'] if m['for-compiler'] == 'AC6')['Link'] = ['--entry=Reset_Handler']
    # Toolbox 2.14.1 otherwise overrides the M85's vector-capable FPU for LLVM.
    clang = next(m for m in proj['misc'] if m['for-compiler'] == 'CLANG')
    clang['C-CPP'] = ['-mfpu=fp-armv8-fullfp16-d16']
    clang['ASM'] = ['-mfpu=fp-armv8-fullfp16-d16']
    path.write_text(yaml.safe_dump(document, sort_keys=False))
    board = root / 'board/Corstone-320'
    path = board / 'main.c'
    startup = path.read_text()
    startup = '#include <stdio.h>\n' + startup
    startup = startup.replace('  return (app_main());', '''  unsigned part = (SCB->CPUID >> 4) & 0xfff;
  printf("MATRIX_TARGET:part=%03x:mve=%u\\n", part, (unsigned)__ARM_FEATURE_MVE);
  if (part != 0xd23 || !(__ARM_FEATURE_MVE & 1)) {
    printf("OPS_FATAL:target-identity\\n"); return 1;
  }
  return (app_main());''')
    path.write_text(startup)
    path = board / 'ethos_setup.c'
    setup = path.read_text()
    setup = setup.replace('void NPU0_Handler(void) {', '''static volatile unsigned ops_npu_irqs;
unsigned ops_npu_irq_count(void) { return ops_npu_irqs; }
void NPU0_Handler(void) {
  ++ops_npu_irqs;''')
    path.write_text(setup)
    path = board / 'regions_SSE-320.h'
    path.write_text(path.read_text().replace('__STACK_SIZE  0x00001000', '__STACK_SIZE  0x00004000'))
    path = root / 'OperatorTest/manifest.json'
    manifest = json.loads(path.read_text())
    manifest['matrix'] = dict(target='M85-U85', **spec, compiler=compiler,
        compiler_package=COMPILERS[compiler], input_format='int8', kernel_variant='Ethos-U',
        comparison='exact integer equality; zero LSB tolerance',
        acceleration_scope='Vela partitions plus per-case NPU completion interrupts; CPU fallback is explicit')
    manifest['fvp'] = dict(executable=spec['model'], config='board/Corstone-320/fvp_config.txt')
    path.write_text(json.dumps(manifest, indent=2))
