"""Generate dedicated Cortex_DFP targets; identify the actual core at runtime."""
import json
import shutil
from pathlib import Path
import yaml
from profile import TARGETS, COMPILERS


def configure(root, evidence, target, compiler):
    if target == 'M85-U85':
        from consumer_u85 import configure as configure_u85
        return configure_u85(root, evidence, compiler)
    spec = TARGETS[target]
    source = Path(__file__).resolve().parents[1]
    board = root / 'MatrixBoard'
    board.mkdir(exist_ok=True)
    shutil.copyfile(root / 'board/Corstone-320/retarget_stdio.c', board / 'retarget_stdio.c')
    regions = (source / f"Target/{spec['folder']}/RTE/Device/{spec['device']}/regions_{spec['device']}.h").read_text()
    regions = regions.replace('__STACK_SIZE 0x00000400', '__STACK_SIZE 0x00004000')
    if spec['mve']:
        # Dedicated simulation configuration: 2 MiB ITCM, 256 KiB DTCM,
        # bit-28 secure aliases. Keep the linker map and FVP model in agreement.
        regions = regions.replace('__ROM0_SIZE 0x00070000', '__ROM0_SIZE 0x00200000')
    (board / 'regions.h').write_text(regions)
    config = (source / f"Target/{spec['folder']}/fvp_config.txt").read_text()
    # Use linker-owned heap/stack. Do not override them with old semihosting ranges.
    config = '\n'.join(l for l in config.splitlines() if not any(s in l for s in ('heap_', 'stack_')))
    if spec['mve']:
        config = ('cpu0.semihosting-enable=1\n'
                  'cpu0.CFGMEMALIAS=16\ncpu0.CFGITCMSZ=12\ncpu0.CFGDTCMSZ=9\n'
                  'cpu0.MVE=2\ncpu0.FPU=1\n'
                  'fvp_mps2.telnetterminal0.start_telnet=0\n'
                  'fvp_mps2.mps2_visualisation.disable-visualisation=1')
    else:
        config = config.replace('armcortexm4ct.vfp-present=0', 'armcortexm4ct.vfp-present=1')
    (board / 'fvp_config.txt').write_text(config + '\n')
    startup = r'''#include <stdio.h>
#include "RTE_Components.h"
#include CMSIS_device_header
extern int app_main(void);
extern int stdio_init(void);
#ifndef __ARM_FEATURE_MVE
#define __ARM_FEATURE_MVE 0
#endif
int main(void) {
  stdio_init();
  unsigned part = (SCB->CPUID >> 4) & 0xfff;
  unsigned mve = __ARM_FEATURE_MVE;
  printf("MATRIX_TARGET:part=%03x:mve=%u\n", part, mve);
  if (part != EXPECTED_PART || ((mve & 1) != EXPECTED_MVE)) {
    printf("OPS_FATAL:target-identity\n"); return 1;
  }
  return app_main();
}
'''
    (board / 'main.c').write_text(startup.replace('EXPECTED_PART', str(spec['part']))
                                .replace('EXPECTED_MVE', str(int(spec['mve']))))
    solution = yaml.safe_load((root / 'cmsis-litert.csolution.yml').read_text())
    sol = solution['solution']
    sol['compiler'] = compiler + '@' + COMPILERS[compiler]
    sol['packs'] = [dict(pack=e['pack'] + '@' + e['version']) for e in evidence] + [
        dict(pack='ARM::CMSIS@6.4.0'), dict(pack='ARM::CMSIS-Compiler@2.1.0'),
        dict(pack='ARM::CMSIS-NN@7.0.0'), dict(pack='ARM::Cortex_DFP@1.2.0')]
    sol['target-types'] = [dict(type=target, device=spec['device'])]
    sol['build-types'] = [dict(type='Release', optimize='size', debug='on')]
    (root / 'cmsis-litert.csolution.yml').write_text(yaml.safe_dump(solution, sort_keys=False))
    project = yaml.safe_load((root / 'cmsis-litert.cproject.yml').read_text())
    proj = project['project']
    proj['description'] = 'Full int8 operator fixtures using the CMSIS-NN kernel variant'
    proj.pop('connections', None)
    proj.pop('layers', None)
    for component in proj['components']:
        component['component'] = component['component'].replace('Kernel&Reference', 'Kernel&CMSIS-NN')
    proj['components'] += [dict(component=c) for c in ['CMSIS:CORE', 'Device:Startup&C Startup',
        'CMSIS:NN Lib', 'CMSIS-Compiler:CORE', 'CMSIS-Compiler:STDOUT:Custom',
        'CMSIS-Compiler:STDERR:Custom', 'CMSIS-Compiler:STDIN:Custom']]
    proj['groups'].append(dict(group='Matrix board', files=[dict(file='./MatrixBoard/main.c'),
                                                           dict(file='./MatrixBoard/retarget_stdio.c')]))
    proj['linker'] = [dict(regions='./MatrixBoard/regions.h')]
    proj['misc'] += [{'for-compiler': c, 'CPP': ['-fno-exceptions', '-fno-rtti', '-fno-fast-math']} for c in ['AC6', 'CLANG']]
    # FVP starts at the ELF entry. AC6 otherwise selects __main and bypasses
    # SystemInit, causing a NOCP fault at the C library's first FP instruction.
    next(m for m in proj['misc'] if m['for-compiler'] == 'AC6')['Link'] = ['--entry=Reset_Handler']
    (root / 'cmsis-litert.cproject.yml').write_text(yaml.safe_dump(project, sort_keys=False))
    manifest_path = root / 'OperatorTest/manifest.json'
    manifest = json.loads(manifest_path.read_text())
    manifest['matrix'] = dict(target=target, **spec, compiler=compiler, compiler_package=COMPILERS[compiler],
        input_format='int8', kernel_variant='CMSIS-NN', comparison='exact integer equality; zero LSB tolerance',
        acceleration_scope='CMSIS-NN kernel variant; operators without optimized implementations use its fallback kernels')
    manifest['fvp'] = dict(executable=spec['model'], config='MatrixBoard/fvp_config.txt')
    manifest_path.write_text(json.dumps(manifest, indent=2))
