"""Check the linked image contains CMSIS-NN kernels and M55 MVE instructions."""
import argparse
import json
import re
import subprocess
from pathlib import Path


def audit(disassembly, require_mve):
    functions = {}
    current = None
    for line in disassembly.splitlines():
        match = re.match(r'^[0-9a-f]+ <([^>]+)>:', line)
        if match:
            current = match[1]
            if current.startswith('arm_'): functions[current] = []
        elif current in functions:
            functions[current].append(line)
    kernels = {n: lines for n, lines in functions.items()
               if re.match(r'arm_(convolve|depthwise_conv|fully_connected|nn_|elementwise|avgpool|max_pool)', n)}
    if not kernels: raise ValueError('No CMSIS-NN kernel symbols in the executable')
    # These are integer MVE vector load/store and arithmetic mnemonics, not scalar VFP.
    pattern = r'\b(vldrb|vldrh|vldrw|vstrb|vstrh|vstrw|vctp|vmladav|vmlaldav|vaddv|vqrdmulh|vqdmladh)[a-z0-9.]*\b'
    mve = {n: [l.strip() for l in lines if re.search(pattern, l)] for n, lines in kernels.items()}
    mve = {n: lines for n, lines in mve.items() if lines}
    if require_mve and not mve: raise ValueError('No integer Helium instructions found in linked CMSIS-NN kernels')
    if not require_mve and mve: raise ValueError('MVE instructions found in Cortex-M4 image')
    return dict(cmsis_nn_symbols=sorted(kernels), mve_instructions=mve,
                scope='Static linked-code evidence; runtime CPUID/MVE check and numerical invocation are separate checks')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkout', type=Path, required=True)
    args = p.parse_args()
    root = args.checkout
    manifest = json.loads((root / 'OperatorTest/manifest.json').read_text())
    images = list(root.glob('out/**/*.elf')) + list(root.glob('out/**/*.axf'))
    if len(images) != 1: raise ValueError('Expected exactly one built executable')
    disassembly = subprocess.check_output(['arm-none-eabi-objdump', '-d', str(images[0])], text=True)
    (root / 'binary-disassembly.txt').write_text(disassembly)
    result = audit(disassembly, manifest['matrix']['mve'])
    (root / 'binary-audit.json').write_text(json.dumps(result, indent=2))
