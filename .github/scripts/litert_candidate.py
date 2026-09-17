"""Select candidate packs in an external LiteRT checkout and verify resolution."""
import argparse
import hashlib
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from zipfile import ZipFile

import yaml

EXPECTED = {
    'tensorflow::tensorflow-lite-micro', 'tensorflow::flatbuffers',
    'tensorflow::gemmlowp', 'tensorflow::kissfft', 'tensorflow::ruy',
    'ARM::ethos-u-core-driver',
}


def prepare(packs, checkout):
    versions, evidence = {}, []
    for archive in sorted(packs.glob('*.pack')):
        with ZipFile(archive) as zipped:
            descriptors = [n for n in zipped.namelist() if n.endswith('.pdsc')]
            if len(descriptors) != 1:
                raise ValueError(f'{archive}: expected one PDSC')
            root = ET.fromstring(zipped.read(descriptors[0]))
        name = f'{root.findtext("vendor")}::{root.findtext("name")}'
        if name not in EXPECTED:
            raise ValueError(f'Unexpected candidate pack: {name}')
        release = root.find('releases/release')
        if release is None or not release.get('version'):
            raise ValueError(f'{archive}: missing release version')
        if name in versions:
            raise ValueError(f'Duplicate candidate pack: {name}')
        versions[name] = release.get('version')
        evidence.append({'pack': name, 'version': versions[name],
                         'file': archive.name,
                         'sha256': hashlib.sha256(archive.read_bytes()).hexdigest()})
    if set(versions) != EXPECTED:
        raise ValueError(f'Missing candidate packs: {sorted(EXPECTED - set(versions))}')
    if len(set(versions.values())) != 1:
        raise ValueError(f'Mixed candidate release versions: {versions}')

    # Patch only pack selections, preserving comments and all other YAML content.
    files = ['cmsis-litert.csolution.yml', 'Model/model.clayer.yml',
             'board/Corstone-320/Board-U85.clayer.yml']
    seen = set()
    pattern = re.compile(r'(?m)^(\s*- pack:\s*)([^\s@]+)(?:@[^\s#]+)?')
    for filename in files:
        path = checkout / filename
        def replace(match):
            name = match[2]
            if name not in versions:
                return match[0]
            seen.add(name)
            return f'{match[1]}{name}@{versions[name]}'
        path.write_text(pattern.sub(replace, path.read_text()), encoding='utf-8')
    if seen != EXPECTED:
        raise ValueError(f'Candidate selections missing in checkout: {EXPECTED - seen}')
    (checkout / 'candidate-packs.json').write_text(
        json.dumps(evidence, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(evidence, indent=2))


def verify(checkout):
    evidence = json.loads((checkout / 'candidate-packs.json').read_text())
    expected = {item['pack']: item['version'] for item in evidence}
    document = yaml.safe_load((checkout / 'cmsis-litert.cbuild-pack.yml').read_text())
    actual = {}
    for entry in document['cbuild-pack']['resolved-packs']:
        name, separator, version = entry['resolved-pack'].partition('@')
        if name in EXPECTED:
            if name in actual:
                raise ValueError(f'Multiple resolved versions for {name}')
            actual[name] = version if separator else ''
    if actual != expected:
        raise ValueError(f'Candidate resolution mismatch: expected {expected}, got {actual}')
    print('Verified all six candidate pack versions:', actual)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['prepare', 'verify'])
    parser.add_argument('--checkout', type=Path, required=True)
    parser.add_argument('--packs', type=Path)
    args = parser.parse_args()
    if args.mode == 'prepare':
        if args.packs is None:
            parser.error('prepare requires --packs')
        prepare(args.packs, args.checkout)
    else:
        verify(args.checkout)
