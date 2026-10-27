"""Select candidate packs in an external LiteRT checkout and verify resolution."""
import argparse
import hashlib
import json
import re
import xml.etree.ElementTree as ET
import urllib.request
from pathlib import Path
from zipfile import ZipFile

import yaml

LEGACY_DRIVER = 'ARM::ethos-u-core-driver'
DRIVER = 'ARM::CMSIS-Ethos-U'
DRIVER_LOCK = Path(__file__).with_name('cmsis_ethos_u.json')

TF_PACKS = {
    'tensorflow::tensorflow-lite-micro', 'tensorflow::flatbuffers',
    'tensorflow::gemmlowp', 'tensorflow::kissfft', 'tensorflow::ruy',
}
EXPECTED = TF_PACKS | {DRIVER}


def fetch_driver(destination):
    lock = json.loads(DRIVER_LOCK.read_text())
    destination.mkdir(parents=True, exist_ok=True)
    target = destination / f"ARM.CMSIS-Ethos-U.{lock['version']}.pack"
    data = urllib.request.urlopen(lock['url'], timeout=120).read()
    if hashlib.sha256(data).hexdigest() != lock['sha256']:
        raise ValueError('CMSIS-Ethos-U SHA256 mismatch')
    target.write_bytes(data)
    return target



def prepare(packs, checkout):
    versions, evidence = {}, []
    driver_lock = json.loads(DRIVER_LOCK.read_text())
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
        digest = hashlib.sha256(archive.read_bytes()).hexdigest()
        if name == DRIVER and (versions[name] != driver_lock['version'] or
                               digest != driver_lock['sha256']):
            raise ValueError('CMSIS-Ethos-U version or SHA256 mismatch')
        if name == 'tensorflow::tensorflow-lite-micro':
            requirements = root.findall('requirements/packages/package')
            if any(p.get('name') == 'ethos-u-core-driver' for p in requirements):
                raise ValueError('TensorFlow PDSC still requires legacy driver; rebuild candidate')
            if not any(p.get('vendor') == 'ARM' and p.get('name') == 'CMSIS-Ethos-U'
                       and p.get('version') == driver_lock['version'] + ':' + driver_lock['version']
                       for p in requirements):
                raise ValueError('TensorFlow PDSC must require the pinned CMSIS-Ethos-U')
        evidence.append({'pack': name, 'version': versions[name],
                         'file': archive.name,
                         'sha256': digest})
    if set(versions) != EXPECTED:
        raise ValueError(f'Missing candidate packs: {sorted(EXPECTED - set(versions))}')
    if len({versions[name] for name in TF_PACKS}) != 1:
        raise ValueError(f'Mixed candidate release versions: {versions}')

    # Patch only pack selections, preserving comments and all other YAML content.
    files = ['cmsis-litert.csolution.yml', 'Model/model.clayer.yml',
             'board/Corstone-320/Board-U85.clayer.yml']
    seen = set()
    pattern = re.compile(r'(?m)^(\s*- pack:\s*)([^\s@]+)(?:@[^\s#]+)?')
    updates = {}
    for filename in files:
        path = checkout / filename
        def replace(match):
            name = DRIVER if match[2] == LEGACY_DRIVER else match[2]
            if name not in versions:
                return match[0]
            seen.add(name)
            return f'{match[1]}{name}@{versions[name]}'
        updates[path] = pattern.sub(replace, path.read_text())
    if seen != EXPECTED:
        raise ValueError(f'Candidate selections missing in checkout: {EXPECTED - seen}')
    board = checkout / files[2]
    text = updates[board]
    if 'ETHOSU_MACS' not in text:
        if '    - ETHOSU85' not in text:
            raise ValueError('Expected U85 board definition is missing')
        text = text.replace('    - ETHOSU85', '    - ETHOSU_MACS: 256')
    if not re.search(r'(?m)^    - ETHOSU_MACS: 256\s*$', text):
        raise ValueError('U85 board must use ETHOSU_MACS: 256')
    updates[board] = text
    for path, text in updates.items():
        path.write_text(text, encoding='utf-8')
    (checkout / 'candidate-packs.json').write_text(
        json.dumps(evidence, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(evidence, indent=2))


def verify(checkout):
    evidence = json.loads((checkout / 'candidate-packs.json').read_text())
    expected = {item['pack']: item['version'] for item in evidence}
    if set(expected) != EXPECTED or len(evidence) != len(EXPECTED):
        raise ValueError('Invalid candidate evidence')
    document = yaml.safe_load((checkout / 'cmsis-litert.cbuild-pack.yml').read_text())
    actual = {}
    for entry in document['cbuild-pack']['resolved-packs']:
        name, separator, version = entry['resolved-pack'].partition('@')
        if name == LEGACY_DRIVER:
            raise ValueError('Legacy driver resolved alongside CMSIS-Ethos-U')
        if name in EXPECTED:
            if name in actual:
                raise ValueError(f'Multiple resolved versions for {name}')
            actual[name] = version if separator else ''
    if actual != expected:
        raise ValueError(f'Candidate resolution mismatch: expected {expected}, got {actual}')
    print('Verified five TensorFlow candidates and the independent driver:', actual)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['prepare', 'verify', 'fetch-driver'])
    parser.add_argument('--checkout', type=Path)
    parser.add_argument('--packs', type=Path)
    args = parser.parse_args()
    if args.mode == 'fetch-driver':
        if args.packs is None:
            parser.error('fetch-driver requires --packs')
        fetch_driver(args.packs)
    elif args.checkout is None:
        parser.error('prepare/verify require --checkout')
    elif args.mode == 'prepare':
        if args.packs is None:
            parser.error('prepare requires --packs')
        prepare(args.packs, args.checkout)
    else:
        verify(args.checkout)
