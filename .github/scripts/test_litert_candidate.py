"""Regression checks for candidate selection without a CMSIS tool installation."""
import contextlib
import hashlib
from unittest.mock import patch
import io
import json
import tempfile
import unittest
from pathlib import Path
from zipfile import ZipFile

import litert_candidate as candidate


class CandidateTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.packs = self.root / 'packs'
        self.packs.mkdir()
        self.checkout = self.root / 'litert'
        for name in candidate.EXPECTED:
            self.archive(name, '1.0.0' if name == candidate.DRIVER else '1.26.8-rc3')
        lock = json.loads(candidate.DRIVER_LOCK.read_text())
        lock['sha256'] = hashlib.sha256((self.packs / 'ARM.CMSIS-Ethos-U.pack').read_bytes()).hexdigest()
        self.lock = self.root / 'driver.json'
        self.lock.write_text(json.dumps(lock))
        mock = patch.object(candidate, 'DRIVER_LOCK', self.lock)
        mock.start()
        self.addCleanup(mock.stop)
        for filename in ['cmsis-litert.csolution.yml', 'Model/model.clayer.yml',
                         'board/Corstone-320/Board-U85.clayer.yml']:
            path = self.checkout / filename
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('solution:\n  packs:\n' + ''.join(
                f'    - pack: {candidate.LEGACY_DRIVER if name == candidate.DRIVER else name}@1.26.2 # retain comment\n'
                for name in sorted(candidate.EXPECTED)) +
                '    - pack: ARM::CMSIS-NN@8.0.0\n  define:\n    - ETHOSU85\n')

    def archive(self, name, version='1.26.8-rc3'):
        vendor, pack = name.split('::')
        with ZipFile(self.packs / f'{vendor}.{pack}.pack', 'w') as archive:
            archive.writestr(f'{vendor}.{pack}.pdsc',
                f'<package><vendor>{vendor}</vendor><name>{pack}</name>'
                f'<releases><release version="{version}"/></releases>'
                '<requirements><packages><package vendor="ARM" name="CMSIS-Ethos-U" '
                'version="1.0.0:1.0.0"/></packages></requirements></package>')

    def prepare(self):
        with contextlib.redirect_stdout(io.StringIO()):
            candidate.prepare(self.packs, self.checkout)

    def test_prepare_and_verify(self):
        self.prepare()
        for filename in self.checkout.rglob('*.yml'):
            text = filename.read_text()
            self.assertNotIn('@1.26.2', text)
            self.assertIn('@1.26.8-rc3 # retain comment', text)
            self.assertIn('ARM::CMSIS-NN@8.0.0', text)
            self.assertIn('ARM::CMSIS-Ethos-U@1.0.0', text)
            self.assertNotIn(candidate.LEGACY_DRIVER, text)
        evidence = json.loads((self.checkout / 'candidate-packs.json').read_text())
        resolved = {'cbuild-pack': {'resolved-packs': [
            {'resolved-pack': f'{item["pack"]}@{item["version"]}'} for item in evidence]}}
        lock = self.checkout / 'cmsis-litert.cbuild-pack.yml'
        lock.write_text(json.dumps(resolved))
        with contextlib.redirect_stdout(io.StringIO()):
            candidate.verify(self.checkout)
        resolved['cbuild-pack']['resolved-packs'][0]['resolved-pack'] += '-wrong'
        lock.write_text(json.dumps(resolved))
        with self.assertRaisesRegex(ValueError, 'mismatch'):
            candidate.verify(self.checkout)

    def test_missing_pack(self):
        next(self.packs.glob('*.pack')).unlink()
        with self.assertRaisesRegex(ValueError, 'Missing candidate'):
            self.prepare()

    def test_mixed_versions(self):
        self.archive('tensorflow::ruy', '1.26.2')
        with self.assertRaisesRegex(ValueError, 'Mixed candidate'):
            self.prepare()

    def test_tampered_driver(self):
        self.archive(candidate.DRIVER, '1.0.1')
        with self.assertRaisesRegex(ValueError, 'SHA256 mismatch'):
            self.prepare()

    def test_legacy_archive_rejected(self):
        self.archive(candidate.LEGACY_DRIVER)
        with self.assertRaisesRegex(ValueError, 'Unexpected candidate'):
            self.prepare()

    def test_prepare_is_idempotent(self):
        self.prepare()
        before = {p: p.read_bytes() for p in self.checkout.rglob('*.yml')}
        self.prepare()
        self.assertEqual(before, {p: p.read_bytes() for p in before})

    def test_failed_prepare_does_not_partially_patch(self):
        board = self.checkout / 'board/Corstone-320/Board-U85.clayer.yml'
        board.write_text(board.read_text().replace('ETHOSU85', 'ETHOSU_MACS: 128'))
        before = {p: p.read_bytes() for p in self.checkout.rglob('*.yml')}
        with self.assertRaisesRegex(ValueError, 'MACS'):
            self.prepare()
        self.assertEqual(before, {p: p.read_bytes() for p in before})

    def test_legacy_resolution_rejected(self):
        self.prepare()
        evidence = json.loads((self.checkout / 'candidate-packs.json').read_text())
        entries = [{'resolved-pack': f"{e['pack']}@{e['version']}"} for e in evidence]
        entries.append({'resolved-pack': candidate.LEGACY_DRIVER + '@1.26.8-rc3'})
        (self.checkout / 'cmsis-litert.cbuild-pack.yml').write_text(
            json.dumps({'cbuild-pack': {'resolved-packs': entries}}))
        with self.assertRaisesRegex(ValueError, 'Legacy driver'):
            candidate.verify(self.checkout)

    def test_legacy_pdsc_dependency_rejected(self):
        archive = self.packs / 'tensorflow.tensorflow-lite-micro.pack'
        with ZipFile(archive) as z:
            name = z.namelist()[0]
            pdsc = z.read(name).replace(b'CMSIS-Ethos-U', b'ethos-u-core-driver')
        with ZipFile(archive, 'w') as z:
            z.writestr(name, pdsc)
        with self.assertRaisesRegex(ValueError, 'PDSC still requires legacy'):
            self.prepare()

    def test_duplicate_pack_identity_rejected(self):
        source = self.packs / 'tensorflow.ruy.pack'
        (self.packs / 'duplicate.pack').write_bytes(source.read_bytes())
        with self.assertRaisesRegex(ValueError, 'Duplicate candidate'):
            self.prepare()

    def test_download_hash_mismatch_does_not_write_artifact(self):
        destination = self.root / 'download'
        with patch.object(candidate.urllib.request, 'urlopen') as download:
            download.return_value.read.return_value = b'tampered'
            with self.assertRaisesRegex(ValueError, 'SHA256 mismatch'):
                candidate.fetch_driver(destination)
        self.assertEqual(list(destination.iterdir()), [])


if __name__ == '__main__':
    unittest.main()
