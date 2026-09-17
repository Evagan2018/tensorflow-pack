"""Regression checks for candidate selection without a CMSIS tool installation."""
import contextlib
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
            self.archive(name)
        for filename in ['cmsis-litert.csolution.yml', 'Model/model.clayer.yml',
                         'board/Corstone-320/Board-U85.clayer.yml']:
            path = self.checkout / filename
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('solution:\n  packs:\n' + ''.join(
                f'    - pack: {name}@1.26.2 # retain comment\n'
                for name in sorted(candidate.EXPECTED)) +
                '    - pack: ARM::CMSIS-NN@8.0.0\n')

    def archive(self, name, version='1.26.8-rc3'):
        vendor, pack = name.split('::')
        with ZipFile(self.packs / f'{vendor}.{pack}.pack', 'w') as archive:
            archive.writestr(f'{vendor}.{pack}.pdsc',
                f'<package><vendor>{vendor}</vendor><name>{pack}</name>'
                f'<releases><release version="{version}"/></releases></package>')

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


if __name__ == '__main__':
    unittest.main()
