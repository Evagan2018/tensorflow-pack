import copy
import unittest
from inventory import reconcile
from run_fvp import evaluate


class CoverageTests(unittest.TestCase):
    def test_missing_stale_overlap_and_empty_reason_fail(self):
        rows = [dict(operator='ADD'), dict(operator='WHILE')]
        for recipes, skips in [({'ADD'}, {}), ({'ADD', 'MUL'}, {'WHILE': 'reason'}),
                               ({'ADD'}, {'ADD': 'overlap', 'WHILE': 'reason'}),
                               ({'ADD'}, {'WHILE': ''})]:
            with self.subTest(recipes=recipes, skips=skips):
                with self.assertRaises(ValueError): reconcile(rows, recipes, skips)
        result = reconcile(rows, {'ADD'}, {'WHILE': 'requires two subgraphs'})
        self.assertEqual([r['status'] for r in result], ['planned', 'skipped'])


class VerdictTests(unittest.TestCase):
    def setUp(self):
        self.manifest = dict(cases=[dict(id=op.lower()+'_reference', operator=op,
                                        expected=[dict(shape=[1, 4])]) for op in ('ADD', 'MUL')],
                             coverage=[])
        self.good = ('OPS_CASE:add_reference:PASS:checks=4:max_abs=0:reason=none\n'
                     'OPS_CASE:mul_reference:PASS:checks=4:max_abs=0:reason=none\nOPS_SUMMARY:2/2\n')
        self.bad = self.good.replace('add_reference:PASS', 'add_reference:FAIL').replace(
            'max_abs=0:reason=none', 'max_abs=100:reason=numerical', 1).replace('2/2', '1/2')

    def test_matrix_requires_correct_runtime_core_and_helium(self):
        self.manifest['matrix'] = dict(part=0xd22, mve=True)
        identity = 'MATRIX_TARGET:part=d22:mve=3\n'
        self.assertTrue(evaluate(self.manifest, identity + self.good, 0)['accepted'])
        for header in ['', identity * 2, identity.replace('d22', 'd23'), identity.replace('mve=3', 'mve=0')]:
            self.assertFalse(evaluate(self.manifest, header + self.good, 0)['accepted'])

    def test_mode_rejects_deliberate_corruption_or_wrong_filter_in_positive_run(self):
        self.manifest['protocol_version'] = 2
        mode = 'OPS_MODE:negative_control=0:case_filter=\n'
        self.assertTrue(evaluate(self.manifest, mode + self.good, 0)['accepted'])
        for header in ['', mode * 2, mode.replace('control=0', 'control=1'),
                       mode.replace('case_filter=', 'case_filter=add_reference')]:
            self.assertFalse(evaluate(self.manifest, header + self.good, 0)['accepted'])

    def test_positive_and_deliberate_negative(self):
        self.assertTrue(evaluate(self.manifest, self.good, 0)['numerical_pass'])
        result = evaluate(self.manifest, self.bad, 0, negative=True)
        self.assertTrue(result['accepted'])
        self.assertFalse(result['numerical_pass'])
        self.assertFalse(evaluate(self.manifest, self.bad, 0)['accepted'])
        self.assertFalse(evaluate(self.manifest, self.good, 0, negative=True)['accepted'])

    def test_int8_pass_requires_zero_recorded_error(self):
        self.manifest['matrix'] = dict(part=0xd22, mve=True, input_format='int8')
        log = 'MATRIX_TARGET:part=d22:mve=3\n' + self.good.replace('max_abs=0', 'max_abs=1', 1)
        self.assertFalse(evaluate(self.manifest, log, 0)['accepted'])

    def test_npu_requires_completion_for_each_delegated_case(self):
        self.manifest['matrix'] = dict(part=0xd23, mve=True, npu='Ethos-U85')
        self.manifest['cases'][0]['vela'] = dict(npu_subgraphs=1)
        self.manifest['cases'][1]['vela'] = dict(npu_subgraphs=0)
        header = ('MATRIX_TARGET:part=d23:mve=3\n'
                  'OPS_NPU:add_reference:subgraphs=1:irqs=1\n'
                  'OPS_NPU:mul_reference:subgraphs=0:irqs=0\n')
        self.assertTrue(evaluate(self.manifest, header + self.good, 0)['accepted'])
        for bad in [header.replace('irqs=1', 'irqs=0'), header.replace('subgraphs=1', 'subgraphs=0'),
                    header.replace('subgraphs=0:irqs=0', 'subgraphs=0:irqs=1'),
                    header + 'OPS_NPU:add_reference:subgraphs=1:irqs=1\n',
                    header.replace('OPS_NPU:mul_reference:subgraphs=0:irqs=0\n', '')]:
            self.assertFalse(evaluate(self.manifest, bad + self.good, 0)['accepted'])

    def test_bad_evidence_fails(self):
        for log in [self.good + self.good, self.good.replace('checks=4', 'checks=0'),
                    self.good.replace('add_reference', 'unknown'),
                    self.good.replace('max_abs=0', 'max_abs=nan'),
                    self.good.replace('OPS_SUMMARY:2/2', 'OPS_SUMMARY:0/0'),
                    self.good.replace('OPS_CASE:mul_reference:PASS:checks=4:max_abs=0:reason=none\n', ''),
                    self.good + 'OPS_FATAL:registration\n', self.bad]:
            with self.subTest(log=log):
                self.assertFalse(evaluate(self.manifest, log, 0)['accepted'])
        self.assertFalse(evaluate(self.manifest, self.good, 0, timed_out=True)['accepted'])
        self.assertFalse(evaluate(self.manifest, self.good, 1)['accepted'])

    def test_empty_or_duplicate_manifest_rejected(self):
        for cases in [[], [self.manifest['cases'][0]] * 2]:
            with self.assertRaises(ValueError): evaluate(dict(cases=cases), '', 0)

    def test_negative_control_cannot_hide_other_failure(self):
        self.assertFalse(evaluate(self.manifest, self.bad.replace('mul_reference:PASS', 'mul_reference:FAIL'),
                                  0, negative=True)['accepted'])
        self.assertFalse(evaluate(self.manifest, self.bad, 0, timed_out=True, negative=True)['accepted'])


if __name__ == '__main__':
    unittest.main()
