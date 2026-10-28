import unittest
from run_fvp import verdict


class VerdictTests(unittest.TestCase):
    def setUp(self):
        self.log = ''.join(f'ADD_VALUE:{i}:actual=0:expected=0\n' for i in range(4))
        self.log += 'ADD_CASE:add_float32_reference:PASS\nADD_SUMMARY:1/1\n'

    def test_complete_pass(self):
        self.assertTrue(verdict(self.log, 0))

    def test_incomplete_duplicate_wrong_case_and_fail_are_rejected(self):
        for log in ['', self.log.replace('ADD_VALUE:3:', 'OTHER:3:'),
                    self.log + self.log, self.log.replace('reference', 'other'),
                    self.log.replace(':PASS', ':FAIL'),
                    self.log.replace('1/1', '0/0'),
                    self.log + 'ADD_REASON:numerical mismatch\n']:
            with self.subTest(log=log):
                self.assertFalse(verdict(log, 0))

    def test_process_failure_and_timeout_override_pass_marker(self):
        self.assertFalse(verdict(self.log, 1))
        self.assertFalse(verdict(self.log, 0, timed_out=True))


if __name__ == '__main__':
    unittest.main()
