import unittest
import json
from pathlib import Path
from profile import COMPILERS, TARGETS
from binary_audit import audit


class AuditTests(unittest.TestCase):
    def test_npu_image_requires_driver_but_may_have_no_cpu_kernels(self):
        driver = '00000100 <ethosu_invoke_v3>:\n 100: adds r0, r1\n'
        self.assertTrue(audit(driver, True, npu=True)['ethosu_invoke_symbols'])
        with self.assertRaises(ValueError): audit('', True, npu=True)

    def test_mve_must_be_in_nn_kernel_not_unrelated_function(self):
        scalar = '00000100 <arm_convolve_s8>:\n 100: adds r0, r1\n'
        vector = '00000120 <arm_nn_mat_mult_nt_t_s8>:\n 120: vldrb.u8 q0, [r0]\n'
        self.assertTrue(audit(scalar, False)['cmsis_nn_symbols'])
        self.assertTrue(audit(scalar + vector, True)['mve_instructions'])
        for text, mve in [(scalar, True), (vector, False), ('00000120 <unrelated>:\n 120: vldrb.u8 q0, [r0]\n' + scalar, True)]:
            with self.assertRaises(ValueError): audit(text, mve)


class PlanTests(unittest.TestCase):
    def test_authorized_u85_target_matches_standard_fvp(self):
        root = Path(__file__).parent
        plan = json.loads((root / 'requested-matrix.json').read_text())
        self.assertEqual(plan['toolchains'], COMPILERS)
        ready = {t['id'] for t in plan['targets'] if t['readiness'] == 'implemented'}
        self.assertEqual(ready, set(TARGETS))
        npu = next(t for t in plan['targets'] if t['id'] == 'M85-U85')
        self.assertEqual(npu['model'], TARGETS['M85-U85']['model'])
        self.assertEqual(npu['core'], 'Cortex-M85')
        self.assertNotIn('M55-U85', {t['id'] for t in plan['targets']})


if __name__ == '__main__': unittest.main()
