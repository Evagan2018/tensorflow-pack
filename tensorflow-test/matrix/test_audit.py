import unittest
from binary_audit import audit


class AuditTests(unittest.TestCase):
    def test_mve_must_be_in_nn_kernel_not_unrelated_function(self):
        scalar = '00000100 <arm_convolve_s8>:\n 100: adds r0, r1\n'
        vector = '00000120 <arm_nn_mat_mult_nt_t_s8>:\n 120: vldrb.u8 q0, [r0]\n'
        self.assertTrue(audit(scalar, False)['cmsis_nn_symbols'])
        self.assertTrue(audit(scalar + vector, True)['mve_instructions'])
        for text, mve in [(scalar, True), (vector, False), ('00000120 <unrelated>:\n 120: vldrb.u8 q0, [r0]\n' + scalar, True)]:
            with self.assertRaises(ValueError): audit(text, mve)


if __name__ == '__main__': unittest.main()
