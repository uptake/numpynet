import numpynet.common as nnc
import unittest
import math
import numpy as np


class TestActivation(unittest.TestCase):
    def test_tanh(self):
        activation = nnc.Activation("tanh")

        # Compare scalars
        self.assertEqual(activation.function(1), math.tanh(1))
        self.assertEqual(activation.function(0), math.tanh(0))

        # Compare vectors
        vect = np.array([1, 2, 3])
        true_values = np.array(
            [0.7615941559557649, 0.9640275800758169, 0.9950547536867305]
        )
        self.assertTrue(np.array_equal(activation.function(vect), true_values))

        # Compare matrices
        matrix = np.array([[0.2, 0.0000001, 3], [-0.5, 1000, -10]])
        true_values = np.array(
            [[0.19737532, 0.0000001, 0.99505475], [-0.46211716, 1.0, -1.0]]
        )
        self.assertTrue(
            np.array_equal(np.round(activation.function(matrix), 8), true_values)
        )

        # Test derivative and scalars
        self.assertEqual(activation._tanh(1, deriv=True), 1 - math.tanh(1) ** 2)
        self.assertEqual(activation._tanh(0, deriv=True), 1 - math.tanh(0) ** 2)

        # Test derivative and vectors
        true_values = np.array(
            [0.41997434161402614, 0.07065082485316443, 0.009866037165440211]
        )
        self.assertTrue(np.array_equal(activation._tanh(vect, deriv=True), true_values))

        # Test derivative and matrices
        true_values = np.array(
            [[0.96104298, 1.0, 0.00986604], [0.78644773, 0.0, 0.00000001]]
        )
        self.assertTrue(
            np.array_equal(
                np.round(activation._tanh(matrix, deriv=True), 8), true_values
            )
        )


if __name__ == "__main__":
    unittest.main()
