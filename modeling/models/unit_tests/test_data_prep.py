"""Unit tests for modeling/models/data_prep.py."""

import unittest
import numpy as np
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modeling.models.data_prep import (
    build_row_normalized_adjacency,
    build_expected_message_matrix,
)


# ===========================================================================
# Tests
# ===========================================================================

class TestAdjacencyMatrices(unittest.TestCase):

    def test_normalized_adjacency_is_expected_message_matrix(self):
        """Test that the row-normalized adjacency matrix is the same as the expected message matrix."""
        nbrs = {
            0: [1, 2],
            1: [0],
            2: [0, 1],
        }
        n = 3
        A = build_row_normalized_adjacency(nbrs, n)
        M = build_expected_message_matrix(nbrs, n)
        np.testing.assert_allclose(A, M, atol=1e-6)

        # Test with a bigger random graph
        rng = np.random.default_rng(42)
        n = 20
        nbrs = {i: rng.choice([j for j in range(n) if j != i], size=rng.integers(1, n//4), replace=False).tolist() for i in range(n)}
        A = build_row_normalized_adjacency(nbrs,n)
        M = build_expected_message_matrix(nbrs,n)
        np.testing.assert_allclose(A, M, atol=1e-6)

        """Repeat for random graphs"""
        for _ in range(5):
            n = rng.integers(10, 50)
            nbrs = {i: rng.choice([j for j in range(n) if j != i], size=rng.integers(1, n//4), replace=False).tolist() for i in range(n)}
            A = build_row_normalized_adjacency(nbrs,n)
            M = build_expected_message_matrix(nbrs,n)
            np.testing.assert_allclose(A, M, atol=1e-6)


if __name__ == '__main__':
    unittest.main()