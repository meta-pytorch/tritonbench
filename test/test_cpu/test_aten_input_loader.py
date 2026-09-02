import unittest
from types import SimpleNamespace

import torch
from tritonbench.operator_loader.aten.input_loader import (
    deserialize_args,
    deserialize_sparse_tensor,
    OperatorInputLoader,
    serialize_sparse_tensor,
)


class AtenInputLoaderTest(unittest.TestCase):
    def test_hybrid_sparse_tensor_round_trip(self):
        original = torch.sparse_coo_tensor(
            torch.tensor([[0, 2, 4]]),
            torch.randn(3, 4),
            (5, 4),
            check_invariants=False,
        ).coalesce()

        serialized = repr(serialize_sparse_tensor(original))
        (restored,), _ = deserialize_args(f"(({serialized},), {{}})")

        self.assertEqual(restored.shape, original.shape)
        self.assertEqual(restored.dtype, original.dtype)
        self.assertEqual(restored.layout, original.layout)
        self.assertEqual(restored.sparse_dim(), original.sparse_dim())
        self.assertEqual(restored.dense_dim(), original.dense_dim())
        self.assertEqual(restored._nnz(), original._nnz())
        self.assertTrue(restored.is_coalesced())

    def test_pure_sparse_tensor_round_trip(self):
        original = torch.sparse_coo_tensor(
            torch.tensor([[0, 1, 2], [1, 0, 3]]),
            torch.randn(3),
            (3, 4),
            check_invariants=False,
        )

        serialized = repr(serialize_sparse_tensor(original))
        (restored,), _ = deserialize_args(f"(({serialized},), {{}})")

        self.assertEqual(restored.shape, original.shape)
        self.assertEqual(restored.sparse_dim(), 2)
        self.assertEqual(restored.dense_dim(), 0)
        self.assertEqual(restored._nnz(), original._nnz())
        self.assertFalse(restored.is_coalesced())

    def test_legacy_payload_defaults_to_one_sparse_dimension(self):
        (restored,), _ = deserialize_args(
            "((ST([5, 4], f32, torch.sparse_coo, False, 3),), {})"
        )

        self.assertEqual(restored.shape, (5, 4))
        self.assertEqual(restored.sparse_dim(), 1)
        self.assertEqual(restored.dense_dim(), 1)
        self.assertEqual(restored._nnz(), 3)

    def test_empty_sparse_tensor_without_nnz(self):
        restored = deserialize_sparse_tensor(
            [0, 4], torch.float32, torch.sparse_coo, True, sparse_dim=1
        )

        self.assertEqual(restored.shape, (0, 4))
        self.assertEqual(restored._nnz(), 0)
        self.assertTrue(restored.is_coalesced())

    def test_rejects_impossible_coalesced_nnz(self):
        with self.assertRaisesRegex(ValueError, "coalesced tensor cannot have"):
            deserialize_sparse_tensor(
                [2, 2],
                torch.float32,
                torch.sparse_coo,
                True,
                nnz=5,
                sparse_dim=2,
            )

    def test_rejects_unsupported_sparse_layout(self):
        with self.assertRaisesRegex(ValueError, "Unsupported sparse tensor layout"):
            deserialize_sparse_tensor(
                [2, 2],
                torch.float32,
                torch.sparse_csr,
                False,
                nnz=1,
                sparse_dim=2,
            )

        with self.assertRaisesRegex(ValueError, "Unsupported sparse tensor layout"):
            serialize_sparse_tensor(SimpleNamespace(layout=torch.sparse_csr))

    def test_embedding_entries_do_not_block_other_operators(self):
        input_config = {
            "aten.add.Tensor": [
                {
                    "inputs": "((T([2], f32), T([2], f32)), {})",
                    "count": 1,
                }
            ],
            "aten.embedding.default": [
                {
                    "inputs": "((T([4, 2], f32), T([2], i64)), {})",
                    "count": 1,
                }
            ],
        }

        loader = OperatorInputLoader("aten.add.Tensor", input_config)
        inputs = list(loader.get_input_iter()())

        self.assertEqual(len(inputs), 1)

    def test_embedding_operator_remains_rejected(self):
        input_config = {
            "aten.embedding.default": [
                {
                    "inputs": "((T([4, 2], f32), T([2], i64)), {})",
                    "count": 1,
                }
            ]
        }

        with self.assertRaisesRegex(
            RuntimeError, "Embedding inputs not yet implemented"
        ):
            OperatorInputLoader("aten.embedding.default", input_config)


if __name__ == "__main__":
    unittest.main()
