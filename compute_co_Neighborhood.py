from __future__ import annotations

import numpy as np

from compute_Function import functions_matrix


def compute_relation(
    C: np.ndarray,
    beta: float,
    function: str = "O_mp",
    p: float = 1,
) -> np.ndarray:
    """Construct the symmetric fuzzy beta-covering relation matrix."""
    C = np.asarray(C)
    if C.ndim != 2:
        raise ValueError("C must be a two-dimensional array")

    n_samples = C.shape[0]
    relation = np.zeros((n_samples, n_samples), dtype="float32")
    effective_beta = float(beta)
    critical_beta = float(np.max(np.min(C, axis=1)))
    if critical_beta >= effective_beta:
        effective_beta = critical_beta + 1e-8

    for sample_index in range(n_samples):
        selected_attributes = C[sample_index] < effective_beta
        relation[sample_index] = 1 - np.max(
            C[:, selected_attributes], axis=1
        )

    np.fill_diagonal(relation, 1)

    _, inverse, counts = np.unique(
        C, axis=0, return_inverse=True, return_counts=True
    )
    for group_id in np.flatnonzero(counts > 1):
        sample_indices = np.flatnonzero(inverse == group_id)
        relation[np.ix_(sample_indices, sample_indices)] = 1

    similarity = functions_matrix(function, p, relation)
    np.fill_diagonal(similarity, 1)
    return similarity


__all__ = ["compute_relation"]
