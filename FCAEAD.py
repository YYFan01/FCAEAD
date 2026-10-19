from __future__ import annotations

import numpy as np
import pandas as pd

from compute_co_Neighborhood import compute_relation


def compute_Rc(data: np.ndarray, beta: float, function: str, p: float) -> np.ndarray:
    """Construct the vectorized fuzzy beta-covering relation."""
    return compute_relation(data, beta, function, p)


def _cardinality_without_each_object(
    values: np.ndarray,
    reduction: str,
) -> np.ndarray:
    """Compute leave-one-out fuzzy cardinalities."""
    values = np.asarray(values)
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError("values must be a square two-dimensional matrix")
    n = values.shape[0]
    if n < 2:
        raise ValueError("at least two objects are required")

    row_indices = np.arange(n)
    if reduction == "min":
        extremum_indices = np.argmin(values, axis=1)
        first = values[row_indices, extremum_indices]
        second = np.partition(values, kth=1, axis=1)[:, 1]
    elif reduction == "max":
        extremum_indices = np.argmax(values, axis=1)
        first = values[row_indices, extremum_indices]
        second = np.partition(values, kth=n - 2, axis=1)[:, n - 2]
    else:
        raise ValueError("reduction must be 'min' or 'max'")

    replacement_delta = second - first

    change_by_deleted_column = np.bincount(
        extremum_indices,
        weights=replacement_delta,
        minlength=n,
    )
    sum_after_column_deletion = first.sum() + change_by_deleted_column

    diagonal_was_selected = extremum_indices == row_indices
    deleted_row_value = first + np.where(
        diagonal_was_selected, replacement_delta, 0
    )
    return sum_after_column_deletion - deleted_row_value


def FCAEAD(data: np.ndarray, beta: float, function: str, p: float) -> np.ndarray:
    """Compute FCAEAD anomaly scores."""
    data = np.asarray(data)
    if data.ndim != 2:
        raise ValueError("data must be a two-dimensional array")

    n, m = data.shape
    if n < 2:
        raise ValueError("FCAEAD requires at least two samples")
    if m < 2:
        raise ValueError("FCAEAD requires at least two attributes")

    attributes = np.arange(m)
    approximation_accuracy = np.zeros((n, m), dtype="float32")
    similarity = np.zeros((n, m), dtype="float32")
    weight = np.zeros((n, m), dtype="float32")
    entropy_without_sample = np.zeros((n, m), dtype="float32")

    for attribute_index in range(m):
        remaining_attributes = np.setdiff1d(attributes, attribute_index)

        # Step 1: construct the relation for the current attribute.
        relation_b = compute_Rc(
            data[:, attribute_index].reshape(-1, 1), beta, function, p
        )

        # Step 2: construct the relation for the remaining attributes.
        relation_p = compute_Rc(data[:, remaining_attributes], beta, function, p)
        relation_b_unique, relation_b_inverse, group_counts = np.unique(
            relation_b,
            return_inverse=True,
            return_counts=True,
            axis=0,
        )

        entropy_term_sum_without = np.zeros(n, dtype="float64")

        for group_index, granule in enumerate(relation_b_unique):
            group_sample_indices = np.flatnonzero(relation_b_inverse == group_index)

            # Step 3: compute fuzzy lower and upper approximations.
            lower_values = np.maximum(1 - relation_p, granule)
            lower = lower_values.min(axis=1)
            lower_card_without = _cardinality_without_each_object(
                lower_values, "min"
            )
            del lower_values

            upper_values = np.minimum(relation_p, granule)
            upper = upper_values.max(axis=1)
            upper_card_without = _cardinality_without_each_object(
                upper_values, "max"
            )
            del upper_values

            accuracy = lower.sum() / upper.sum()
            approximation_accuracy[group_sample_indices, attribute_index] = accuracy

            relative_granule_size = granule.sum() / n
            similarity[group_sample_indices, attribute_index] = relative_granule_size

            # Step 4: compute the weight.
            weight[group_sample_indices, attribute_index] = relative_granule_size ** (1 / 3)

            # Step 5: compute the leave-one-out approximation entropy.
            remaining_multiplicity = np.full(
                n, int(group_counts[group_index]), dtype=np.int64
            )
            remaining_multiplicity[group_sample_indices] -= 1
            valid = remaining_multiplicity > 0

            granule_card_without = granule.sum() - granule
            accuracy_without = np.zeros(n, dtype="float32")
            np.divide(
                lower_card_without,
                upper_card_without,
                out=accuracy_without,
                where=valid,
            )
            similarity_without = np.zeros(n, dtype="float32")
            np.divide(
                granule_card_without,
                n - 1,
                out=similarity_without,
                where=valid,
            )
            entropy_terms = np.zeros(n, dtype="float32")
            entropy_terms[valid] = accuracy_without[valid] * np.log2(
                similarity_without[valid] + 1e-10
            )

            entropy_term_sum_without += remaining_multiplicity * entropy_terms

        entropy_without_sample[:, attribute_index] = np.asarray(
            -entropy_term_sum_without / (n - 1), dtype="float32"
        )

    # Step 6: compute approximation entropy and relative entropy.
    generalized_entropy = approximation_accuracy * np.log2(similarity + 1e-10)
    entropy = -np.mean(generalized_entropy, axis=0, keepdims=True)
    relative_entropy = entropy_without_sample / (entropy + 1e-10) + 1 / n

    # Step 7: compute anomaly scores.
    return np.asarray(2 - np.mean(relative_entropy * weight, axis=1))


__all__ = ["FCAEAD", "compute_Rc"]

if __name__ == "__main__":
    data = pd.read_excel("Example.xlsx", header=None).values.astype(float)
    beta = 0.5
    function = "O_mp"
    p = 2
    print(FCAEAD(data, beta, function, p))
