from __future__ import annotations

import numpy as np
from scipy.linalg import eigh


def symmetrize(matrix):
    """Return the symmetric part of a square matrix."""
    matrix = np.asarray(matrix, dtype=float)
    return 0.5 * (matrix + matrix.T)


def top_eigpairs_symmetric(matrix, n_components):
    """Return leading eigenpairs of a symmetric matrix in descending order."""
    matrix = symmetrize(matrix)
    p = matrix.shape[0]
    k = int(n_components)
    if matrix.shape != (p, p):
        raise ValueError("matrix must be square.")
    if not (1 <= k <= p):
        raise ValueError(f"Need 1 <= n_components <= p, got {k} and p={p}.")

    values, vectors = eigh(matrix, subset_by_index=[p - k, p - 1])
    order = np.argsort(values)[::-1]
    return np.asarray(values[order], dtype=float), np.asarray(vectors[:, order], dtype=float)


def projector_from_basis(directions):
    """Return the orthogonal projector spanned by orthonormal columns."""
    directions = np.asarray(directions, dtype=float)
    return directions @ directions.T

