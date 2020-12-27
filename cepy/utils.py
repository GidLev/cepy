import numpy as np
import warnings

def normalize(X, norm='l2', axis=1):
    """Scale input vectors individually to unit norm (vector length).

    Parameters
    ----------
    X : {array-like, sparse matrix} of shape (n_samples, n_features)
        The data to normalize, element by element.
        scipy.sparse matrices should be in CSR format to avoid an
        un-necessary copy.
    norm : {'l1', 'l2', 'max'}, default='l2'
        The norm to use to normalize each non zero sample (or each non-zero
        feature if axis is 0).
    axis : {0, 1}, default=1
        axis used to normalize the data along. If 1, independently normalize
        each sample, otherwise (if 0) normalize each feature.

    Returns
    -------
    X : {ndarray, sparse matrix} of shape (n_samples, n_features)
        Normalized input X.


    """
    if norm not in ('l1', 'l2', 'max'):
        raise ValueError("'%s' is not a supported norm" % norm)

    if axis == 0:
        X = X.T


    if norm == 'l1':
        norms = np.abs(X).sum(axis=1)
    elif norm == 'l2':
        norms = row_norms(X)
    elif norm == 'max':
        norms = np.max(abs(X), axis=1)
    norms = _handle_zeros_in_scale(norms, copy=False)
    X /= norms[:, np.newaxis]

    if axis == 0:
        X = X.T

    return X

def row_norms(X, squared=False):
    """Row-wise (squared) Euclidean norm of X.
    Equivalent to np.sqrt((X * X).sum(axis=1)), but also supports sparse
    matrices and does not create an X.shape-sized temporary.
    Performs no input validation.
    Parameters
    ----------
    X : array-like
        The input array.
    squared : bool, default=False
        If True, return squared norms.
    Returns
    -------
    array-like
        The row-wise (squared) Euclidean norm of X.
    """
    norms = np.einsum('ij,ij->i', X, X)
    if not squared:
        np.sqrt(norms, norms)
    return norms

def _handle_zeros_in_scale(scale, copy=True):
    """Makes sure that whenever scale is zero, we handle it correctly.
    This happens in most scalers when we have constant features.
    """

    # if we are fitting on 1D arrays, scale might be a scalar
    if np.isscalar(scale):
        if scale == .0:
            scale = 1.
        return scale
    elif isinstance(scale, np.ndarray):
        if copy:
            # New array to avoid side-effects
            scale = scale.copy()
        scale[scale == 0.0] = 1.0
        return scale


def check_adjacency_matrix(X):
    assert type(X) == np.ndarray, ('Input is expected as a numpy array')
    assert np.all(X >= 0), ('No negative edges allowed in the adjacency matrix')
    assert np.all(X == X.T), ('The adjacency matrix is expected to be symmetric')
    if np.any(X.sum(axis=0) == 0):
        warnings.warn('The input adjacency matrix contains zero connected nodes.')