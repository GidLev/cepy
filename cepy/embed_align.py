import numpy as np
from collections.abc import Iterable
import copy

def align(base_ce, target_ce, base_index = 0, target_indices ='all'):
    '''
    Aligned connectome embeddings originated from independent fitting iteration

    Parameters
    ----------
    base_ce : CE
        Containes the latent space for which all target connectome embeddings would be aligned to
    target_ce : CE
        The connectome embeddings to be aligned
    base_index : int, optional
        The index of the connectome embedding iteration within base_ce
    target_indices : str or list
        Index of the connectome embedding within target_ce to be aligned. if set to 'all' then all available fitting iteration are aligned.

    Examples
    --------
    >>> #Load, align and measure the similarity among two connectome embedding:
    >>> import numpy as np
    >>> import cepy as ce
    >>> ce_subject1 = ce.get_example('ce_subject1')
    >>> ce_subject2 = ce.get_example('ce_subject2')
    >>> sim = ce.similarity(ce_subject1, ce_subject2, method='cosine_similarity')
    >>> diagonal_indices = np.diag_indices(sim.shape[0])
    >>> '%.8f' % sim[diagonal_indices].mean()  # measure the similarity among all corresponding nodes across subjects
    '0.57424025'
    >>> # now we repeat the process but first align the two:
    >>> ce_group = ce.get_example('ce_group')
    >>> ce_subject1_aligned = ce.align(ce_group, ce_subject1)
    >>> ce_subject2_aligned = ce.align(ce_group, ce_subject2)
    >>> sim = ce.similarity(ce_subject1_aligned,ce_subject2_aligned,method='cosine_similarity')
    >>> '%.8f' % sim[diagonal_indices].mean()
    '0.79352460'
    '''

    aligned_target_ce = copy.deepcopy(target_ce)
    # compute the inverse of the base's W'
    transformation_to_base = np.linalg.pinv(base_ce.weights.w_apos[base_index]).T

    if target_indices == 'all':
        target_indices = np.arange(len(base_ce.weights.w))
    elif type(target_indices) == int:
        target_indices = [target_indices]
    if not isinstance(target_indices, Iterable):
        raise TypeError('The "targets_index" argument must be set to "all" or int or list of integers')

    for target_index in target_indices:
        aligned_target_ce.weights.w[target_index] = np.dot(transformation_to_base,
                                                           np.dot(target_ce.weights.w_apos[target_index].T,
                                                                  target_ce.weights.w[target_index].T)).T
        aligned_target_ce.weights.w_apos[target_index] = base_ce.weights.w_apos[base_index]
    return aligned_target_ce





