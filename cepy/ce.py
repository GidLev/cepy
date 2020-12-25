import random
from collections import defaultdict
import numpy as np
import networkx as nx
from joblib import Parallel, delayed
from tqdm.auto import tqdm
from .parallel import parallel_generate_walks, parallel_learn_embeddings
import tempfile
import shutil
import pickle
import os
import gzip
from cepy.utils import normalize
import warnings

class CE:
    FIRST_TRAVEL_KEY = 'first_travel_key'
    PROBABILITIES_KEY = 'probabilities'
    NEIGHBORS_KEY = 'neighbors'
    WEIGHT_KEY = 'weight'
    NUM_WALKS_KEY = 'num_walks'
    WALK_LENGTH_KEY = 'walk_length'
    P_KEY = 'p'
    Q_KEY = 'q'

    def __init__(self, dimensions: int = 30, walk_length: int = 20, num_walks: int = 800,
                 permutations: int = 100, p: float = 1, q: float = 1,
                 weight_key: str = 'weight', workers: int = 1, sampling_strategy: dict = None,
                 verbosity: int = 1, temp_folder: str = None, seed: int = None, window: int = 3,
                 min_count: int = 0, iter: int = 1, word2vec_kws: dict = {}):

        """
        Initiates the Node2Vec object, precomputes walking probabilities and generates the walks.

        Parameters
        ----------
        dimensions : int, optional
            Number of embedding dimensions.
        walk_length : int, optional
            Number of nodes in each walk.
        num_walks : int, optional
            Number of walks initiated from each node.
        permutations : int, optional
            Number of independent fitting iteration.
        p : float, optional
            Return hyper parameter (see Grover & Leskovec, 2016).
        q : float, optional
            In-out parameter (see Grover & Leskovec, 2016).
        weight_key : str, optional
            On weighted graphs, this is the key for the weight attribute.
        workers : int, optional
            Number of workers for parallel execution.
        sampling_strategy : dict, optional
            Node specific sampling strategies, supports setting node specific 'q', 'p',
            'num_walks' and 'walk_length'. Set to None for homogeneous sampling.
        verbosity : int, optional
            Verbosity level from 2 (high) to 0 (low).
        seed : int, optional
            Seed for the random number generator. Deterministic results can be obtained if seed is set and workers=1.
        window : int, optional
            The maximum number of steps between the current and predicted node within a sequence.
        min_count : int, optional
            Ignores all nodes with total frequency lower than this.
        iter : int, optional
            Number of iterations (epochs) over all random walk samples.
        word2vec_kws : dict, optional
            Additional parameteres for gensim.models.Word2Vec. Notice that window, min_count,
            iter should be entered as separate parameters (would be ignored).
        temp_folder : str, optional
            Path to folder with enough space to hold the memory map of self.d_graph
            (for big graphs); to be passed joblib.Parallel.temp_folder.

        References
        ----------
        .. [1] Grover, A., & Leskovec, J. (2016, August). node2vec: Scalable feature learning for networks.
               In Proceedings of the 22nd ACM SIGKDD international conference on Knowledge discovery and
               data mining (pp. 855-864).

        Examples
        --------
        Learn embeddings for a given connectome:
        >>> import numpy as np
        >>> import cepy as ce
        >>> sc_group = ce.get_example('sc_group_matrix')
        >>> ce_group = ce.CE(permutations=1, seed=1)  # initiate the connectome embedding model
        >>> ce_group.fit(sc_group)  # fit the model
        Start training  1  word2vec models on  1 threads.
        >>> ce_group.similarity()[0, 1]  # Extract the cosine similarity between node 0 and 1
        0.6134564518636313
        >>> ce_group.save_model('group_ce_copy.pkl')  # save a model:
        >>> ce_loaded_copy = ce.load_model('group_ce_copy.pkl')  # load it
        >>> # Extract the same cosine similarity again, this should be identical apart from minor numerical difference
        >>> ce_loaded_copy.similarity()[0, 1]
        0.6134564518636314
        """

        self.dimensions = dimensions
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.p = p
        self.q = q
        self.weight_key = weight_key
        self.workers = workers
        self.verbosity = verbosity
        self.d_graph = defaultdict(dict)
        self.word2vec_kws = word2vec_kws
        self.permutations = permutations

        if sampling_strategy is None:
            self.sampling_strategy = {}
        else:
            self.sampling_strategy = sampling_strategy

        self.temp_folder, self.require = None, None
        if temp_folder:
            if not os.path.isdir(temp_folder):
                raise NotADirectoryError("temp_folder does not exist or is not a directory. ({})".format(temp_folder))

            self.temp_folder = temp_folder
            self.require = "sharedmem"

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            self.seed = seed
            self.word2vec_kws['seed'] = seed

        if 'workers' not in self.word2vec_kws:
            self.word2vec_kws['workers'] = self.workers

        if 'size' not in self.word2vec_kws:
            self.word2vec_kws['size'] = self.dimensions

        # window, min_count, iter should be entered as separate parameters and not to [word2vec_kws] (would be ignored)
        self.word2vec_kws['window'] = window
        self.word2vec_kws['min_count'] = min_count
        self.word2vec_kws['iter'] = iter

    def _precompute_probabilities(self):
        """
        Precomputes transition probabilities for each node.
        """

        d_graph = self.d_graph

        nodes_generator = self.graph.nodes() if self.verbosity > 1 \
            else tqdm(self.graph.nodes(), desc='Computing transition probabilities')

        for source in nodes_generator:

            # Init probabilities dict for first travel
            if self.PROBABILITIES_KEY not in d_graph[source]:
                d_graph[source][self.PROBABILITIES_KEY] = dict()

            for current_node in self.graph.neighbors(source):

                # Init probabilities dict
                if self.PROBABILITIES_KEY not in d_graph[current_node]:
                    d_graph[current_node][self.PROBABILITIES_KEY] = dict()

                unnormalized_weights = list()
                d_neighbors = list()

                # Calculate unnormalized weights
                for destination in self.graph.neighbors(current_node):

                    p = self.sampling_strategy[current_node].get(self.P_KEY,
                                                                 self.p) if current_node in self.sampling_strategy else self.p
                    q = self.sampling_strategy[current_node].get(self.Q_KEY,
                                                                 self.q) if current_node in self.sampling_strategy else self.q

                    if destination == source:  # Backwards probability
                        ss_weight = self.graph[current_node][destination].get(self.weight_key, 1) * 1 / p
                    elif destination in self.graph[source]:  # If the neighbor is connected to the source
                        ss_weight = self.graph[current_node][destination].get(self.weight_key, 1)
                    else:
                        ss_weight = self.graph[current_node][destination].get(self.weight_key, 1) * 1 / q

                    # Assign the unnormalized sampling strategy weight, normalize during random walk
                    unnormalized_weights.append(ss_weight)
                    d_neighbors.append(destination)

                # Normalize
                unnormalized_weights = np.array(unnormalized_weights)
                d_graph[current_node][self.PROBABILITIES_KEY][
                    source] = unnormalized_weights / unnormalized_weights.sum()

            # Calculate first_travel weights for source
            first_travel_weights = []

            for destination in self.graph.neighbors(source):
                first_travel_weights.append(self.graph[source][destination].get(self.weight_key, 1))

            first_travel_weights = np.array(first_travel_weights)
            d_graph[source][self.FIRST_TRAVEL_KEY] = first_travel_weights / first_travel_weights.sum()

            # Save neighbors
            d_graph[source][self.NEIGHBORS_KEY] = list(self.graph.neighbors(source))

    def _generate_walks(self) -> list:
        """
        Generates the random walks which will be used later for model fitting as List of walks.
        Each walk is a list of nodes.
        """

        flatten = lambda l: [item for sublist in l for item in sublist]

        # Split num_walks for each worker
        num_walks_lists = np.array_split(range(self.num_walks), self.workers)

        walk_results = Parallel(n_jobs=self.workers, temp_folder=self.temp_folder, require=self.require)(
            delayed(parallel_generate_walks)(self.d_graph,
                                             self.walk_length,
                                             len(num_walks),
                                             idx,
                                             self.sampling_strategy,
                                             self.NUM_WALKS_KEY,
                                             self.WALK_LENGTH_KEY,
                                             self.NEIGHBORS_KEY,
                                             self.PROBABILITIES_KEY,
                                             self.FIRST_TRAVEL_KEY,
                                             self.seed,
                                             self.verbosity) for
            idx, num_walks
            in enumerate(num_walks_lists, 1))

        self.walks = flatten(walk_results)

    def _learn_embeddings(self):
        '''
        fit the node2vec models and retrieve the learned weights
        '''
        if self.verbosity > 0:
            print('Start training ', self.permutations, ' word2vec models on ', self.word2vec_kws['workers'],
                  'threads.')

        if (self.workers < self.permutations) and (self.workers > 1):
            # parallel model fittings, each on a single thread
            word2vec_kws_parallel = self.word2vec_kws.copy()
            word2vec_kws_parallel['workers'] = 1
            learned_parameters = Parallel(n_jobs=self.workers, temp_folder=self.temp_folder, require=self.require)(
                delayed(parallel_learn_embeddings)(self.temp_walks_path,
                                                   word2vec_kws_parallel,
                                                   self.nonzero_indices,
                                                   self.X.shape[0],
                                                   idx,
                                                   self.verbosity) for
                idx
                in np.arange(self.permutations))
        else:
            # sequential model fittings, each on a multiple thread
            learned_parameters = []
            for idx in np.arange(self.permutations):
                learned_parameters.append(parallel_learn_embeddings(self.temp_walks_path,
                                                                    self.word2vec_kws,
                                                                    self.nonzero_indices,
                                                                    self.X.shape[0],
                                                                    idx,
                                                                    self.verbosity))

        # unpack the learned embeddings
        self.weights = self.Weights()
        self.weights.w = [learned_parameters[x]['w'] for x in np.arange(len(learned_parameters))]
        self.weights.w_apos = [learned_parameters[x]['w_apos'] for x in np.arange(len(learned_parameters))]
        self.training_losses = [learned_parameters[x]['training_loss'] for x in np.arange(len(learned_parameters))]

    def fit(self, X: np.array):
        '''
        Sample random walks and fit a word2vec model.

        Parameters
        ----------
        X : ndarray, shape: (n_nodes, n_nodes)
            Input adjacency matrix
        
        Returns
        ----------
        ce_model: CE
            Fitted connectome embedding object

        '''

        assert type(X) == np.ndarray, ('Input is expected as a numpy array')
        assert np.all(X >= 0), ('No negative edges allowed in the adjacency matrix')
        assert np.all(X == X.T), ('The adjacency matrix is expected to be symmetric')
        self.X = X

        # deal with zero-connected components
        self.nonzero_indices = np.where(self.X.sum(axis=0) > 0)[0]
        nonzero_adjacency_mat = self.X[self.nonzero_indices, :][:, self.nonzero_indices]

        self.graph = nx.convert_matrix.from_numpy_matrix(nonzero_adjacency_mat)

        self._precompute_probabilities()
        self._generate_walks()

        self.temp_walks_path = tempfile.mkdtemp() + '/walks.txt'
        with open(self.temp_walks_path, 'w') as walks_file:
            walks_file.write('\n'.join(' '.join(map(str, sl)) for sl in self.walks))
        self._learn_embeddings()

        shutil.rmtree(os.path.dirname(self.temp_walks_path))
        del self.walks

    class Weights:
        '''
        Stores the trained weight (W and W' matrices) of all fitting permutations.

        Extract the weights with ``get_w_permut(index, norm_flag)`` and ``get_w_mean(norm_flag)``
        or ``get_w_apos_permut(index, norm_flag)`` and ``get_w_apos_mean(norm_flag)`. If norm_flag
        is set to True l2 normalization would apply on each vector before extraction.

        '''

        def __init__(self):
            self.w = []
            self.w_apos = []

        def get_w_permut(self, index=0, norm=True):
            if norm:  # L2 norm
                return self.w[index] / np.linalg.norm(self.w[index], axis=1)[:, np.newaxis]
            else:
                return self.w[index]

        def get_w_mean(self, norm=True):
            all_w = [self.get_w_permut(i, norm=norm) for i in np.arange(len(self.w))]
            return np.mean(all_w, axis=0)

        def get_w_apos_permut(self, index=0, norm=True):
            if norm:  # L2 norm
                return self.w_apos[index] / np.linalg.norm(self.w_apos[index], axis=0)[np.newaxis, :]
            else:
                return self.w_apos[index]

        def get_w_apos_mean(self, norm=True):
            all_w_apos = [self.get_w_apos_permut(i, norm=norm) for i in np.arange(len(self.w_apos))]
            return np.mean(all_w_apos, axis=0)

    def similarity(self, *args, **kwargs):
        return similarity(self, *args, **kwargs)

    def save_model(self, path, compress=False):
        '''
        Save a model to a pikle object

        Parameters
        ----------
        path : str
            Path to the file.
        compress: bool
            Whether to compress the file with gzip

        Examples
        --------
        Load, align and measure the similarity among two connectome embedding:
        >>> import cepy as ce
        >>> data_path = ce.get_examples_path()
        >>> ce_subject1 = ce.load_model(data_path + '/ce_subject1.pkl.gz')
        >>> ce_subject1.save_model('saved_model.pkl')
        '''
        if path[-7:] == '.pkl.gz':
            compress = True
        if compress:
            if path[-7:] != '.pkl.gz':
                path = path + '.pkl.gz'
            with gzip.open(path, 'wb') as f:
                pickle.dump(self, f, 3)
        else:
            if path[-4:] != '.pkl':
                path = path + '.pkl'
            with open(path, 'wb') as output:
                pickle.dump(self, output, 3)


def load_model(path):
    '''
    Returns a saved model from a pikle object

    Parameters
    ----------
    path : str
        Path to the file.

    Returns
    -------
    x : CE

    Examples
    --------
    Save and Load, and measure the similarity among two connectome embedding:
    >>> import cepy as ce
    >>> ce_subject1 = ce.get_example('ce_subject1')
    >>> sim = ce_subject1.similarity()
    >>> sim[2,5]
    0.1438583470288893
    >>> ce_subject1.save_model('ce_subject1_copy.pkl')
    >>> ce_subject1_copy = ce.load_model('ce_subject1_copy.pkl')
    >>> sim = ce_subject1_copy.similarity()
    >>> sim[2,5]
    0.1438583470288893
    '''
    if path[-7:] == '.pkl.gz':
        with gzip.open(path, 'rb') as f:
            loaded_model = pickle.load(f)
    else:
        with open(path, 'rb') as f:
            loaded_model = pickle.load(f)
    return loaded_model


def similarity(X, Y=None, permut_indices=None, method='cosine_similarity', norm=None):
    '''
    Derive several similarity measures among nodes within the same connectome embeding or among differnet embeddings

    Parameters
    ----------
    X : CE
        The first connectome embedding class on which we perform the similarity measurement
    Y : CE, optional
        The second connectome embedding class on which we perform the similarity measurement. If None, then Y = X.
    permut_indices : tuple or list of tuple, optional
        Indices pairs of permutation (idependent fitting iterations) of the first and secocond CEs. Similarity would be taken for X[index1] and Y[index2]. For a list of tuples similarity would be taken for all pairs. If None all possible pairs are tested.
    method : str, optional
        The similarity measure, one of 'cosine_similarity' | 'hadamard' | 'l1' | 'l2'.
    norm : str, optional
        Which norm sholud be taken before the smilarity measure, on of 'l1' | 'l2' | 'max'. If None no normalization is applied. This has no effect on cosine similarity.

    Returns
    -------
    x : {(num_nodes, num_nodes), (num_nodes, num_nodes, num_embedding_dim)} ndarray or list of ndarray

    Examples
    --------
    Load, align and measure the similarity among two connectome embedding:
    >>> import numpy as np
    >>> import cepy as ce
    >>> ce_subject1 = ce.get_example('ce_subject1')
    >>> sim = ce.similarity(ce_subject1, ce_subject1, method='cosine_similarity')
    >>> sim[3,2]
    0.8230196615715807
    >>> sim = ce_subject1.similarity(ce_subject1, method='cosine_similarity') # equivalent
    >>> sim[3,2]
    0.8230196615715808
    >>> ce_subject2 = ce.get_example('ce_subject2')
    >>> ce_group = ce.get_example('ce_group')
    >>> # aligned both subject to the group consensus space
    >>> ce_subject1_aligned = ce.align(ce_group, ce_subject1)
    >>> ce_subject2_aligned = ce.align(ce_group, ce_subject2)
    >>> # and measure the similarity among all corresponding nodes across subjects
    >>> sim = ce.similarity(ce_subject1, ce_subject2, method='cosine_similarity')
    >>> diagonal_indices = np.diag_indices(sim.shape[0])
    >>> sim[diagonal_indices].mean()
    0.048276127600268885
    '''
    if Y == None:
        Y = X

    if permut_indices == None:
        permut_indices = [(i, i) for i in np.arange(min(len(X.weights.w), len(Y.weights.w)))]
    if type(permut_indices) == tuple:
        permut_indices = [permut_indices]

    similarity_measures = []
    for permut_index in permut_indices:
        x = X.weights.w[permut_index[0]]
        y = Y.weights.w[permut_index[1]]
        if not np.all(np.isclose(X.weights.w_apos[permut_index[0]], X.weights.w_apos[permut_index[1]])):
            warnings.warn('x and y are not aligned.')
        node_dim = x.shape[0]

        if method == 'cosine_similarity':
            norm = 'l2'
        if norm != None:
            if x is y:
                x = normalize(x, norm)
                y = x
            else:
                x = normalize(x, norm)
                y = normalize(y, norm)

        if method == 'cosine_similarity':
            similarity_measure = np.dot(x, y.T)
        elif method in ['hadamard', 'l1', 'l2']:
            array_x = np.transpose(np.tile(x[:, :, np.newaxis], (1, 1, node_dim)), (0, 2, 1))
            array_y = np.transpose(np.tile(y[:, :, np.newaxis], (1, 1, node_dim)), (2, 0, 1))
            if method == 'hadamard':
                similarity_measure = array_x * array_y
            elif method == 'l1':
                similarity_measure = np.abs(array_x - array_y)
            elif method == 'l1':
                similarity_measure = (array_x - array_y) ** 2
        else:
            raise Exception('Methods ', method, 'is not supported.')

        similarity_measures.append(similarity_measure)

    if len(similarity_measures) == 1:
        return similarity_measures[0]
    else:
        return np.mean(similarity_measures, axis=0)


def get_example(name):
    import pathlib
    '''
    Returns an existing file example. Can be used for testing/ experimenting.

    Parameters
    ----------
    file : str
        File name (without the extention).

    Returns
    -------
    path : str
        path to the file

    Examples
    --------
    Load an existing connectome embedding model:
    >>> import cepy as ce
    >>> ce_subject1= ce.get_example('ce_subject1')
    >>> w = ce_subject1.weights.get_w_mean()
    >>> w.shape
    (200, 30)
    '''

    import cepy as ce
    files = ['ce_subject1.pkl.gz', 'ce_subject2.pkl.gz', 'ce_group.pkl.gz', 'sc_subject1_matrix.npz',
             'sc_subject2_matrix.npz', 'sc_group_matrix.npz']
    names = [file[:file.find('.')] for file in files]
    file_types = [''.join(pathlib.Path(file).suffixes) for file in files]
    names_to_files = dict(zip(names, files))
    names_to_file_types = dict(zip(names, file_types))

    if not name in names:
        raise ValueError(name, 'is not recognized.')

    data_path = os.path.dirname(os.path.dirname(ce.__file__)) + '/data'
    path = data_path + '/' + names_to_files[name]
    if not os.path.isfile(path):
        raise Exception('The file', path, ' is missing.')

    if names_to_file_types[name] == '.npz':
        res = np.load(path)['x']
    elif names_to_file_types[name] in ['.pkl', '.pkl.gz']:
        res = load_model(path)
    return res


def get_examples_path():
    '''
    Returns the file examples path.
    '''
    import cepy as ce
    return os.path.dirname(os.path.dirname(ce.__file__)) + '/data'


if __name__ == "__main__":
    import doctest

    doctest.testmod()