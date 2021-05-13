import random
import numpy as np
from tqdm import tqdm
import gensim
import time
import pkg_resources

def parallel_generate_walks(d_graph: dict, global_walk_length: int, num_walks: int, cpu_num: int,
                            sampling_strategy: dict = None, num_walks_key: str = None, walk_length_key: str = None,
                            neighbors_key: str = None, probabilities_key: str = None, first_travel_key: str = None,
                            seed: int = None, verbosity: int = 1) -> list:
    """
    Generates the random walks which will be used as the skip-gram input.

    :return: List of walks. Each walk is a list of nodes.
    """
    np.random.seed(seed)
    walks = list()

    if verbosity > 1:
        pbar = tqdm(total=num_walks, desc='Generating walks (CPU: {})'.format(cpu_num))
    for n_walk in range(num_walks):

        # Update progress bar
        if verbosity > 1:
            pbar.update(1)

        # Shuffle the nodes
        shuffled_nodes = list(d_graph.keys())
        random.shuffle(shuffled_nodes)

        # Start a random walk from every node
        for source in shuffled_nodes:

            # Skip nodes with specific num_walks
            if source in sampling_strategy and \
                    num_walks_key in sampling_strategy[source] and \
                    sampling_strategy[source][num_walks_key] <= n_walk:
                continue

            # Start walk
            walk = [source]

            # Calculate walk length
            if source in sampling_strategy:
                walk_length = sampling_strategy[source].get(walk_length_key, global_walk_length)
            else:
                walk_length = global_walk_length

            # Perform walk
            while len(walk) < walk_length:

                walk_options = d_graph[walk[-1]].get(neighbors_key, None)

                # Skip dead end nodes
                if not walk_options:
                    break

                if len(walk) == 1:  # For the first step
                    probabilities = d_graph[walk[-1]][first_travel_key]
                    walk_to = np.random.choice(walk_options, size=1, p=probabilities)[0]
                else:
                    probabilities = d_graph[walk[-1]][probabilities_key][walk[-2]]
                    walk_to = np.random.choice(walk_options, size=1, p=probabilities)[0]

                walk.append(walk_to)

            walk = list(map(str, walk))  # Convert all to strings

            walks.append(walk)

    if verbosity > 1:
        pbar.close()

    return walks

def get_hash(astring):
    '''
    Returns consistent values to the word2vec model to ensure reproducibility.

    Replace python's inconsistent hashing function (notice this is not a real hashing function but it will work for the current use).
    '''
    return int(astring)


def parallel_learn_embeddings(walks_file, word2vec_kws, nonzero_indices, num_nodes, cpu_num, verbosity):
    """
    Fit the node2vec model on the sampled walks and returns the learned parameters.

    :return: A dictionary with the w and w' parameters and the final training loss.
    """
    if verbosity > 1:
        s_time = time.time()

    model = gensim.models.Word2Vec(corpus_file=walks_file, hashfxn = get_hash, **word2vec_kws)

    # The word2vec algorithm does not preserve the nodes order, so we should sort it
    gensim_version = pkg_resources.get_distribution("gensim").version
    if gensim_version > '4.0.0':
        nodes_unordered = np.array([int(node) for node in model.wv.index_to_key])
    else:
        nodes_unordered = np.array([int(node) for node in model.wv.index2word])
    sorting_indices = np.argsort(nodes_unordered)

    # initiate  W and W'
    embed_dims = word2vec_kws['vector_size'] if 'vector_size' in word2vec_kws else word2vec_kws['size']
    w = np.empty((num_nodes, embed_dims))
    w_apos = np.empty((embed_dims, num_nodes))

    # get the trained matrices for the non-zero connected nodes
    w[nonzero_indices, :] = model.wv.vectors[sorting_indices, :]
    if pkg_resources.get_distribution("gensim").version >= '4.0.0':
        w_apos[:, nonzero_indices] = model.syn1neg.T[:, sorting_indices]
    else:
        w_apos[:, nonzero_indices] = model.trainables.syn1neg.T[:, sorting_indices]
    # set random values for the zero connected nodes
    if len(nonzero_indices) < num_nodes:
        zero_indices = np.ones((num_nodes), dtype = bool)
        zero_indices[nonzero_indices] = 0
        w[zero_indices, :] = np.random.uniform(low=-0.5, high=0.5,  \
            size=(int(zero_indices.sum()), embed_dims)) / embed_dims
        w_apos[:, zero_indices] = np.random.uniform(low=-0.5, high=0.5,  \
            size=(embed_dims, int(zero_indices.sum()))) / embed_dims

    training_loss = model.get_latest_training_loss()

    if verbosity > 1:
        print('Done training the word2vec model ', cpu_num, 'in {:.4f} seconds.'.format(time.time() - s_time))

    return {'w': w, 'w_apos': w_apos, 'training_loss': training_loss}
