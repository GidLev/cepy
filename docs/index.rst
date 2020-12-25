.. cepy documentation master file, created by
   sphinx-quickstart on Fri Dec 25 09:49:05 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. image:: docs_header.png


Welcome to Cepy's documentation!
================================

The Cepy pacakge is a python implementation of the connectome embedding (CE) framework.

Embedding of brain graph or connectome embedding (CE) involves finding a compact vectorized 
representation of nodes that captures their higher-order topological attributes. CE are 
obtained using the node2vec algorithm fitted on random walk on a brain graph. The current
framework includes a novel approach to align separately learned embeddings to the same 
latent space.


Example Notebooks are available!
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
If you're looking for hands-on examples on connectome embeddings (CE) and Cepy, check out the links below! These are notebooks that go over the general methodology of the CE framework and how it is used in Cepy.

We recommend going over them in the following order:


We recommend you follow the notebooks in order:

- [1. ----](https://github.com/---), a notebook for ---.

- [2. ----](https://github.com/---), a notebook for ---.

- [3. ----](https://github.com/---), a notebook for ---.

- [4. ----](https://github.com/---), a notebook for ---.




Quick start
================================

.. code-block:: python
	:linenos:

	import cepy as ce
	import numpy as np

	# Load an adjacency matrix (structural connectivity matrix)
	sc_group = ce.get_example('sc_group_matrix')

	# Initiate and fit the connectome embedding model
	ce_group = ce.CE(permutations = 1, seed=1)  
	ce_group.fit(sc_group)

	# Extract the cosine similarity matrix among pairwise nodes
	cosine_sim = ce_group.similarity()

	# Save and load the model
	ce_group.save_model('group_ce.pkl') 
	ce_loaded = ce.load_model('group_ce.pkl') # load it

	# Load two existing CE models  
	ce_subject1 = ce.get_example('ce_subject1')
	ce_subject2 = ce.get_example('ce_subject2')

	# Align the two to the space of the [ce]:
	ce_subject1_aligned = ce.align(ce, ce_subject1)
	ce_subject2_aligned = ce.align(ce, ce_subject2)

	# Extract the node vectorized representations (normalized) for subsequent use (prediction, for example) 
	w_sbject1 = ce_subject1_aligned.weights.get_w_mean(norm = True)
	w_sbject2 = ce_subject2_aligned.weights.get_w_mean(norm = True)
	 


.. toctree::
   :maxdepth: 2
   :caption: Contents:



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
