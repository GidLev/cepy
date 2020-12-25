.. image:: docs_header.png


Welcome to Cepy's documentation!
================================

The Cepy pacakge is a python implementation of the connectome embedding (CE) framework.

Embedding of brain graph or connectome embedding (CE) involves finding a compact vectorized 
representation of nodes that captures their higher-order topological attributes. CE are 
obtained using the node2vec algorithm fitted on random walk on a brain graph. The current
framework includes a novel approach to align separately learned embeddings to the same 
latent space.

Installation
============

Cepy can be installed using pip:

.. code-block:: shell

  pip install cepy

Example Notebooks
=================
If you're looking for hands-on examples on connectome embeddings (CE) and Cepy, check out the links below. These are notebooks that go over the general methodology of the CE framework and how it is used in Cepy.

We recommend going over them in the following order:


- `1. ---- <https://github.com/--->`_ , coming soon.



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

- see the full `API  <source/cepy.html>`_

Cite
================

If you find *cepy* useful for your research, please consider citing the following paper:
    
	Levakov, G., Faskowitz, J., Avidan, G. & Sporns, O. (2020). Mapping structure to function 
	and behavior with individual-level connectome embedding. In preparation

Acknowledgements
================

* The node2vec implementation is modeified from the `node2vec <https://github.com/eliorc/node2vec>`_ package by Elior Cohen and the `connectome_embedding <https://github.com/gidonro/Connectome-embeddings`_ code by Gideon Rosenthal.
* Rosenthal, G., Váa, F., Griffa, A., Hagmann, P., Amico, E., Goñi, J., ... & Sporns, O. (2018). Mapping higher-order relations between brain structure and function with embedded vector representations of connectomes. Nature communications, 9(1), 1-12.


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
