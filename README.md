# cepy

Implementation of the connectome embedding (CE) framework.

Embedding of brain graph or connectome embedding (CE) involves finding a compact vectorized 
representation of nodes that captures their higher-order topological attributes. CE are 
obtained using the node2vec algorithm fitted on random walk on a brain graph. The current
 framework includes a novel approach to align separately learned embeddings to the same 
 latent space. Cepy is tested for Python 3.6, 3.7 and 3.8.

- **Documentation:** https://cepy.readthedocs.io/en/latest/
 
## Installation

`pip install cepy`

## Usage
```python
import cepy as ce
import numpy as np

# Load an adjacency matrix (structural connectivity matrix)
sc_group = ce.get_example('sc_group_matrix')

# Initiate and fit the connectome embedding model
ce_group = ce.CE(permutations = 1, seed=1)  
ce_group.fit(sc_group)

# Extract the cosine similarity matrix among all pairwise nodes
cosine_sim = ce_group.similarity()

# Save and load the model
ce_group.save_model('group_ce.json') 
ce_loaded = ce.load_model('group_ce.json') # load it

# Load two existing CE models  
ce_subject1 = ce.get_example('ce_subject1')
ce_subject2 = ce.get_example('ce_subject2')

# Align the two to the space of the [ce_group]:
ce_subject1_aligned = ce.align(ce_group, ce_subject1)
ce_subject2_aligned = ce.align(ce_group, ce_subject2)

# Extract the node vectorized representations (normalized) for subsequent use - prediction, for example 
w_sbject1 = ce_subject1_aligned.weights.get_w_mean(norm = True)
w_sbject2 = ce_subject2_aligned.weights.get_w_mean(norm = True)
 

```

A set of example interactive Jupyter notebooks are also available [here](https://github.com/GidLev/cepy/tree/master/examples).

### Citing
If you find *cepy* useful for your research, please consider citing the following paper:
    
    Levakov, G., Faskowitz, J., Avidan, G. & Sporns, O. (2021). Mapping structure to function
     and behavior with individual-level connectome embedding. bioRxiv. doi: https://doi.org/10.1101/2021.01.13.426513 

### Contributing
Cepy is an open-source software project, and we welcome contributions from anyone. 
We suggest [raising](https://github.com/GidLev/cepy/issues) an issue prior to 
working on a new feature. 

### Reference
* The node2vec implementation is modeified from the [node2vec](https://github.com/eliorc/node2vec) package by Elior Cohen and the [connectome_embedding](https://github.com/gidonro/Connectome-embeddings) code by Gideon Rosenthal.
* Rosenthal, G., Váša, F., Griffa, A., Hagmann, P., Amico, E., Goñi, J., ... & Sporns, O. (2018). Mapping higher-order relations between brain structure and function with embedded vector representations of connectomes. Nature communications, 9(1), 1-12.
;
