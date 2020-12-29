[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/GidLev/cepy/master)

# Connectome embedding workflow step-by-step


As originally proposed by Rosenthal et al. (2018), Cepy utilize the word2vec 
algorithm (Mikolov et al., 2013) to create a vectorized representation of 
brain nodes based on their high-level topological relations. The word2vec 
algorithm is use to create word embeddings that preserve 
their context as it typically appears in a sentence. In our work nodes 
instead of words are embedded preserving their "context" defined by their 
neighbors in a random walks instead of sentences. 

**In the following notebooks we will cover the basics of the CE implementation
 and demonstrate its ability for mapping structural connectivity to functional 
 connectivity and to individual differences.**   

Python Jupyter notebooks are available as [GitHub](https://github.com/) **static** pages or as **interactive** [binder](https://mybinder.readthedocs.io/en/latest/) notebooks.   



<img src="https://raw.githubusercontent.com/GidLev/cepy/master/examples/images/ce_workflow_full.png" alt="The connectome embedding framework"/>



* Random walk sampling (see a.i.) - [static](https://github.com/GidLev/cepy/blob/master/examples/random_walks_generation.ipynb),
 [interactive](https://mybinder.org/v2/gh/GidLev/cepy/master?filepath=examples%2Frandom_walks_generation.ipynb) 

* Learning (fitting) connectome embedding (CE; see a.ii, a.iii) and mapping structural to functional connectivity (c.i) -  
[static](https://github.com/GidLev/cepy/blob/master/examples/learn_embedding.ipynb), 
[interactive](https://mybinder.org/v2/gh/GidLev/cepy/master?filepath=examples%2Flearn_embedding.ipynb) 

* Aligning CEs within the same individual (independent fitting of the same subject) -  [static](https://github.com/GidLev/cepy/blob/master/examples/intra_embedding_alignment.ipynb), [interactive](http://link....) 

* Aligning CEs between individuals (across subjects; see b) -  [static](https://github.com/GidLev/cepy/blob/master/examples/inter_embedding_alignment.ipynb), 
[interactive](https://mybinder.org/v2/gh/GidLev/cepy/master?filepath=examples%2Fintra_embedding_alignment.ipynb) 

* Learning and aligning CEs of a large cohort (see a,b) -  [static](https://github.com/GidLev/cepy/blob/master/examples/ce_subjects_pipeline.ipynb) 

* Predicting age from aligned CEs (see c.ii)-  [static](https://github.com/GidLev/cepy/blob/master/examples/ce_prediction.ipynb), 
[interactive](https://mybinder.org/v2/gh/GidLev/cepy/master?filepath=examples%2Fce_prediction.ipynb) 


## Reference

* Mikolov, T., Sutskever, I., Chen, K., Corrado, G. S., & Dean, J. (2013). Distributed representations of words and phrases and their compositionality. In Advances in neural information processing systems (pp. 3111-3119).


* Rosenthal, G., Váša, F., Griffa, A., Hagmann, P., Amico, E., Goñi, J., ... & Sporns, O. (2018). Mapping higher-order relations between brain structure and function with embedded vector representations of connectomes. Nature communications, 9(1), 1-12.