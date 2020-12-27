# Connectome embedding workflow step-by-step

## overview

As originally proposed by Rosenthal et al. (2018), Cepy utilize the word2vec 
algorithm (Mikolov et al., 2013) to create a vectorized representation of 
brain nodes based on their high-level topological relations. The word2vec 
algorithm is use to create word embeddings that preserve 
their context as it typically appears in a sentence. In our work nodes 
instead of words are embedded preserving their "context" defined by their 
neighbors in a random walks instead of sentences. Here is a graphical 
depiction of the complete workflow: 


<img src="https://raw.githubusercontent.com/GidLev/cepy/master/examples/ce_workflow.png" alt="p and q parameters" width="413" height = "558"/>



* i. The structural connectome is the input to the CE model. Multiple random 
walks are sampled from the structural connectivity graph. Letters represent
 all unique nodes in both hemisphere, dash lines represent edges that exist
  in the connectome and colored line represent sampled random walks. 
* ii. Sliding, fixed size windows are taken from the random walk sequences. 
Within each window, the center node is used as the target (black) and the 
surroundings as context (white).
* iii. Pairs of context and target nodes  are used as the input and target
 of an artificial neural network with a single hidden layer, i.e. the 
 embedding layer. The input, output and target layers are k dimensional
  vectors, where k is the number of nodes 
 in the brain graph. The embedding layer is a k' dimensional vector, 
 where k' is set to be k'<k. W and W' are the learned weight matrices 
 that define the transformation between the input and embedding layer,
  and the embedding and output layer, respectively. The model parameters, 
  W and W', are iteratively updated using stochastic gradient descent. 



## Reference
