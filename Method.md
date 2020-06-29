# 1. PFam dataset:

Transfer learning plays an important role in the growth of Deep Learning, especially when amount of data is limited. So we think about finding a big dataset to transfer from. PFam dataset is our choice.

### Data of PFam:

PFam dataset contains data to train a model to predict the function of protein domains.

Domains are functional sub-parts of proteins; much like images in ImageNet are pre-segmented to 
 contain exactly one object class, this data is pre-segmented to contain exactly and only one
 domain.

The task is: given the amino acid sequence of the protein domain, predict which class it belongs
 to. There are about 1 million training examples, and 18,000 output classes.

# 2. Pretrained with Fast Text on PFam dataset:

Cause our target is not classify 18000 classes of PFam dataset and limitation of computation power so we first try transfer learning from simple model like Fast Text (The features are embedded and averaged to form the hidden variable come along with Hierarchical softmax). Consequences of that is performance is pretty poor (F1 Score is just around 4%)

To avoid the embedding which will be extracted from Fast Text model remember protein sequences in our data set, we remove all protein sequences that contain any of them. So result on CV and test set are reliable.

# 3. Features combination:

To avoid model rely on only one type of feature (such as pretrained embedding) we use several type of feature and combine them in one end-to-end model to help fight overfitting. Along with pretrained embedding from Fast Text model, we use 3 more (so we have 4 branches of input):

- TF-IDF: help capture global frequency of amino acid in protein sequence.
- PSSM: ...
- CP: help capture chemical/physical information of protein sequence.

# 4. Topology:

Fast Text embedding and PSSM feature are 3D tensor (batch size, length, dimension size) so we use Conv1D and RNN variations (LSTM/GRU) for the 2 branches

TF-IDF and CP are 2D tensor (batch size, dimension size) so we use a fully connected layer for the 2 branches

We concatenate outputs of the branches to feed to few last fully connected layer with Drop-out between them.

<#Image of topology>

# 5. Tuning Strategy:

Cause of multi-branches inputs, so our model have a lot of hyper-parameter on each branch. To reduce time spending for tuning them, we tune each branch first and then combine them to the last end-to-end model.

<#Image of branch Embedding>

<#Image of branch PSSM>

<#Image of branch TF-IDF>

<#Image of branch CP>

On the last model, we use random search on all hyper-parameter with range around the values from tuning each branch. 

# 6. Future work:

- Try to get better performance on PFam dataset
- Instead of concatenate the branches inputs, we can use weighted losses contribute from each branch
- Use better topology such as Attention variations

