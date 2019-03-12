# Tiny LeNet designed to run on low memory, high speed

# Problem Statement
Deep Neural networks are among the most powerful machine learning techniques that are becoming very interesting for Big Data  
applications. In the context of embedded platforms, implementing efficient DNNs in terms of performance and energy consumption while  
they maintain a required quality is very challenging. Sparsity can be  used as an intelligent technique for reducing the size of DNNs. The  purpose of this project is to explore the possibilities of various sparsity models for LeNet which can be then used to evaluate their performance in a reconfigurable computing system.

# Approach
Part- I : Input Sparsity
1. Train + Test LeNet for five datasets
2. Save the trained model (I call this base model)
3. Apply sparsity of input using LASSO (I call this model LeNet - I)
4. Results for inference with & without retraining LeNet -I

Part- II : Model sparsity
1. Apply sparsity on the connecting nodes of base model using LASSO
2. Sort the importance of nodes in decreasing order
3. Apply mean and geometric mean on the consecutive sorted node priorities to generate the approximated weight matrices
4. Sort the weights obtained in (3) in descending
5. Drop them iteratively in steps of 5% upto 90%
6. I call this sparse model as LeNet - II to which corresponding sparse inputs are applied.
7. Results for inference with & without retraining LeNet -II

# Datasets
1. MNIST handwritten digit database
2. Fashion-MNIST image database
3. NIST alphabet recognition database (https://www.nist.gov/itl/iad/image-group/emnist-dataset)
4. CIFAR10 database
5. Street View Housing Numbers (http://ufldl.stanford.edu/housenumbers/)


