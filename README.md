# Neural-Net
Basic neural network with one hidden layer that learns using stochastic gradient descent (on-line training)

## Implementation
- The network works for binary classification problems, and therefore it has one output unit with a sigmoid function. The sigmoid predicts 0 for the first class listed in the input ARFF files, and 1 for the second class.
- Stochasic gradient descent is used to minimize cross-entropy error.
- If h = 0, the network has no hidden units (i.e. no hidden layer), and the input units are directly connected to the output unit. Otherwise, if h > 0, the network constitutes a single layer of h hidden units with each fully connected to the input units and the output unit.
- For each numeric feature, network design uses one input unit. For each discrete feature, one-of-k encoding is used. (Optionally, we can use a thermometer encoding for discrete numeric features).
- To ensure that hidden unit activations don't get saturated numeric features are standardized.
- A momentum term is not used during training.
- Each epoch is one complete pass through the training instances. Note, the order of the training instances is randomized before starting the training, but each epoch can go through the instances in the same order.
- All weights and bias parameters are initialized to random values in [-0.01, 0.01].

## Running the program
```sh
nnet l h e train-set-file test-set-file
```
l specifies the learning rate, h gives the number of hidden units in the hidden layer (if non-zero), e represents the number of trainings epochs.  After training for e epochs on the training set, the learned neural net is used to predict a classification for every instance in the test set.
