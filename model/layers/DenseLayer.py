import theano.tensor as T


class DenseLayer:

    def __init__(self, weights, bias, activation='linear'):
        self.weights = self.to_list(weights)
        self.bias = bias
        self.activation = activation

    @staticmethod
    def to_list(x):
        if isinstance(x, list):
            return x
        return [x]

    def forward(self, x, shape):
        # Compute linear combination.
        x = self.to_list(x)
        shape = self.to_list(shape)
        y = None
        for i in range(len(x)):
            if y is None:
                y = T.batched_tensordot(self.weights[i], T.reshape(x[i], shape[i]), axes=[[2], [1]])
            else:
                y += T.batched_tensordot(self.weights[i], T.reshape(x[i], shape[i]), axes=[[2], [1]])
        y += self.bias

        # Apply non-linear activation function.
        if self.activation == 'soft_plus':
            y = T.nnet.softplus(y)
        if self.activation == 'relu':
            y = T.nnet.relu(y)
        if self.activation == 'tanh':
            y = T.tanh(y)
        return y
