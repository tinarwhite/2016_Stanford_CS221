import numpy as np

from cs231n.layers import *
from cs231n.layer_utils import *


class AutoencoderNet(object):
    """
    A fully-connected neural network with an arbitrary number of hidden layers,
    ReLu nonlinearities, and a L2 norm loss function. This will also implement
    dropout and batch normalization as options. For a network with L layers,
    the architecture will be
    {affine - [batch norm] - ReLu - [dropout]} x (L - 1) - affine - L2 norm
    where batch normalization and dropout are optional, and the {...} block is
    repeated L - 1 times.
    Similar to the TwoLayerNet above, learnable parameters are stored in the
    self.params dictionary and will be learned using the Solver class.
    """

    def __init__(self, hidden_dims, input_dim=1000, num_classes=1000,
                 dropout=0, use_batchnorm=False, reg=0.0,
                 weight_scale=1e-2, dtype=np.float32, seed=None, nonlin = 'relu'):
        """
        Initialize a new FullyConnectedNet.
        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=0 then
          the network should not use dropout at all.
        - use_batchnorm: Whether or not the network should use batch normalization.
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using
          this datatype. float32 is faster but less accurate, so you should use
          float64 for numeric gradient checking.
        - seed: If not None, then pass this random seed to the dropout layers. This
          will make the dropout layers deteriminstic so we can gradient check the
          model.
        """
        self.use_batchnorm = use_batchnorm
        self.use_dropout = dropout > 0
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.nonlin = nonlin
        self.params = {}
        self.cache = {}

        ############################################################################
        # TODO: Initialize the parameters of the network, storing all values in    #
        # the self.params dictionary. Store weights and biases for the first layer #
        # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
        # initialized from a normal distribution with standard deviation equal to  #
        # weight_scale and biases should be initialized to zero.                   #
        #                                                                          #
        # When using batch normalization, store scale and shift parameters for the #
        # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
        # beta2, etc. Scale parameters should be initialized to one and shift      #
        # parameters should be initialized to zero.                                #
        ############################################################################
        for i in range(self.num_layers):
            if i == 0:
                self.params['W' + repr(i)] = np.random.normal(loc=0, scale=weight_scale,
                                                              size=(input_dim, hidden_dims[i]))
                self.params['b' + repr(i)] = np.zeros(hidden_dims[0])
                if self.use_batchnorm:
                    self.params['gamma' + repr(i)] = np.ones(hidden_dims[0])
                    self.params['beta' + repr(i)] = np.zeros(hidden_dims[0])
            elif i == self.num_layers - 1:
                self.params['W' + repr(i)] = np.random.normal(loc=0, scale=weight_scale,
                                                              size=(hidden_dims[i - 1], num_classes))
                self.params['b' + repr(i)] = np.zeros(num_classes)
                # Last layer does not actually use batchnorm
                # if self.use_batchnorm:
                #     self.params['gamma' + repr(i)] = np.ones(num_classes)
                #     self.params['beta' + repr(i)] = np.zeros(num_classes)
            else:
                self.params['W' + repr(i)] = np.random.normal(loc=0, scale=weight_scale,
                                                              size=(hidden_dims[i - 1], hidden_dims[i]))
                self.params['b' + repr(i)] = np.zeros(hidden_dims[i])
                if self.use_batchnorm:
                    self.params['gamma' + repr(i)] = np.ones(hidden_dims[i])
                    self.params['beta' + repr(i)] = np.zeros(hidden_dims[i])

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the mode
        # (train / test). You can pass the same dropout_param to each dropout layer.
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {'mode': 'train', 'p': dropout}
            if seed is not None:
                self.dropout_param['seed'] = seed

        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer. You should pass self.bn_params[0] to the forward pass
        # of the first batch normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
        self.bn_params = []
        if self.use_batchnorm:
            self.bn_params = [{'mode': 'train'} for i in xrange(self.num_layers - 1)]

        # Cast all parameters to the correct datatype
        for k, v in self.params.iteritems():
            self.params[k] = v.astype(dtype)



    def loss(self, X, y=None):
        """
        Compute loss and gradient for the fully-connected net.
        Input / output: Same as TwoLayerNet above.
        """
        X = X.astype(self.dtype)
        mode = 'test' if y is None else 'train'

        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        if self.dropout_param is not None:
            self.dropout_param['mode'] = mode
        if self.use_batchnorm:
            for bn_param in self.bn_params:
                bn_param[mode] = mode

        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the fully-connected net, computing  #
        # the class scores for X and storing them in the scores variable.          #
        #                                                                          #
        # When using dropout, you'll need to pass self.dropout_param to each       #
        # dropout forward pass.                                                    #
        #                                                                          #
        # When using batch normalization, you'll need to pass self.bn_params[0] to #
        # the forward pass for the first batch normalization layer, pass           #
        # self.bn_params[1] to the forward pass for the second batch normalization #
        # layer, etc.                                                              #
        ############################################################################
        layer_x = X
        last_layer = self.num_layers - 1
        for i in range(last_layer):
            w, b = self.params['W' + repr(i)], self.params['b' + repr(i)]
            layer_x, self.cache['affine' + repr(i)] = affine_forward(layer_x, w, b)
            if self.use_batchnorm:
                gamma, beta = self.params['gamma' + repr(i)], self.params['beta' + repr(i)]
                layer_x, self.cache['bnorm'+repr(i)] = batchnorm_forward(layer_x, gamma, beta, self.bn_params[i])
            if self.nonlin == 'sigmoid':
                layer_x, self.cache['relu' + repr(i)] = sigmoid_forward(layer_x)
            else:
                layer_x, self.cache['relu' + repr(i)] = relu_forward(layer_x)
            if self.use_dropout:
                layer_x, self.cache['dropout' + repr(i)] = dropout_forward(layer_x, self.dropout_param)

        w, b = self.params['W' + repr(last_layer)], self.params['b' + repr(last_layer)]
        scores, self.cache['affine' + repr(last_layer)] = affine_forward(layer_x, w, b)
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If test mode return early
        if mode == 'test':
            return scores

        loss, grads = 0.0, {}
        ############################################################################
        # TODO: Implement the backward pass for the fully-connected net. Store the #
        # loss in the loss variable and gradients in the grads dictionary. Compute #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        # When using batch normalization, you don't need to regularize the scale   #
        # and shift parameters.                                                    #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        loss, d_out = L2_norm_loss(scores,y)

        d_hidden, grads['W' + repr(last_layer)], grads['b' + repr(last_layer)] = \
            affine_backward(d_out, self.cache['affine' + repr(last_layer)])
        for i in range(last_layer - 1, -1, -1):
            w_name = 'W' + repr(i)
            b_name = 'b' + repr(i)
            a_name = 'affine' + repr(i)
            r_name = 'relu' + repr(i)
            d_name = 'dropout' + repr(i)

            w = self.params[w_name]
            loss += 0.5 * self.reg * np.sum(w * w)

            if self.use_dropout:
                dropout_cache = self.cache[d_name]
                d_hidden = dropout_backward(d_hidden, dropout_cache)

            affine_cache, relu_cache = self.cache[a_name], self.cache[r_name]
            if self.nonlin == 'sigmoid':
                da = sigmoid_backward(d_hidden, relu_cache)
            else:
                da = relu_backward(d_hidden, relu_cache)

            if self.use_batchnorm:
                bn_name = 'bnorm' + repr(i)
                dgamma = 'gamma' + repr(i)
                dbeta = 'beta' + repr(i)

                bn_cache = self.cache[bn_name]
                da, grads[dgamma], grads[dbeta] = batchnorm_backward(da, bn_cache)

            d_hidden, grads[w_name], grads[b_name] = affine_backward(da, affine_cache)

            grads[w_name] += self.reg * w
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads