import numpy as np

from cs231n.layers import *
from cs231n.rnn_layers import *


class PDEmodelRNN(object):
  """
  A PDEsolveRNN produces solutions from parameters using a recurrent
  neural network.

  The RNN receives input vectors of size D, has a solution size of V, works on
  sequences of length T, has an RNN hidden dimension of H, uses solution vectors
  of dimension W, and operates on minibatches of size N.

  Note that we don't use any regularization for the PDEsolveRNN.
  """
  def __init__(self, input_dim=3, solution_dim=1000, hidden_dim=15, hidden_dim2=30, cell_type='rnn', dtype=np.float32):
    """
    Construct a new PDEsolveRNN instance.

    Inputs:
    - input_dim: Dimension D of input parameter vectors.
    - solution_dim: Dimension W of solution vectors.
    - hidden_dim: Dimension H for the hidden state of the RNN.
    - cell_type: What type of RNN to use; either 'rnn' or 'lstm'.
    - dtype: numpy datatype to use; use float32 for training and float64 for
      numeric gradient checking.
    """
    if cell_type not in {'rnn', 'lstm'}:
      raise ValueError('Invalid cell_type "%s"' % cell_type)
    
    self.cell_type = cell_type
    self.dtype = dtype
    self.params = {}
    
    self._null = np.ones(1000)*0.999
    self._start = np.zeros(1000)
    self._end = np.ones(1000)
    
    # Initialize word vectors
    self.params['W_embed'] = np.random.randn(solution_dim, hidden_dim2)
    self.params['W_embed'] /= np.sqrt(solution_dim)
    self.params['b_embed'] = np.zeros(hidden_dim2)
    # Initialize CNN -> hidden state projection parameters
    self.params['W_proj'] = np.random.randn(input_dim, hidden_dim)
    self.params['W_proj'] /= np.sqrt(input_dim)
    self.params['b_proj'] = np.zeros(hidden_dim)

    # Initialize parameters for the RNN
    dim_mul = {'lstm': 4, 'rnn': 1}[cell_type]
    self.params['Wx'] = np.random.randn(hidden_dim2, dim_mul * hidden_dim)
    self.params['Wx'] /= np.sqrt(hidden_dim2)
    self.params['Wh'] = np.random.randn(hidden_dim, dim_mul * hidden_dim)
    self.params['Wh'] /= np.sqrt(hidden_dim)
    self.params['b'] = np.zeros(dim_mul * hidden_dim)
    
    # Initialize output to solution weights
    self.params['W_deco'] = np.random.randn(hidden_dim, hidden_dim2)
    self.params['W_deco'] /= np.sqrt(hidden_dim)
    self.params['b_deco'] = np.zeros(hidden_dim2)	

    # Initialize output to solution weights
    self.params['W_sol'] = np.random.randn(hidden_dim2, solution_dim)
    self.params['W_sol'] /= np.sqrt(hidden_dim2)
    self.params['b_sol'] = np.zeros(solution_dim)
  
      
    # Cast parameters to correct dtype
    for k, v in self.params.iteritems():
      self.params[k] = v.astype(self.dtype)


  def loss(self, features, solutions):
    """
    Compute training-time loss for the RNN. We input features and
    ground-truth solutions for those features, and use an RNN (or LSTM) to compute
    loss and gradients on all parameters.
    
    Inputs:
    - features: chosen features, of shape (N, D)
    - solutions: Ground-truth solutions; an integer array of shape (N, T) where
      each element is in the range 0 <= y[i, t] < V
      
    Returns a tuple of:
    - loss: Scalar loss
    - grads: Dictionary of gradients parallel to self.params
    """
    # Cut solutions into two pieces: solutions_in has everything but the last sol
    # and will be input to the RNN; solutions_out has everything but the first
    # sol and this is what we will expect the RNN to generate. These are offset
    # by one relative to each other because the RNN should produce sol (t+1)
    # after receiving sol t. The first element of solutions_in will be the START
    # token, and the first element of solutions_out will be the first sol.
    solutions_in = solutions[:, :-1]
    solutions_out = solutions[:, 1:]
    
    # You'll need this 
    mask = (solutions_out != self._null)

    # Weight and bias for the affine transform from image features to initial
    # hidden state
    W_proj, b_proj = self.params['W_proj'], self.params['b_proj']
    
    # Word embedding matrix
    W_embed, b_embed = self.params['W_embed'], self.params['b_embed']
    # Input-to-hidden, hidden-to-hidden, and biases for the RNN
    Wx, Wh, b = self.params['Wx'], self.params['Wh'], self.params['b']

    # Weight and bias for the RNN hidden-to-hidden transformation.
    W_deco, b_deco = self.params['W_deco'], self.params['b_deco']
	
    # Weight and bias for the hidden-to-solution transformation.
    W_sol, b_sol = self.params['W_sol'], self.params['b_sol']
    
    loss, grads = 0.0, {}
    ############################################################################
    # TODO: Implement the forward and backward passes for the PDEsolveRDD.     #
    # In the forward pass you will need to do the following:                   #
    # (1) Use an affine transformation to compute the initial hidden state     #
    #     from the features. This should produce an array of shape (N, H)      #
    # (2) Use either a vanilla RNN or LSTM (depending on self.cell_type) to    #
    #     process the sequence in solutions_in and produce hidden state        #
    #     vectors for all timesteps, producing an array of shape (N, T-1, H).  #
    # (3) Use a (temporal) affine transformation to compute predictions        #
    #     at every timestep using the hidden states, giving an                 #
    #     array of shape (N, T-1, V).                                          #
    # (4) Use (temporal) softmax to compute loss using solutions_out, ignoring #
    #     the points where the output sol is <NULL> using the mask above.      #
    #                                                                          #
    # In the backward pass you will need to compute the gradient of the loss   #
    # with respect to all model parameters. Use the loss and grads variables   #
    # defined above to store loss and gradients; grads[k] should give the      #
    # gradients for self.params[k].                                            #
    ############################################################################
    # keep track of and initialize c
    
    # Forward pass
    h0, cache_aff = affine_forward(features, W_proj, b_proj)
    sol_embed, cache_embed = temporal_affine_forward(solutions_in, W_embed, b_embed)
    if self.cell_type == 'lstm':
      h, cache_lstm = lstm_forward(sol_embed, h0, Wx, Wh, b)
    else:
      h, cache_rnn = rnn_forward(sol_embed, h0, Wx, Wh, b)
    h_deco, cache_temp_aff_deco = temporal_affine_forward(h, W_deco, b_deco)
    solutions_pred, cache_temp_aff = temporal_affine_forward(h_deco, W_sol, b_sol)
    loss, dout = L2_norm_loss(solutions_pred, solutions_out)
    
    # Backward pass
    dh_deco, dW_sol, db_sol = temporal_affine_backward(dout, cache_temp_aff)
    dh, dW_deco, db_deco = temporal_affine_backward(dh_deco, cache_temp_aff_deco)
    if self.cell_type == 'lstm':
      dsol_embed, dh0, dWx, dWh, db = lstm_backward(dh, cache_lstm)
    else:
      dsol_embed, dh0, dWx, dWh, db = rnn_backward(dh, cache_rnn)
    dsols, dW_embed, db_embed = temporal_affine_backward(dsol_embed, cache_embed)
    dfeatures, dW_proj, db_proj = affine_backward(dh0, cache_aff)
    
    # Store gradients
    grads['W_proj'] = dW_proj
    grads['b_proj'] = db_proj
    grads['W_embed'] = dW_embed
    grads['b_embed'] = db_embed
    grads['Wx'] = dWx
    grads['Wh'] = dWh
    grads['b'] = db
    grads['W_deco'] = dW_deco
    grads['b_deco'] = db_deco
    grads['W_sol'] = dW_sol
    grads['b_sol'] = db_sol
    
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    return loss, grads


  def sample(self, features, max_length=501):
    """
    Run a test-time forward pass for the model, sampling solutions for input
    feature vectors.

    At each timestep, pass the current solution and the previous hidden
    state to the RNN to get the next hidden state, use the hidden state to get
    predictions at each step, and choose the current solution as
    the input sol. The initial hidden state is computed by applying an affine
    transform to the input features, and the initial sol is the zero vector.

    For LSTMs you will also have to keep track of the cell state; in that case
    the initial cell state should be zero.

    Inputs:
    - features: Array of input features of shape (N, D).
    - max_length: Maximum length T of generated solutions.

    Returns:
    - solutions: Array of shape (N, max_length) giving sampled solutions,
      where each element is an integer in the range [0, V). The first element
      of solutions should be the first solution, not the <START> token.
    """
    N = features.shape[0]
    _nul = self._null.shape[0]
    solutions = np.zeros((N, max_length, _nul), dtype=np.float32)

    # Unpack parameters
    W_proj, b_proj = self.params['W_proj'], self.params['b_proj']
    W_embed, b_embed = self.params['W_embed'], self.params['b_embed']
    Wx, Wh, b = self.params['Wx'], self.params['Wh'], self.params['b']
    W_deco, b_deco = self.params['W_deco'], self.params['b_deco']
    W_sol, b_sol = self.params['W_sol'], self.params['b_sol']
    
    ###########################################################################
    # TODO: Implement test-time sampling for the model. You will need to      #
    # initialize the hidden state of the RNN by applying the learned affine   #
    # transform to the input image features. The first sol that you feed to   #
    # the RNN should be the zero vector; its value is stored in the           #
    # variable self._start. At each timestep you will need to:                #
    # (1) Make an RNN step using the previous hidden state and the            #
    #     current sol to get the next hidden state.                           #
    # (2) Apply the learned affine transformation to the next hidden state to #
    #     get solutions                                                       #
    # (3) Select the current sol as the next input sol, writing it            #
    #     to the appropriate slot in the solutions variable                   #
    #                                                                         #
    # For simplicity, you do not need to stop generating after an <END> token #
    # is sampled, but you can if you want to.                                 #
    #                                                                         #
    # HINT: You will not be able to use the rnn_forward or lstm_forward       #
    # functions; you'll need to call rnn_step_forward or lstm_step_forward in #
    # a loop.                                                                 #
    ###########################################################################
    
    # Forward pass
    N, D = features.shape
    solutions[:, 0] = self._start
    h, _ = affine_forward(features, W_proj, b_proj)
    c = 0
    for i in range(1,max_length):
        capt_embed, _ = temporal_affine_forward(solutions[:, i-1], W_embed, b_embed)
        if self.cell_type == 'lstm':
          #print solutions[:, i-1].shape
          h, c, _ = lstm_step_forward(capt_embed, h, c, Wx, Wh, b)
        else:
          h, _ = rnn_step_forward(capt_embed, h, Wx, Wh, b)
        h_deco, _ = temporal_affine_forward(h, W_deco, b_deco)
        scores, _ = temporal_affine_forward(h_deco, W_sol, b_sol)
        solutions[:, i] = scores

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    return solutions
