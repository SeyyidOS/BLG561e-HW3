from .layers_with_weights import LayerWithWeights
import numpy as np


class RNNLayer(LayerWithWeights):
    """ Simple RNN Layer - only calculates hidden states """

    def __init__(self, in_size, out_size):
        """ RNN Layer constructor
        Args:
            in_size: input feature dimension - D
            out_size: hidden state dimension - H
        """
        self.in_size = in_size
        self.out_size = out_size
        self.Wx = np.random.rand(in_size, out_size)
        self.Wh = np.random.rand(out_size, out_size)
        self.b = np.random.rand(out_size)
        self.cache = None
        self.grad = {'dx': None, 'dh0': None, 'dWx': None, 'dWh': None, 'db': None}

    def forward_step(self, x, prev_h):
        """ Forward pass for a single timestep
        Args:
            x: input, of shape (N, D)
            prev_h: previous hidden state, of shape (N, H)
        Returns:
            next_h: next hidden state, of shape (N, H)
            cache: Values necessary for backpropagation, tuple
        """
        next_h = np.tanh(self.b + np.dot(prev_h, self.Wh) + np.dot(x, self.Wx))
        cache = (x, prev_h, next_h)
        return next_h, cache

    def forward(self, x, h0):
        """ Forward pass for the whole data sequence (of length T) of size minibatch N
        Values necessary in backpropagation need to be kept in self.cache as a list
        Args:
            x: input, of shape (N, T, D)
            h0: initial hidden state, of shape (N, H)
        Returns:
            h: hidden states of whole sequence, of shape (N, T, H)
        """
        N, T, D = x.shape
        H = h0.shape[1]
        h = np.zeros((N, T, H))
        self.cache = []
        prev_h = h0

        for t in range(T):
            next_h, cache = self.forward_step(x[:, t, :], prev_h)
            h[:, t, :] = next_h
            self.cache.append(cache)
            prev_h = next_h

        return h

    def backward_step(self, dnext_h, cache):
        """ Backward pass for a single timestep
        Args:
            dnext_h: gradient of loss with respect to
                     hidden state, of shape (N, H)
            cache: necessary values from last forward pass
        Returns:
            dx: gradients of input, of shape (N, D)
            dprev_h: gradients of previous hidden state, of shape (N, H)
            dWx: gradients of weights Wx, of shape (D, H)
            dWh: gradients of weights Wh, of shape (H, H)
            db: gradients of bias b, of shape (H,)
        """
        x, prev_h, next_h = cache
        dtanh = dnext_h * (1 - next_h**2)
        db = np.sum(dtanh, axis=0)
        dWx = np.dot(x.T, dtanh)
        dWh = np.dot(prev_h.T, dtanh)
        dx = np.dot(dtanh, self.Wx.T)
        dprev_h = np.dot(dtanh, self.Wh.T)
        return dx, dprev_h, dWx, dWh, db

    def backward(self, dh):
        """ Backward pass for whole sequence
        Necessary data for backpropagation should be obtained from self.cache
        Args:
            dh: gradients of all hidden states, of shape (N, T, H)
        Calculates gradients and saves them to the dictionary self.grad
        self.grad = {
            dx: gradients of inputs, of shape (N, T, D)
            dh0: gradients of initial hidden state, of shape (N, H)
            dWx: gradients of weights Wx, of shape (D, H)
            dWh: gradients of weights Wh, of shape (H, H)
            db: gradients of bias b, of shape (H,)
            }
        """
        N, T, H = dh.shape
        D = self.in_size

        dx = np.zeros((N, T, D))
        dWx = np.zeros_like(self.Wx)
        dWh = np.zeros_like(self.Wh)
        db = np.zeros_like(self.b)
        dprev_h_t = np.zeros((N, H))

        for t in reversed(range(T)):
            dnext_h = dh[:, t, :] + dprev_h_t
            cache = self.cache[t]
            dx_t, dprev_h_t, dWx_t, dWh_t, db_t = self.backward_step(dnext_h, cache)

            dx[:, t, :] = dx_t
            dWx += dWx_t
            dWh += dWh_t
            db += db_t

        self.grad = {'dx': dx, 'dh0': dprev_h_t, 'dWx': dWx, 'dWh': dWh, 'db': db}


class LSTMLayer(LayerWithWeights):
    """ Simple LSTM Layer - only calculates hidden states and cell states """

    def __init__(self, in_size, out_size):
        """ LSTM Layer constructor
        Args:
            in_size: input feature dimension - D
            out_size: hidden state dimension - H
        """
        self.in_size = in_size
        self.out_size = out_size
        self.Wx = np.random.rand(in_size, 4 * out_size)
        self.Wh = np.random.rand(out_size, 4 * out_size)
        self.b = np.random.rand(4 * out_size)
        self.cache = None
        self.grad = {'dx': None, 'dh0': None, 'dWx': None,
                     'dWh': None, 'db': None}

    def forward_step(self, x, prev_h, prev_c):
        """ Forward pass for a single timestep
        Args:
            x: input, of shape (N, D)
            prev_h: previous hidden state, of shape (N, H)
            prev_c: previous cell state, of shape (N, H)
        Returns:
            next_h: next hidden state, of shape (N, H)
            next_c: next cell state, of shape (N, H)
            cache: Values necessary for backpropagation, tuple
        """
        
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        A = np.dot(x, self.Wx) + np.dot(prev_h, self.Wh) + self.b
        H = prev_h.shape[1]
        i = sigmoid(A[:, :H])
        f = sigmoid(A[:, H:2*H])
        o = sigmoid(A[:, 2*H:3*H])
        g = np.tanh(A[:, 3*H:])
        next_c = f * prev_c + i * g
        next_h = o * np.tanh(next_c)
        cache = (x, prev_h, prev_c, i, f, o, g, next_c)
        return next_h, next_c, cache

    def forward(self, x, h0):
        """ Forward pass for the whole data sequence (of length T) of size minibatch N
        Values necessary in backpropagation need to be kept in self.cache as a list
        Cell state should be initialized to 0.
        Args:
            x: input, of shape (N, T, D)
            h0: initial hidden state, of shape (N, H)
        Returns:
            h: hidden states of whole sequence, of shape (N, T, H)
        """
        N, T, D = x.shape
        H = h0.shape[1]
        h = np.zeros((N, T, H))
        self.cache = []
        prev_h = h0
        prev_c = np.zeros_like(h0)

        for t in range(T):
            next_h, next_c, cache = self.forward_step(x[:, t, :], prev_h, prev_c)
            h[:, t, :] = next_h
            self.cache.append(cache)
            prev_h = next_h
            prev_c = next_c

        return h

    def backward_step(self, dnext_h, dnext_c, cache):
        """ Backward pass for a single timestep
        Args:
            dnext_h: gradient of loss with respect to
                     hidden state, of shape (N, H)
            dnext_c: gradient of loss with respect to
                     cell state, of shape (N, H)
            cache: necessary values from last forward pass
        Returns:
            dx: gradients of input, of shape (N, D)
            dprev_h: gradients of previous hidden state, of shape (N, H)
            dprev_c: gradients of previous cell state, of shape (N, H)
            dWx: gradients of weights Wx, of shape (D, 4H)
            dWh: gradients of weights Wh, of shape (H, 4H)
            db: gradients of bias b, of shape (4H,)
        """
        x, prev_h, prev_c, i, f, o, g, next_c = cache
        dnext_c += dnext_h * o * (1 - np.tanh(next_c)**2)
        di = dnext_c * g * i * (1 - i)
        df = dnext_c * prev_c * f * (1 - f)
        do = dnext_h * np.tanh(next_c) * o * (1 - o)
        dg = dnext_c * i * (1 - g**2)
        dA = np.hstack((di, df, do, dg))
        db = np.sum(dA, axis=0)
        dWx = np.dot(x.T, dA)
        dWh = np.dot(prev_h.T, dA)
        dx = np.dot(dA, self.Wx.T)
        dprev_h = np.dot(dA, self.Wh.T)
        dprev_c = dnext_c * f
        return dx, dprev_h, dprev_c, dWx, dWh, db

    def backward(self, dh):
        """ Backward pass for whole sequence
        Necessary data for backpropagation should be obtained from self.cache
        Args:
            dh: gradients of all hidden states, of shape (N, T, H)
        Calculates gradients and saves them to the dictionary self.grad
        self.grad = {
            dx: gradients of inputs, of shape (N, T, D)
            dh0: gradients of initial hidden state, of shape (N, H)
            dWx: gradients of weights Wx, of shape (D, 4H)
            dWh: gradients of weights Wh, of shape (H, 4H)
            db: gradients of bias b, of shape (4H,)
            }
        """
        N, T, H = dh.shape
        D = self.in_size

        dx = np.zeros((N, T, D))
        dWx = np.zeros_like(self.Wx)
        dWh = np.zeros_like(self.Wh)
        db = np.zeros_like(self.b)
        dprev_h_t = np.zeros((N, H))
        dprev_c_t = np.zeros((N, H))

        for t in reversed(range(T)):
            dnext_h = dh[:, t, :] + dprev_h_t
            cache = self.cache[t]
            dx_t, dprev_h_t, dprev_c_t, dWx_t, dWh_t, db_t = self.backward_step(dnext_h, dprev_c_t, cache)

            dx[:, t, :] = dx_t
            dWx += dWx_t
            dWh += dWh_t
            db += db_t

        self.grad = {'dx': dx, 'dh0': dprev_h_t, 'dWx': dWx, 'dWh': dWh, 'db': db}