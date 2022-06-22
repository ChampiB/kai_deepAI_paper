import theano
import theano.tensor as tensor
import numpy


# ADAM Optimizer, following Kingma & Ba (2015), c.f. https://arxiv.org/abs/1412.6980
class Adam:

    def __init__(self, b1, b2, alpha, epsilon=10e-8):
        self.b1 = b1
        self.b2 = b2
        self.alpha = alpha
        self.epsilon = epsilon

    def get_updates(self, grads, p):
        t = theano.shared(value=numpy.cast[theano.config.floatX](1.0))
        t_next = t + 1

        g = grads.astype(dtype=theano.config.floatX)
        m = theano.shared(
            value=numpy.zeros_like(p.get_value(), dtype=theano.config.floatX),
            name='m', borrow=True, broadcastable=p.broadcastable
        )
        m_next = self.b1 * m + (1 - self.b1) * g
        v = theano.shared(
            value=numpy.zeros_like(p.get_value(), dtype=theano.config.floatX),
            name='v', borrow=True, broadcastable=p.broadcastable
        )
        v_next = self.b2 * v + (1 - self.b2) * g * g
        m_ub = m / (1 - self.b1 ** t)
        v_ub = v / (1 - self.b2 ** t)
        update = p - self.alpha * m_ub / (tensor.sqrt(v_ub) + self.epsilon)
        return [(t, t_next), (m, m_next), (v, v_next), (p, update)]
