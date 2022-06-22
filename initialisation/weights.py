import numpy
import scipy
import theano
import theano.tensor as tensor


# Xavier initialization
def init_xavier(shape):
    val = numpy.random.randn(*shape) / numpy.sqrt(shape[1])
    return val.astype(theano.config.floatX)


# Unitary transform
def init_ortho(shape):
    x = numpy.random.normal(0.0, 0.1, shape)
    xo = scipy.linalg.orth(x)
    return xo.astype(theano.config.floatX)


# Uniformly distributed weight
def init_weight(shape, min_val=-0.05, max_val=0.05):
    val = numpy.random.rand(*shape)
    val = min_val + (max_val - min_val) * val
    return val.astype(theano.config.floatX)


# Constant weight
def init_const(shape, val=0.0):
    val = val * numpy.ones(shape, dtype=theano.config.floatX)
    return val.astype(theano.config.floatX)


def init_sigmas(params, init_sig_perturbations):
    sigmas = []
    for param in params:
        value = init_sig_perturbations * numpy.ones(param.get_value().shape).astype(dtype=theano.config.floatX)
        sigma = theano.shared(name='sigma_' + param.name, value=value, borrow=True, broadcastable=param.broadcastable)
        sigmas.append(sigma)
    return sigmas


def randomize_parameters(params, sigmas, sig_min_perturbations, theano_rng, n_perturbations):
    r_params = []
    r_epsilons = []
    for i in range(len(params)):
        epsilon_half = theano_rng.normal(
            [int(n_perturbations / 2), int(params[i].shape[1].eval()), int(params[i].shape[2].eval())],
            dtype=theano.config.floatX
        )
        r_epsilon = tensor.concatenate([epsilon_half, -1.0 * epsilon_half], axis=0)
        r_param = params[i] + r_epsilon * (tensor.nnet.softplus(sigmas[i]) + sig_min_perturbations)
        r_params.append(r_param)
        r_epsilons.append(r_epsilon)
    return r_params, r_epsilons
