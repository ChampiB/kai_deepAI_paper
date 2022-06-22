import theano
import theano.tensor as T
from initialisation import weights
import numpy


def gaussian_nll(y, mu, sig):
    """
    Compute the negative log likelihood of a Gaussian distribution.
    :param y: the input.
    :param mu: the mean parameter.
    :param sig: the diagonal of the covariance matrix.
    :return: the negative log likelihood.
    """
    return 0.5 * T.sum(T.sqr(y - mu) / sig ** 2 + 2 * T.log(sig) + T.log(2 * numpy.pi), axis=1)


def kl_gaussian(mu1, sig1, mu2, sig2):
    """
    Compute the KL-divergence between two Gaussian.
    :param mu1: the mean vector of the first Gaussian.
    :param sig1: the diagonal of the covariance matrix of the first Gaussian.
    :param mu2: the mean vector of the second Gaussian.
    :param sig2: the diagonal of the covariance matrix of the second Gaussian.
    :return: the KL-divergence.
    """
    return T.sum(0.5 * (2 * T.log(sig2) - 2 * T.log(sig1) + (sig1 ** 2 + (mu1 - mu2 )**2) / sig2**2 - 1), axis=1)


def create_tensor(name, initialisation_type, shape, bc=(True, False, False), val=0, min_val=-0.05, max_val=0.05):
    # Get the parameters values.
    value = None
    if initialisation_type == "init_weight":
        value = weights.init_weight(shape, -0.5, 0.5)
    if initialisation_type == 'init_ortho':
        value = weights.init_ortho(shape)
    if initialisation_type == 'init_const':
        value = weights.init_const(shape, val)
    if initialisation_type == 'init_weight':
        value = weights.init_weight(shape, min_val, max_val)
    if len(shape) <= 2:
        value = value.reshape(1, shape[0], shape[1])

    # Create the theano tensor.
    if bc is None:
        return theano.shared(value=value, name=name, borrow=True)
    else:
        return theano.shared(value=value, name=name, borrow=True, broadcastable=bc)


def compute_delta(variance, config, epsilon, fe_t):
    normalisation = T.nnet.softplus(variance) + config['sig_min_perturbations']
    return T.tensordot(fe_t, epsilon, axes=[[0], [0]]) / normalisation / config['n_perturbations']


def compute_delta_sigma(sigma, config, epsilon, fe_mean_perturb):
    normalisation = T.nnet.softplus(sigma) + config['sig_min_perturbations']
    outer_der = (epsilon * epsilon - 1.0) / normalisation
    inner_der = T.exp(sigma) / (1.0 + T.exp(sigma))
    return T.tensordot(fe_mean_perturb, outer_der * inner_der, axes=[[0], [0]]) / config['n_perturbations']
