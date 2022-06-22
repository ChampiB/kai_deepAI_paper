from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import theano.tensor as tensor
import theano
import numpy as np


def init_sigmas(params, init_sig_perturbations):
    sigmas = []
    for param in params:
        value = init_sig_perturbations * np.ones(param.get_value().shape).astype(dtype=theano.config.floatX)
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


if __name__ == '__main__':

    ii32 = np.iinfo(np.int32)
    theano_rng = RandomStreams(np.random.randint(ii32.max))

    w1 = theano.shared(name="w1", value=np.ones(shape=(1, 10, 10)), borrow=True, broadcastable=(True, False, False))  # (1,n_s,n_s)
    w2 = theano.shared(name="w2", value=np.ones(shape=(1, 10, 1)), borrow=True, broadcastable=(True, False, False))   # (1,n_s,n_o)
    w3 = theano.shared(name="w3", value=np.ones(shape=(1, 10, 1)), borrow=True, broadcastable=(True, False, False))   # (1,n_s,n_oh)
    w4 = theano.shared(name="w4", value=np.ones(shape=(1, 10, 1)), borrow=True, broadcastable=(True, False, False))   # (1,n_s,n_oa)
    w = [w1, w2, w3, w4]

    sigmas = init_sigmas(w, 10000)
    w, _ = randomize_parameters(w, sigmas, -3, theano_rng, 10000)

    shape = [
        (10000, 10, 1),
        (10000, 1, 1),
        (10000, 1, 1),
        (10000, 1, 1)
    ]
    x1 = tensor.ones(shape=(10000, 10, 1))  # (n_perturbations,n_s, n_proc)
    x2 = tensor.ones(shape=(10000, 1, 1))   # (n_perturbations,n_o, n_proc)
    x3 = tensor.ones(shape=(10000, 1, 1))   # (n_perturbations,n_oh, n_proc)
    x4 = tensor.ones(shape=(10000, 1, 1))   # (n_perturbations,n_oa, n_proc)
    x = [x1, x2, x3, x4]

    print("0")
    print(w1.broadcastable)
    y0 = tensor.batched_tensordot(w[0], tensor.reshape(x[0], shape[0]), axes=[[2], [1]])
    print(y0.eval().shape)

    print("1")
    y1 = tensor.batched_tensordot(w[1], tensor.reshape(x[1], shape[1]), axes=[[2], [1]])
    print(y1.eval().shape)

    print("2")
    y2 = tensor.batched_tensordot(w[2], tensor.reshape(x[2], shape[2]), axes=[[2], [1]])
    print(y2.eval().shape)

    print("3")
    y3 = tensor.batched_tensordot(w[3], tensor.reshape(x[3], shape[3]), axes=[[2], [1]])
    print(y3.eval().shape)

    print("3")

    #y = None
    #for i in range(len(x)):
    #    if y is None:
    #        y = tensor.batched_tensordot(w[i], tensor.reshape(x[i], shape[i]), axes=[[2], [1]])
    #    else:
    #        y += tensor.batched_tensordot(w[i], tensor.reshape(x[i], shape[i]), axes=[[2], [1]])
    #print(y.eval())
