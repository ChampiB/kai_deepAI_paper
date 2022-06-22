from model.layers.DenseLayer import DenseLayer
from math_fc import functions as math_fc
from initialisation import weights
from theano.ifelse import ifelse
import theano.tensor as tensor


class TransitionNetwork:

    def __init__(self, config, theano_rng):

        self.n_proc = config['n_proc']
        self.sig_min_perturb = config['sig_min_perturbations']
        self.n_s = config['n_s']
        self.n_perturb = config['n_perturbations']
        self.shape = (self.n_perturb, self.n_s, self.n_proc)
        self.sig_min_states = config['sig_min_states']

        w_l1 = math_fc.create_tensor('Wl_stmu_stm1', 'init_ortho', [self.n_s, self.n_s], bc=(True, False, False))
        b_l1 = math_fc.create_tensor('bl_stmu', 'init_const', [self.n_s, 1], bc=(True, False, True))
        w_l2 = math_fc.create_tensor('Wl_stsig_stm1', 'init_weight', [self.n_s, self.n_s], bc=(True, False, False))
        b_l2 = math_fc.create_tensor('bl_stsig', 'init_const', [self.n_s, 1], bc=(True, False, True))

        self.params = [w_l1, b_l1, w_l2, b_l2]
        self.sigmas = weights.init_sigmas(self.params, self.n_perturb)
        self.r_params, self.r_epsilons = weights.randomize_parameters(
            self.params, self.sigmas, self.sig_min_perturb, theano_rng, self.n_perturb
        )

        self.l1 = DenseLayer(self.r_params[0], self.r_params[1], activation='tanh')
        self.l2 = DenseLayer(self.r_params[2], self.r_params[3], activation='soft_plus')

    def forward(self, x, t):
        # Compute mean and variance of prior over state.
        prior_mu = self.l1.forward(x, self.shape)
        prior_sig = self.l2.forward(x, self.shape) + self.sig_min_states

        # Explicitly encode expectations on homeostatic state variable
        prior_mu = ifelse(tensor.lt(t, 20), prior_mu, tensor.set_subtensor(prior_mu[:, 0, :], 0.1))
        prior_sig = ifelse(tensor.lt(t, 20), prior_sig, tensor.set_subtensor(prior_sig[:, 0, :], 0.005))

        return prior_mu, prior_sig
