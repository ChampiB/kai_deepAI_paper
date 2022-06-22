from model.layers.DenseLayer import DenseLayer
from math_fc import functions as math_fc
from initialisation import weights


class PolicyNetwork:

    def __init__(self, config, theano_rng):

        self.sig_min_action = config['sig_min_action']
        self.sig_min_perturb = config['sig_min_perturbations']
        self.n_proc = config['n_proc']
        self.n_s = config['n_s']
        self.n_oa = config['n_oa']
        self.n_perturb = config['n_perturbations']
        self.init_sig_action = config['init_sig_action']

        w_l1 = math_fc.create_tensor('Wa_aht_st', 'init_ortho', [self.n_s, self.n_s], bc=(True, False, False))
        b_l1 = math_fc.create_tensor('ba_aht', 'init_ortho', [self.n_s, 1], bc=(True, False, True))
        w_l2 = math_fc.create_tensor('Wa_atmu_aht', 'init_weight', [self.n_oa, self.n_s], bc=(True, False, False), min_val=-1.0, max_val=1.0)
        b_l2 = math_fc.create_tensor('ba_atmu', 'init_const', [self.n_oa, 1], bc=(True, False, True))
        w_l3 = math_fc.create_tensor('Wa_atsig_aht', 'init_weight', [self.n_oa, self.n_s], bc=(True, False, False))
        b_l3 = math_fc.create_tensor('ba_atsig', 'init_const', [self.n_oa, 1], bc=(True, False, True), val=self.init_sig_action)

        self.params = [w_l1, b_l1, w_l2, b_l2, w_l3, b_l3]
        self.sigmas = weights.init_sigmas(self.params, self.n_perturb)
        self.r_params, self.r_epsilons = weights.randomize_parameters(
            self.params, self.sigmas, self.sig_min_perturb, theano_rng, self.n_perturb
        )

        self.l1 = DenseLayer(self.r_params[0], self.r_params[1])
        self.l2 = DenseLayer(self.r_params[2], self.r_params[3])
        self.l3 = DenseLayer(self.r_params[4], self.r_params[5], activation='soft_plus')

        self.theano_rng = theano_rng

    def forward(self, x):
        aht = self.l1.forward(x, (self.n_perturb, self.n_s, self.n_proc))
        at_mu = self.l2.forward(aht, (self.n_perturb, self.n_s, self.n_proc))
        at_sig = self.l3.forward(aht, (self.n_perturb, self.n_s, self.n_proc)) + self.sig_min_action
        at = self.theano_rng.normal((self.n_perturb, self.n_oa, self.n_proc), avg=at_mu, std=at_sig)
        return at, at_mu, at_sig
