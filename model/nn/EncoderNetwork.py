from model.layers.DenseLayer import DenseLayer
from math_fc import functions as math_fc
import theano.tensor as T
from initialisation import weights


class EncoderNetwork:

    def __init__(self, config, theano_rng):

        self.sig_min_states = config['sig_min_states']
        self.sig_min_perturb = config['sig_min_perturbations']
        self.n_proc = config['n_proc']
        self.n_s = config['n_s']
        self.n_o = config['n_o']
        self.n_oh = config['n_oh']
        self.n_oa = config['n_oa']
        self.n_perturb = config['n_perturbations']
        self.input_shape = [
            (self.n_perturb, self.n_s, self.n_proc),
            (self.n_perturb, self.n_o, self.n_proc),
            (self.n_perturb, self.n_oh, self.n_proc),
            (self.n_perturb, self.n_oa, self.n_proc)
        ]
        self.shape = (self.n_perturb, self.n_s, self.n_proc)
        self.init_sig_states = config['init_sig_states']
        self.theano_rng = theano_rng

        w_l1_ot = math_fc.create_tensor('Wq_hst_ot', 'init_weight', [self.n_s, self.n_o], bc=(True, False, False), min_val=-0.5, max_val=0.5)
        w_l1_stm1 = math_fc.create_tensor('Wq_hst_stm1', 'init_ortho', [self.n_s, self.n_s], bc=(True, False, False))
        w_l1_oht = math_fc.create_tensor('Wq_hst_oht', 'init_weight', [self.n_s, self.n_oh], bc=(True, False, False), min_val=-0.5, max_val=0.5)
        w_l1_oat = math_fc.create_tensor('Wq_hst_oat', 'init_weight', [self.n_s, self.n_oa], bc=(True, False, False), min_val=-0.5, max_val=0.5)
        b_l1 = math_fc.create_tensor('bq_hst', 'init_ortho', [self.n_s, 1], bc=(True, False, True))
        w_l2 = math_fc.create_tensor('Wq_hst2_hst', 'init_ortho', [self.n_s, self.n_s], bc=(True, False, False))
        b_l2 = math_fc.create_tensor('bq_hst2', 'init_const', [self.n_s, 1], bc=(True, False, True))
        w_l3 = math_fc.create_tensor('Wq_stmu_hst2', 'init_ortho', [self.n_s, self.n_s], bc=(True, False, False))
        b_l3 = math_fc.create_tensor('bq_stmu', 'init_const', [self.n_s, 1], bc=(True, False, True))
        w_l4 = math_fc.create_tensor('Wq_stsig_hst2', 'init_weight', [self.n_s, self.n_s], bc=(True, False, False))
        b_l4 = math_fc.create_tensor('bq_stsig', 'init_const', [self.n_s, 1], val=self.init_sig_states, bc=(True, False, True))

        self.params = [w_l1_stm1, w_l1_ot, w_l1_oht, w_l1_oat, b_l1, w_l2, b_l2, w_l3, b_l3, w_l4, b_l4]
        self.sigmas = weights.init_sigmas(self.params, self.n_perturb)
        self.r_params, self.r_epsilons = weights.randomize_parameters(
            self.params, self.sigmas, self.sig_min_perturb, theano_rng, self.n_perturb
        )

        self.l1 = DenseLayer(
            [self.r_params[0], self.r_params[1], self.r_params[2], self.r_params[3]], self.r_params[4],
            activation='relu'
        )
        self.l2 = DenseLayer(self.r_params[5], self.r_params[6], activation='relu')
        self.l3 = DenseLayer(self.r_params[7], self.r_params[8], activation='tanh')
        self.l4 = DenseLayer(self.r_params[9], self.r_params[10], activation='soft_plus')

    def forward(self, stm1, ot, oht, oat):
        hst = self.l1.forward([stm1, ot, oht, oat], self.input_shape)
        hst2 = self.l2.forward(hst, self.shape)
        mu = self.l3.forward(hst2, self.shape)
        sig = self.l4.forward(hst2, self.shape) + self.sig_min_states

        # Explicitly encode position as homeostatic state variable
        mu = T.set_subtensor(mu[:, 0, :], 0.1 * ot[:, 0, :]).reshape(self.shape)
        sig = T.set_subtensor(sig[:, 0, :], 0.005).reshape(self.shape)

        # Sample a state
        st = mu + self.theano_rng.normal((self.n_perturb, self.n_s, self.n_proc)) * sig

        return st, mu, sig
