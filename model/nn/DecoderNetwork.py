from model.layers.DenseLayer import DenseLayer
from math_fc import functions as math_fc
from initialisation import weights


class DecoderNetwork:

    def __init__(self, config, theano_rng):

        self.sig_min_obs = config['sig_min_obs']
        self.sig_min_perturb = config['sig_min_perturbations']
        self.n_proc = config['n_proc']
        self.n_s = config['n_s']
        self.n_o = config['n_o']
        self.n_oh = config['n_oh']
        self.n_oa = config['n_oa']
        self.n_perturb = config['n_perturbations']
        self.shape = (self.n_perturb, self.n_s, self.n_proc)
        self.init_sig_obs = config['init_sig_obs']

        w_l1 = math_fc.create_tensor('Wl_ost_st', 'init_ortho', [self.n_s, self.n_s], bc=(True, False, False))
        b_l1 = math_fc.create_tensor('bl_ost', 'init_const', [self.n_s, 1], bc=(True, False, True))
        w_l2 = math_fc.create_tensor('Wl_ost2_ost', 'init_ortho', [self.n_s, self.n_s], bc=(True, False, False))
        w_l3 = math_fc.create_tensor('Wl_ost3_ost2', 'init_ortho', [self.n_s, self.n_s], bc=(True, False, False))
        w_l4 = math_fc.create_tensor('Wl_otmu_st', 'init_weight', [self.n_o, self.n_s], bc=(True, False, False))
        w_l5 = math_fc.create_tensor('Wl_otsig_st', 'init_weight', [self.n_o, self.n_s], bc=(True, False, False))
        w_l6 = math_fc.create_tensor('Wl_ohtmu_st', 'init_weight', [self.n_oh, self.n_s], bc=(True, False, False))
        w_l7 = math_fc.create_tensor('Wl_ohtsig_st', 'init_weight', [self.n_oh, self.n_s], bc=(True, False, False))
        w_l8 = math_fc.create_tensor('Wl_oatmu_st', 'init_weight', [self.n_oa, self.n_s], bc=(True, False, False))
        w_l9 = math_fc.create_tensor('Wl_oatsig_st', 'init_weight', [self.n_oa, self.n_s], bc=(True, False, False))
        b_l2 = math_fc.create_tensor('bl_ost2', 'init_const', [self.n_s, 1], bc=(True, False, True))
        b_l3 = math_fc.create_tensor('bl_ost3', 'init_const', [self.n_s, 1], bc=(True, False, True))
        b_l4 = math_fc.create_tensor('bl_otmu', 'init_const', [self.n_o, 1], bc=(True, False, True))
        b_l5 = math_fc.create_tensor('bl_otsig', 'init_const', [self.n_o, 1], bc=(True, False, True))
        b_l6 = math_fc.create_tensor('bl_ohtmu', 'init_const', [self.n_oh, 1], bc=(True, False, True))
        b_l7 = math_fc.create_tensor('bl_ohtsig', 'init_const', [self.n_oh, 1], bc=(True, False, True), val=self.init_sig_obs)
        b_l8 = math_fc.create_tensor('bl_oatmu', 'init_const', [self.n_oa, 1], bc=(True, False, True))
        b_l9 = math_fc.create_tensor('bl_oatsig', 'init_const', [self.n_oa, 1], bc=(True, False, True), val=self.init_sig_obs)

        self.params = [w_l1, b_l1, w_l2, b_l2, w_l3, b_l3, w_l4, b_l4, w_l5, b_l5, w_l6, b_l6, w_l7, b_l7, w_l8, b_l8, w_l9, b_l9]
        self.sigmas = weights.init_sigmas(self.params, self.n_perturb)
        self.r_params, self.r_epsilons = weights.randomize_parameters(
            self.params, self.sigmas, self.sig_min_perturb, theano_rng, self.n_perturb
        )

        self.l1 = DenseLayer(self.r_params[0], self.r_params[1], activation='relu')
        self.l2 = DenseLayer(self.r_params[2], self.r_params[3], activation='relu')
        self.l3 = DenseLayer(self.r_params[4], self.r_params[5], activation='relu')
        self.l4 = DenseLayer(self.r_params[6], self.r_params[7])
        self.l5 = DenseLayer(self.r_params[8], self.r_params[9], activation='soft_plus')
        self.l6 = DenseLayer(self.r_params[10], self.r_params[11])
        self.l7 = DenseLayer(self.r_params[12], self.r_params[13], activation='soft_plus')
        self.l8 = DenseLayer(self.r_params[14], self.r_params[15])
        self.l9 = DenseLayer(self.r_params[16], self.r_params[17], activation='soft_plus')

    def forward(self, x):
        ost = self.l1.forward(x, self.shape)
        ost2 = self.l2.forward(ost, self.shape)
        ost3 = self.l3.forward(ost2, self.shape)
        otmu = self.l4.forward(ost3, self.shape)
        otsig = self.l5.forward(ost3, self.shape) + self.sig_min_obs
        ohtmu = self.l6.forward(ost3, self.shape)
        ohtsig = self.l7.forward(ost3, self.shape) + self.sig_min_obs
        oatmu = self.l8.forward(ost3, self.shape)
        oatsig = self.l9.forward(ost3, self.shape) + self.sig_min_obs
        return otmu, otsig, ohtmu, ohtsig, oatmu, oatsig
