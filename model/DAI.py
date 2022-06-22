import theano
import pickle
from math_fc import functions as math_fc
from model.nn.PolicyNetwork import PolicyNetwork
from model.nn.DecoderNetwork import DecoderNetwork
from model.nn.EncoderNetwork import EncoderNetwork
from model.nn.TransitionNetwork import TransitionNetwork
import numpy
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams


class DAI:

    def __init__(self, config, env):
        # Initialize random generator.
        ii32 = numpy.iinfo(numpy.int32)
        self.theano_rng = RandomStreams(numpy.random.randint(ii32.max))

        # Store environment.
        self.env = env

        # Create the various deep neural networks.
        self.policy = PolicyNetwork(config, self.theano_rng)
        self.decoder = DecoderNetwork(config, self.theano_rng)
        self.encoder = EncoderNetwork(config, self.theano_rng)
        self.transition = TransitionNetwork(config, self.theano_rng)

    def action_perception_cycle(self, t, stm1):
        # Feed the state into the policy to generate an action
        at, at_mu, at_sig = self.policy.forward(stm1)

        # Update environment
        ot, oht = self.env.step(self, at)

        # Compute posterior over hidden state from last hidden state and current observations.
        st, st_mu, st_sig = self.encoder.forward(stm1, ot, oht, at)

        # Calculate parameters of likelihood distributions from sampled state
        ot_mu, ot_sig, oht_mu, oht_sig, oat_mu, oat_sig = self.decoder.forward(stm1)

        # Calculate prior distribution of the hidden state from previous state
        prior_mu, prior_sig = self.transition.forward(stm1, t)

        # Compute the variational free energy
        p_ot = math_fc.gaussian_nll(ot, ot_mu, ot_sig)
        p_oht = math_fc.gaussian_nll(oht, oht_mu, oht_sig)
        p_oat = math_fc.gaussian_nll(at, oat_mu, oat_sig)
        kl_st = math_fc.kl_gaussian(st_mu, st_sig, prior_mu, prior_sig)
        fe_t = kl_st + p_ot + p_oht + p_oat

        return st, fe_t, kl_st, p_ot, p_oht, p_oat

    def epsilons(self):
        return self.encoder.r_epsilons + self.transition.r_epsilons + self.decoder.r_epsilons + self.policy.r_epsilons

    def params(self):
        return self.encoder.params + self.transition.params + self.decoder.params + self.policy.params

    def sigmas(self):
        return self.encoder.sigmas + self.transition.sigmas + self.decoder.sigmas + self.policy.sigmas

    def initial_states(self):
        value = numpy.zeros(self.decoder.shape).astype(dtype=theano.config.floatX)
        if self.decoder.n_proc == 1:
            return theano.shared(name='s_t0', value=value, borrow=True, broadcastable=(False, False, True))
        else:
            return theano.shared(name='s_t0', value=value, borrow=True)

    def save_model(self, filename):
        with open(filename, 'wb') as f:
            for param in self.params():
                pickle.dump(param.get_value(borrow=True), f, -1)
            for sigma in self.sigmas():
                pickle.dump(sigma.get_value(borrow=True), f, -1)

    def load_model(self, filename):
        with open(filename, 'r') as f:
            for param in self.params():
                param.set_value(pickle.load(f), borrow=True)
            for sigma in self.sigmas():
                sigma.set_value(pickle.load(f), borrow=True)

    def save(self, config, o_fe_mean, fe_min, i):
        # Save current parameters every nth loop
        if i % config['saving_steps'] == 0:
            self.save_model(config['base_name'] + '_%d.pkl' % i)
            self.save_model(config['base_name'] + '_current.pkl')

        # Save best parameters
        if fe_min is None or o_fe_mean < fe_min:
            fe_min = o_fe_mean
            self.save_model(config['base_name'] + '_best.pkl')
            if config['save_best_trajectory']:
                self.save_model(config['base_name'] + '_best_%d.pkl' % i)
        return fe_min
