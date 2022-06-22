from math_fc import functions as math_fc
import theano.tensor as tensor
import theano


class MountainCar:

    def __init__(self, config):
        self.shape = (config['n_perturbations'], config['n_o'], config['n_proc'])
        self.pos_t = math_fc.create_tensor('pos_t0', 'init_const', self.shape, bc=None, val=-0.5)
        self.v_t = math_fc.create_tensor('v_t0', 'init_const', self.shape, bc=None, val=0)

    def step(self, dai, action):
        # Update car's position and velocity
        action_force = tensor.tanh(action)
        force = tensor.switch(
            tensor.lt(self.pos_t, 0.0),
            -2 * self.pos_t - 1,
            - tensor.pow(1 + 5 * tensor.sqr(self.pos_t), -0.5)
            - tensor.sqr(self.pos_t) * tensor.pow(1 + 5 * tensor.sqr(self.pos_t), -1.5)
            - tensor.pow(self.pos_t, 4) / 16.0
        ) - 0.25 * self.v_t
        self.v_t = self.v_t + 0.05 * force + 0.03 * action_force
        self.pos_t = self.pos_t + self.v_t

        # Generate sensory inputs
        ot = self.pos_t + dai.theano_rng.normal(self.shape) * 0.01
        oht = tensor.exp(-tensor.sqr(self.pos_t - 1.0) * 5.55555)
        return ot, oht
