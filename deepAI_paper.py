"""
Implementation of Deep Active Inference for General Artificial Intelligence
Kai Ueltzhoeffer, 2017
"""

from optimiser import Adam
from math_fc import functions as math_fc
import timeit
import numpy as np
import theano
from model import DAI
from env import MountainCar as Env


def train_dai():
    # Create the project config
    config = {
        'n_s': 10,
        'base_name': 'deepAI_paper',  # Name for saves and logfile
        'learning_rate': 1e-3,  # Learning Rate
        'saving_steps': 10,  # Save progress every nth step
        'save_best_trajectory': True,  # Save best trajectory of parameter sets
        'n_run_steps': 30,  # No. of time steps to simulate
        'n_proc': 1,  # No. of processes to simulate for each Sample from the population density
        'n_perturbations': 10000,  # No. of samples from population density per iteration
        'n_o': 1,  # Sensory Input encoding Position
        'n_oh': 1,  # Transformed channel after non-linearity (OPTIONAL!)
        'n_oa': 1,  # Proprioception (OPTIONAL!)
        'sig_min_obs': 1e-6,  # Minimum value of standard deviations, to prevent division by zero
        'sig_min_states': 1e-6,  # Minimum value of standard deviations, to prevent division by zero
        'sig_min_action': 1e-6,  # Minimum value of standard deviations, to prevent division by zero
        'init_sig_obs': 0.0,
        'init_sig_states_likelihood': 0.0,
        'init_sig_states': -3.0,
        'init_sig_action': -3.0,
        'sig_min_perturbations': 1e-6,
        'init_sig_perturbations': -3.0,
        'n_steps': 1000000,  # Max. number of optimization steps
    }

    # Create the DAI agent, the mountain car environment, and the Adam optimiser.
    env = Env.MountainCar(config)
    dai = DAI.DAI(config, env)
    optimiser = Adam.Adam(0.9, 0.999, config['learning_rate'], epsilon=10e-6)

    # Compute variational free energy.
    (_, fe_t, kl_st, p_ot, p_oht, p_oat), _ = theano.scan(
        fn=dai.action_perception_cycle,
        sequences=[np.arange(config['n_run_steps'], dtype=theano.config.floatX)],
        outputs_info=[dai.initial_states()] + [None] * 5
    )
    fe_t_mean = fe_t.mean(axis=0).mean(axis=1)
    fe_t = fe_t.mean()
    kl_st = kl_st.mean()
    p_ot = p_ot.mean()
    p_oht = p_oht.mean()
    p_oat = p_oat.mean()

    # Create list of updates
    updates = []
    for i, (param, sigma, epsilon) in enumerate(zip(dai.params(), dai.sigmas(), dai.epsilons())):
        # Compute deltas
        delta = math_fc.compute_delta(sigma, config, epsilon, fe_t_mean)
        delta_sigma = math_fc.compute_delta_sigma(sigma, config, epsilon, fe_t_mean)

        # Use adam optimiser
        updates = updates + optimiser.get_updates(delta, param) + optimiser.get_updates(delta_sigma, sigma)

    # Define Training Function
    train = theano.function(
        inputs=[],
        outputs=[fe_t, kl_st, p_ot, p_oht, p_oat],
        updates=updates,
        on_unused_input='ignore',
        allow_input_downcast=True
    )

    # Run Optimization
    fe_min = None
    for i in range(config['n_steps']):

        # Perform stochastic gradient descent using ADAM updates
        start_time = timeit.default_timer()
        fe_t, kl_st, p_ot, p_oht, p_oat = train()
        end_time = timeit.default_timer()

        # Display useful information related to the training step
        print('Free Energies:\n', fe_t, kl_st, p_ot, p_oht, p_oat)
        print('Time for iteration: %f' % (end_time - start_time))

        # Save model
        fe_min = dai.save(config, fe_t, fe_min, i)

    # Save final parameters
    dai.save_model(config['base_name'] + '_final.pkl')


if __name__ == '__main__':
    # Train the agent.
    train_dai()
