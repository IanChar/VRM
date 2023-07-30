"""run_experiment.
Usage:
  run_experiment.py run [--env=<kn>] [--steps=<kn>] [--seed=<kn>] [--render]
  run_experiment.py (-h | --help)
Options:
  -h --help     Show this screen.
  --env=<kn>  Environment (see readme.txt) [default: PendulumV].
  --steps=<kn>  How many steps to run [default: 50000].
  --seed=<kn>  Random seed [default: 0].
"""
from docopt import docopt
import numpy as np
import torch
from vrdm import VRM, VRDM

import os
import warnings
import scipy.io as sio
import pickle as pkl


arguments = docopt(__doc__, version='1.0')


def test_performance(agent_test, env_test, action_filter, times=10):

    EpiTestRet = 0
    for _ in range(times):

        # reset each episode
        s0 = env_test.reset().astype(np.float32)
        r0 = np.array([0.], dtype=np.float32)
        x0 = np.concatenate([s0, r0])
        a = agent_test.init_episode(x0).reshape(-1)

        for t in range(max_steps):
            if np.any(np.isnan(a)):
                raise ValueError
            sp, r, done, _ = env_test.step(action_filter(a))
            a = agent_test.select(sp, r, action_return='normal')
            EpiTestRet += r
            if done:
                break

    return EpiTestRet / times


def custom_test(agent_test, env_test, action_filter, settings):
    test_ret = 0
    for task, target, start in zip(settings['task'], settings['targets'],
                                   settings['start']):
        # reset each episode
        s0 = env_test.reset(task=task, targets=target, start=start).astype(np.float32)
        r0 = np.array([0.], dtype=np.float32)
        x0 = np.concatenate([s0, r0])
        a = agent_test.init_episode(x0).reshape(-1)

        for t in range(max_steps):
            if np.any(np.isnan(a)):
                raise ValueError
            sp, r, done, _ = env_test.step(action_filter(a))
            a = agent_test.select(sp, r, action_return='normal')
            test_ret += r
            if done:
                break

    return test_ret / len(settings['task'])




seed = int(arguments["--seed"])  # random seed
np.random.seed(seed)
torch.manual_seed(seed)


beta_h = 'auto_1.0'
optimizer_st = 'adam'
minibatch_size = 4
seq_len = 64
reward_scale = 1.0
lr_vrm = 8e-4
gamma = 0.95
max_all_steps = int(arguments["--steps"])  # total steps to learn
step_perf_eval = 5000  # how many steps to do evaluation

env_name = arguments["--env"]


if arguments["--render"]:
    rendering = True
else:
    rendering = False

vary_dict = {
    'fixed': {
        'damping_constant_bounds': (4.0, 4.0),
        'spring_stiffness_bounds': (2.0, 2.0),
        'mass_bounds': (20.0, 20.0),
    },
    'small': {
        'damping_constant_bounds': (3.5, 5.5),
        'spring_stiffness_bounds': (1.75, 3.0),
        'mass_bounds': (17.5, 40.0),
    },
    'med': {
        'damping_constant_bounds': (3.0, 7.0),
        'spring_stiffness_bounds': (1.25, 4.0),
        'mass_bounds': (15.0, 60.0),
    },
    'large': {
        'damping_constant_bounds': (2.0, 10.0),
        'spring_stiffness_bounds': (0.5, 6.0),
        'mass_bounds': (10.0, 100.0),
    },
    'sim': {
        'mass_bounds': (15.0, 25.0),
        'friction_bounds': (0.0, 0.0),
        'skfdiff_bounds': (0.0, 0.0),
    },
    'real': {
        'mass_bounds': (5.0, 35.0),
        'friction_bounds': (0.05, 0.25),
        'skfdiff_bounds': (0.25, 0.75),
    },
}
eval_dict = {
    'fixed': ['fixed', 'small', 'large'],
    'small': ['fixed', 'small', 'large'],
    'large': ['fixed', 'small', 'large'],
    'sim': ['sim', 'real'],
    'real': ['sim', 'real'],
}

savepath = f'./data/{env_name}/{seed}'
os.makedirs(savepath)

if env_name == "Sequential":

    from task import TaskT
    env = TaskT(3)
    env_test = TaskT(3)
    action_filter = lambda a: a.reshape([-1])

    max_steps = 128
    est_min_steps = 10

elif env_name == "CartPole":

    from task import ContinuousCartPoleEnv
    env = ContinuousCartPoleEnv()
    env_test = ContinuousCartPoleEnv()

    action_filter = lambda a: a.reshape([-1])

    max_steps = 1000
    est_min_steps = 5

elif env_name == "CartPoleP":

    from task import CartPoleP
    env = CartPoleP()
    env_test = CartPoleP()

    action_filter = lambda a: a.reshape([-1])

    max_steps = 1000
    est_min_steps = 10

elif env_name == "CartPoleV":

    from task import CartPoleV
    env = CartPoleV()
    env_test = CartPoleV()

    action_filter = lambda a: a.reshape([-1])

    max_steps = 1000
    est_min_steps = 10

elif env_name == "Pendulum":

    import gym
    env = gym.make("Pendulum-v0")
    env_test = gym.make("Pendulum-v0")

    action_filter = lambda a: a.reshape([-1]) * 2 # because range of pendulum's action is [-2, 2]. For other environments, * 2 is not needed

    max_steps = 200
    est_min_steps = 199

elif env_name == "PendulumP":

    from task import PendulumP
    env = PendulumP()
    env_test = PendulumP()

    action_filter = lambda a: a.reshape([-1]) * 2

    max_steps = 200
    est_min_steps = 199

elif env_name == "PendulumV":

    from task import PendulumV
    env = PendulumV()
    env_test = PendulumV()

    action_filter = lambda a: a.reshape([-1]) * 2

    max_steps = 200
    est_min_steps = 199

elif env_name == "Hopper":

    import gym
    import roboschool
    env = gym.make("RoboschoolHopper-v1")
    env_test = gym.make("RoboschoolHopper-v1")

    action_filter = lambda a: a.reshape([-1])

    max_steps = 1000
    est_min_steps = 5

elif env_name == "HopperP":

    from task import RsHopperP
    env = RsHopperP()
    env_test = RsHopperP()

    action_filter = lambda a: a.reshape([-1])

    max_steps = 1000
    est_min_steps = 5

elif env_name == "HopperV":

    from task import RsHopperV
    env = RsHopperV()
    env_test = RsHopperV()

    action_filter = lambda a: a.reshape([-1])

    max_steps = 1000
    est_min_steps = 5

elif env_name == "Walker2d":

    import gym
    import roboschool
    env = gym.make("RoboschoolWalker2d-v1")
    env_test = gym.make("RoboschoolWalker2d-v1")

    action_filter = lambda a: a.reshape([-1])

    max_steps = 1000
    est_min_steps = 5

elif env_name == "Walker2dV":

    from task import RsWalker2dV
    env = RsWalker2dV()
    env_test = RsWalker2dV()

    action_filter = lambda a: a.reshape([-1])

    max_steps = 1000
    est_min_steps = 5

elif env_name == "Walker2dP":

    from task import RsWalker2dP
    env = RsWalker2dP()
    env_test = RsWalker2dP()

    action_filter = lambda a: a.reshape([-1])

    max_steps = 1000
    est_min_steps = 5

elif env_name == "Ant":

    import gym
    import roboschool
    env = gym.make("RoboschoolAnt-v1")
    env_test = gym.make("RoboschoolAnt-v1")

    action_filter = lambda a: a.reshape([-1])

    max_steps = 1000
    est_min_steps = 20

elif env_name == "AntV":

    from task import RsAntV
    env = RsAntV()
    env_test = RsAntV()

    action_filter = lambda a: a.reshape([-1])

    max_steps = 1000
    est_min_steps = 20

elif env_name == "AntP":

    from task import RsAntP
    env = RsAntP()
    env_test = RsAntP()

    action_filter = lambda a: a.reshape([-1])

    max_steps = 1000
    est_min_steps = 20

else:
    assert '-' in env_name
    env_task, env_type = env_name.split('-')
    if env_task.lower() == 'msd':
        from additional_gym_envs.msd_env import MassSpringDamperEnv
        env_class = MassSpringDamperEnv
    elif env_task.lower() == 'dmsd':
        from additional_gym_envs.double_msd_env import DoubleMassSpringDamperEnv
        env_class = DoubleMassSpringDamperEnv
    elif env_task.lower() == 'nav':
        from additional_gym_envs.nav import Navigation
        env_class = Navigation
    env = env_class(
        observations='xtf',
        action_is_change=True,
        **vary_dict[env_type],
    )
    env_test = {k: env_class(
        observations='xtf',
        action_is_change=True,
        **vary_dict[k],
    ) for k in eval_dict[env_type]}
    settings = {}
    for test_type in eval_dict[env_type]:
        with open(f'env_settings/{env_task.lower()}-{test_type}-te.pkl', 'rb') as f:
            settings[test_type] = pkl.load(f)
    action_filter = lambda a: a.reshape([-1])
    max_steps = 100
    est_min_steps = 10
    with open(os.path.join(savepath, 'log.csv'), 'w') as log:
        log.write(','.join(['Steps'] + eval_dict[env_type] + ['\n']))


rnn_type = 'mtlstm'
d_layers = [256, ]
z_layers = [64, ]
x_phi_layers = [128]
decode_layers = [128, 128]

value_layers = [256, 256]
policy_layers = [24]

step_start_rl = 1000
step_start_st = 1000
step_end_st = np.inf
fim_train_times = 5000

train_step_rl = 1  # how many times of RL training after step_start_rl
train_step_st = 5

train_freq_rl = 1. / train_step_rl
train_freq_st = 1. / train_step_st

max_episodes = int(max_all_steps / est_min_steps) + 1  # for replay buffer


fim = VRM(input_size=env.observation_space.shape[0] + 1,
          action_size=env.action_space.shape[0],
          rnn_type=rnn_type,
          d_layers=d_layers,
          z_layers=z_layers,
          decode_layers=decode_layers,
          x_phi_layers=x_phi_layers,
          optimizer=optimizer_st,
          lr_st=lr_vrm)

klm = VRM(input_size=env.observation_space.shape[0] + 1,
          action_size=env.action_space.shape[0],
          rnn_type=rnn_type,
          d_layers=d_layers,
          z_layers=z_layers,
          decode_layers=decode_layers,
          x_phi_layers=x_phi_layers,
          optimizer=optimizer_st,
          lr_st=lr_vrm)

agent = VRDM(fim, klm, gamma=gamma,
             beta_h=beta_h,
             value_layers=value_layers,
             policy_layers=policy_layers)

agent_test = VRDM(fim, klm, gamma=gamma,
                  beta_h=beta_h,
                  value_layers=value_layers,
                  policy_layers=policy_layers)


SP_real = np.zeros([max_episodes, max_steps, env.observation_space.shape[0]], dtype=np.float32)  # observation (t+1)
A_real = np.zeros([max_episodes, max_steps, env.action_space.shape[0]], dtype=np.float32)  # action
R_real = np.zeros([max_episodes, max_steps], dtype=np.float32)  # reward
D_real = np.zeros([max_episodes, max_steps], dtype=np.float32)  # done
V_real = np.zeros([max_episodes, max_steps], dtype=np.float32)  # mask, indicating whether a step is valid. value: 1 (compute gradient at this step) or 0 (stop gradient at this step)

performance_wrt_step = []
global_steps = []

e_real = 0
global_step = 0
loss_sts = []

#  Run
while global_step < max_all_steps:

    s0 = env.reset().astype(np.float32)
    r0 = np.array([0.], dtype=np.float32)
    x0 = np.concatenate([s0, r0])
    a = agent.init_episode(x0).reshape(-1)

    for t in range(max_steps):

        if global_step == max_all_steps:
            break

        if np.any(np.isnan(a)):
            raise ValueError

        sp, r, done, _ = env.step(action_filter(a))
        if rendering:
            env.render()

        A_real[e_real, t, :] = a
        SP_real[e_real, t, :] = sp.reshape([-1])
        R_real[e_real, t] = r
        D_real[e_real, t] = 1 if done else 0
        V_real[e_real, t] = 1

        a = agent.select(sp, r)

        global_step += 1

        if global_step == step_start_st + 1:
            print("Start training the first-impression model!")
            _, _, loss_st = agent.learn_st(True, False,
                                           SP_real[0:e_real], A_real[0:e_real], R_real[0:e_real],
                                           D_real[0:e_real], V_real[0:e_real],
                                           times=fim_train_times, minibatch_size=minibatch_size)
            loss_sts.append(loss_st)
            print("Finish training the first-impression model!")
            print("Start training the keep-learning model!")

        if global_step > step_start_st and global_step <= step_end_st and np.random.rand() < train_freq_st:
            _, _, loss_st = agent.learn_st(False, True,
                                           SP_real[0:e_real], A_real[0:e_real], R_real[0:e_real],
                                           D_real[0:e_real], V_real[0:e_real],
                                           times=max(1, int(train_freq_st)), minibatch_size=minibatch_size)
            loss_sts.append(loss_st)

        if global_step > step_start_rl and np.random.rand() < train_freq_rl:
            if global_step == step_start_rl + 1:
                print("Start training the RL controller!")
            agent.learn_rl_sac(SP_real[0:e_real], A_real[0:e_real], R_real[0:e_real],
                               D_real[0:e_real], V_real[0:e_real],
                               times=max(1, int(train_freq_rl)), minibatch_size=minibatch_size,
                               reward_scale=reward_scale, seq_len=seq_len)

        if global_step % step_perf_eval == 0:
            agent_test.load_state_dict(agent.state_dict())  # update agent_test
            if isinstance(env_test, dict):
                rets = []
                for k in eval_dict[env_type]:
                    ret = custom_test(agent_test, env_test[k], action_filter,
                                      settings[k])
                    rets.append(ret)
                    if k == env_type:
                        performance_wrt_step.append(ret)
                # Log all of the rets.
                with open(os.path.join(savepath, 'log.csv'), 'a') as log:
                    log.write(','.join([str(global_step)] + [str(r) for r in rets]
                                       + ['\n']))
            else:
                EpiTestRet = test_performance(agent_test, env_test, action_filter,
                                              times=5)
                performance_wrt_step.append(EpiTestRet)
            global_steps.append(global_step)
            warnings.warn(env_name + ": global step: {}, : steps {}, test return {}".format(
                global_step, t, EpiTestRet))

        if done:
            break

    print(env_name + " -- episode {} : steps {}, mean reward {}".format(e_real, t, np.mean(R_real[e_real])))
    e_real += 1

performance_wrt_step_array = np.reshape(performance_wrt_step, [-1]).astype(np.float64)
global_steps_array = np.reshape(global_steps, [-1]).astype(np.float64)
loss_sts = np.reshape(loss_sts, [-1]).astype(np.float64)

data = {"loss_sts": loss_sts,
        "max_episodes": max_episodes,
        "step_start_rl": step_start_rl,
        "step_start_st": step_start_st,
        "step_end_st": step_end_st,
        "rnn_type": rnn_type,
        "optimizer": optimizer_st,
        "reward_scale": reward_scale,
        "beta_h": beta_h,
        "minibatch_size": minibatch_size,
        "train_step_rl": train_step_rl,
        "train_step_st": train_step_st,
        "R": np.sum(R_real, axis=-1).astype(np.float64),
        "steps": np.sum(V_real, axis=-1).astype(np.float64),
        "performance_wrt_step": performance_wrt_step_array,
        "global_steps": global_steps_array}

sio.savemat(savepath + env_name + "_vrm.mat", data)
torch.save(agent, savepath + env_name + "_vrm.model")
