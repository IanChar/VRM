from gym.envs.registration import register

from additional_gym_envs.msd_env import MassSpringDamperEnv
from additional_gym_envs.double_msd_env import DoubleMassSpringDamperEnv


###########################################################################
# Register all of the mass spring environments.
###########################################################################
obs_dict = {
    'xt': 'xt',
    'xtf': 'xtf',
    'xdtf': 'xdtf',
    'xpdtf': 'xpdtf',
    'xpitf': 'xpitf',
    'xidtf': 'xidtf',
    'pid': 'pid',
    'xpidt': 'xpidt',
    'xpidtf': 'xpidtf',
    'oracle': 'xpidtvckm',
}
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
}
for obk, obv in obs_dict.items():
    for vak, vav in vary_dict.items():
        kwargs = dict(vav)
        kwargs['observations'] = obv
        kwargs['action_is_change'] = 'f' in obv
        register(
            id=f'msd-{vak}-{obk}-v0',
            entry_point=MassSpringDamperEnv,
            kwargs=kwargs,
        )
        register(
            id=f'dmsd-{vak}-{obk}-v0',
            entry_point=DoubleMassSpringDamperEnv,
            kwargs=kwargs,
        )
