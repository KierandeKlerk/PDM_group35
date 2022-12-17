from gym.envs.registration import register

register(
    id='pointmass-aviary-v0',
    entry_point='quadrotor_project.envs:PointMassAviary',
)