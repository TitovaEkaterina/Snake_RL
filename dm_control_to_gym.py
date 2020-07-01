import glfw
try:
    glfw.init()
except:
    pass

from dm_control import suite
from dm_control.suite import humanoid_CMU
import gym.spaces as spaces
from gym.envs.registration import EnvSpec
import matplotlib.pyplot as plt
import numpy as np
import cv2


class Env_DM_Control(object):
    def __init__(self, name, img_size=84, camera_id='side', max_step=-1):
        self.env_name = name
        self.img_size = img_size
        self.camera_id = camera_id
        self.max_step = max_step
        if self.env_name == 'Humanoid_CMU':
            self.env = humanoid_CMU.run()
        else:
            domain, task = self.env_name.split('/')
            self.env = suite.load(domain_name=domain, task_name=task)
        self.control_min = self.env.action_spec().minimum[0]
        self.control_max = self.env.action_spec().maximum[0]
        self.control_shape = self.env.action_spec().shape
        self._action_space = spaces.Box(self.control_min, self.control_max, self.control_shape)
        total_size = 0
        for i, j in self.env.observation_spec().items():
            total_size += j.shape[0] if len(j.shape) > 0 else 1
        self._observation_space = spaces.Box(-np.inf, np.inf, (total_size, ))
        self.step_count = 0
        self.reward_range = (-np.inf, np.inf)
        self.metadata = {'render.modes': ['human', 'rgb_array'], 'video.frames_per_second': 67}
        self.spec = EnvSpec('Humanoid-v2', max_episode_steps=1000)

    @property
    def action_space(self):
        return self._action_space

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def physics(self):
        return self.env.physics

    def reset(self):
        obs = self.env.reset().observation
        l = []
        for i, j in obs.items():
            l.append(j if len(j.shape) > 0 else j.reshape(1))
        return np.concatenate(l)

    def step(self, action):
        ret = self.env.step(action)
        l = []
        for i, j in ret.observation.items():
            l.append(j if len(j.shape) > 0 else j.reshape(1))
        state = np.concatenate(l)
        reward = ret.reward
        done = (ret.step_type == 2) or (self.step_count == self.max_step)
        info = {}
        self.step_count += 1
        if done:
            self.step_count = 0
        return state, reward, done, info

    def render(self):
        height = width = self.img_size
        camera_id = self.camera_id
        if camera_id:
            img = self.env.physics.render(height, width, camera_id=camera_id)
        else:
            img = self.env.physics.render(height, width)
        return img

    def seed(self, seed):
        if self.env_name == 'Humanoid_CMU':
            self.env = humanoid_CMU.run(random=seed)
        else:
            domain, task = self.env_name.split('+')
            self.env = suite.load(domain_name=domain, task_name=task, task_kwargs={'random':seed})

    def close(self):
        pass

if __name__ == '__main__':
    max_frame = 1001

    width = 480
    height = 480
    video = np.zeros((10001, height, 2 * width, 3), dtype=np.uint8)

    env = Env_DM_Control('swimmer/swimmer6')
    env.reset()
    print(env.action_space)
    print(env.observation_space)
    while True:
        video = np.hstack([env.physics.render(height, width, camera_id=0),
                                            env.physics.render(height, width, camera_id=1)])
        img = plt.imshow(video)
        plt.pause(0.01)  # Need min display time > 0.0.
        plt.draw()
