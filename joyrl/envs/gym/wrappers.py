
import turtle
import gymnasium as gym_wrapper
import torch
import numpy as np
from torchvision import transforms
from gymnasium.spaces import Box
from gymnasium.wrappers import FrameStack
from typing import List


class BaseSkipFrame(gym_wrapper.Wrapper):
    def __init__(
            self, 
            env, 
            skip: int, 
            cut_slices: List[List[int]]=None,
            start_skip: int=None,
            int_action_flag: bool=False
        ):
        """_summary_
        Args:
            env (_type_): _description_
            skip (int): skip frames
            cut_slices (List[List[int]], optional): pic observation cut. Defaults to None.
            start_skip (int, optional): skip several frames to start. Defaults to None.
            int_action_flag (bool): if the action only a int,  set true. Defaults to False.
        """
        super().__init__(env)
        self._skip = skip
        self.pic_cut_slices = cut_slices
        self.start_skip = start_skip
        self.int_action_flag = int_action_flag

    def _cut_slice(self, obs):
        slice_list = []
        for idx, dim_i_slice in enumerate(self.pic_cut_slices):
            slice_list.append(eval('np.s_[{st}:{ed}]'.format(st=dim_i_slice[0], ed=dim_i_slice[1])))

        obs = obs[tuple(i for i in slice_list)]
        return obs

    def step(self, action):
        tt_reward_list = []
        done = False
        total_reward = 0
        if self.int_action_flag:
            action = action[0]
        for i in range(self._skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            done_f = terminated or truncated
            total_reward += reward
            tt_reward_list.append(reward)
            if done:
                break

        obs = self._cut_slice(obs)  if self.pic_cut_slices is not None else obs
        return obs, total_reward, done_f, truncated, info

    def _start_skip(self):
        a = np.array([0.0, 0.0, 0.0]) if hasattr(self.env.action_space, 'low') else np.array(0) 
        for i in range(self.start_skip):
            obs, reward, terminated, truncated, info = self.env.step(a)
        return obs, info

    def reset(self, seed=0, options=None):
        s, info = self.env.reset(seed=seed, options=options)
        if self.start_skip is not None:
            obs, info = self._start_skip()
        obs = self._cut_slice(obs)  if self.pic_cut_slices is not None else obs
        return obs, info


class CartPoleActionWrapper(gym_wrapper.Wrapper):
    def __init__(self, env):
        gym_wrapper.Wrapper.__init__(self, env)
        # self.action_space = Box(low=-1.0, high=1.0, shape=(1, ), dtype=np.float32)

    def step(self, action):
        action = action[0]
        return self.env.step(action)

    def reset(self, seed=0, options=None):
        return self.env.reset(seed=seed, options=options)
    
class CliffWalkingWapper(gym_wrapper.Wrapper):
    def __init__(self, env):
        gym_wrapper.Wrapper.__init__(self, env)
        self.t = None
        self.unit = 50
        self.max_x = 12
        self.max_y = 4

    def draw_x_line(self, y, x0, x1, color='gray'):
        assert x1 > x0
        self.t.color(color)
        self.t.setheading(0)
        self.t.up()
        self.t.goto(x0, y)
        self.t.down()
        self.t.forward(x1 - x0)

    def draw_y_line(self, x, y0, y1, color='gray'):
        assert y1 > y0
        self.t.color(color)
        self.t.setheading(90)
        self.t.up()
        self.t.goto(x, y0)
        self.t.down()
        self.t.forward(y1 - y0)

    def draw_box(self, x, y, fillcolor='', line_color='gray'):
        self.t.up()
        self.t.goto(x * self.unit, y * self.unit)
        self.t.color(line_color)
        self.t.fillcolor(fillcolor)
        self.t.setheading(90)
        self.t.down()
        self.t.begin_fill()
        for i in range(4):
            self.t.forward(self.unit)
            self.t.right(90)
        self.t.end_fill()

    def move_player(self, x, y):
        self.t.up()
        self.t.setheading(90)
        self.t.fillcolor('red')
        self.t.goto((x + 0.5) * self.unit, (y + 0.5) * self.unit)

    def render(self):
        if self.t == None:
            self.t = turtle.Turtle()
            self.wn = turtle.Screen()
            self.wn.setup(self.unit * self.max_x + 100,
                          self.unit * self.max_y + 100)
            self.wn.setworldcoordinates(0, 0, self.unit * self.max_x,
                                        self.unit * self.max_y)
            self.t.shape('circle')
            self.t.width(2)
            self.t.speed(0)
            self.t.color('gray')
            for _ in range(2):
                self.t.forward(self.max_x * self.unit)
                self.t.left(90)
                self.t.forward(self.max_y * self.unit)
                self.t.left(90)
            for i in range(1, self.max_y):
                self.draw_x_line(
                    y=i * self.unit, x0=0, x1=self.max_x * self.unit)
            for i in range(1, self.max_x):
                self.draw_y_line(
                    x=i * self.unit, y0=0, y1=self.max_y * self.unit)

            for i in range(1, self.max_x - 1):
                self.draw_box(i, 0, 'black')
            self.draw_box(self.max_x - 1, 0, 'yellow')
            self.t.shape('turtle')

        x_pos = self.s % self.max_x
        y_pos = self.max_y - 1 - int(self.s / self.max_x)
        self.move_player(x_pos, y_pos)



class CarV2SkipFrame(gym_wrapper.Wrapper):
    def __init__(self, env, skip: int = 5, continue_flag: bool=False):
        """skip frame
        Args:
            env (_type_): _description_
            skip (int): skip frames
            continue_flag (bool): continuous flag in carRacing env 
        """
        super().__init__(env)
        self._skip = skip
        self.continue_flag = continue_flag
    
    def step(self, action):
        tt_reward_list = []
        done = False
        total_reward = 0
        for i in range(self._skip):
            obs, reward, done, info, _ = self.env.step(action)
            out_done = self.judge_out_of_route(obs)
            # done_f = done
            done_f = done or out_done
            reward = -10 if out_done else reward
            total_reward += reward
            tt_reward_list.append(reward)
            if done_f:
                break
        return obs[:84, 6:90, :], total_reward, done_f, info, _
    
    def judge_out_of_route(self, obs):
        s = obs[:84, 6:90, :]
        out_sum = (s[75, 35:48, 1][:2] > 200).sum() + (s[75, 35:48, 1][-2:] > 200).sum()
        return out_sum == 4

    def reset(self, seed=0, options=None):
        s, info = self.env.reset(seed=seed, options=options)
        # continue
        a = 0
        if self.continue_flag:
            a = np.array([0.0, 0.0, 0.0]) 
        for i in range(45):
            obs, reward, done, info, _ = self.env.step(a)

        return obs[:84, 6:90, :], info


class GrayScaleObservation(gym_wrapper.ObservationWrapper):
    def __init__(self, env):
        """RGP -> Gray
        (high, width, channel) -> (1, high, width) 
        """
        super().__init__(env)
        self.observation_space = Box(
            low=0, high=255, shape=self.observation_space.shape[:2], dtype=np.uint8
        )
    
    def observation(self, observation):
        tf = transforms.Grayscale()
        # channel first
        return tf(torch.tensor(np.transpose(observation, (2, 0, 1)).copy(), dtype=torch.float))


class ResizeObservation(gym_wrapper.ObservationWrapper):
    def __init__(self, env, shape: int = 84):
        """reshape observe
        Args:
            env (_type_): _description_
            shape (int): reshape size
        """
        super().__init__(env)
        self.shape = (shape, shape)
        obs_shape = self.shape + self.observation_space.shape[2:]
        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def observation(self, observation):
        #  Normalize -> input[channel] - mean[channel]) / std[channel]
        tf = transforms.Compose([transforms.Resize(self.shape), transforms.Normalize(0, 255)])
        return tf(observation).squeeze(0)

# 跳帧
class SkipFrame(gym_wrapper.Wrapper):
    def __init__(self, env, skip: int = 5, skip_start: int = 0):
        """skip frame
        Args:
            env (_type_): _description_
            skip (int): skip frames
            skip_start (int): skip start frames
        """
        super().__init__(env)
        self._skip = skip
        self._skip_start = skip_start
    
    def step(self, action):
        tt_reward_list = []
        done = False
        total_reward = 0
        for i in range(self._skip):
            if type(action) != int:
                action = action[0]
            obs, reward, done, truncated, info = self.env.step(action)
            total_reward += reward
            tt_reward_list.append(reward)
            if done:
                break
        return obs, total_reward, done, truncated, info

    def reset(self, seed=0, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        action = 0
        for i in range(self._skip_start):
            obs, reward, done, truncated, info = self.env.step(action)

        return obs, info

# 裁剪
class CropFrame(gym_wrapper.ObservationWrapper):
    def __init__(self, env, x1=0, x2=0, y1=0, y2=0):
        """crop frame
        Args:
            env (_type_): _description_
            x1 (int): crop left
            x2 (int): crop right
            y1 (int): crop top
            y2 (int): crop bottom
        """
        super().__init__(env)
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2
    
    def observation(self, obs):
        # save_img(obs)
        if self.x1 or self.x2:
            obs = obs[:, self.x1:self.x2, :]
        if self.y1 or self.y2:
            obs = obs[self.y1:self.y2, :, :]
        return obs

class InfoRewardFrame(gym_wrapper.Wrapper):
    def __init__(self, env, goods=[], bads=[]):
        """skip frame
        Args:
            env (_type_): _description_
            skip (int): skip frames
            skip_start (int): skip start frames
        """
        super().__init__(env)
        self._goods = goods
        self._bads = bads
        self.last_info = None
    
    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        if not self.last_info:
            self.last_info = info
        for good in self._goods:
            rate = self._goods[good]
            reward += (info[good] - self.last_info[good]) * rate
        for bad in self._bads:
            rate = self._bads[bad]
            reward += (info[bad] - self.last_info[bad]) * -rate
        self.last_info = info
        # print(info, reward)
        return obs, reward, done, truncated, info

    def reset(self, seed=0, options=None):
        self.last_info = None
        obs, info = self.env.reset(seed=seed, options=options)
        return obs, info