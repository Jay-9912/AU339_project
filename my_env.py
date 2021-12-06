# import sys
# sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import logging
import random
import gym
import time
from Graph import *
import numpy as np
import cv2
from gym.envs.classic_control import rendering

logger = logging.getLogger(__name__)

# gym中显示的方格与序号关系，以3*4(l*w)为例
# 9 10 11
# 6  7  8
# 3  4  5
# 0  1  2


class School(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 1
    }

    def __init__(self, graph):
        self.graph = graph
        self.scale = 1
        self.states = range(self.graph.l*self.graph.w)  # 状态空间

        # self.x = [150,250,350,450] * 4
        # self.x = list(range(150, 500, 100)) * 4
        self.x = list(range(110, 500, 20)) * 20
        # print(self.x)
        # self.y=[150] * 4 + [250] * 4 + [350] * 4 + [450] * 4
        self.y = []
        for i in range(110, 500, 20):
            self.y = self.y + [i] * 20
        self.terminate_states = dict()  # 终止状态为字典格式
        # self.terminate_states[11] = 1
        # self.terminate_states[12] = 1
        # self.terminate_states[15] = 1

        self.actions = ['n', 's', 'w', 'e', 'stop']  # 上下左右以及不动五个动作

        self.rewards = dict()        # 回报的数据结构为字典
        # self.rewards['8_s'] = -1.0
        # self.rewards['13_w'] = -1.0
        # self.rewards['7_s'] = -1.0
        # self.rewards['10_e'] = -1.0
        # self.rewards['14_4'] = 1.0

        self.t = dict()             # 状态转移的数据格式为字典
        for state in range(self.graph.l*self.graph.w):

            key = "%d_stop" % (state)
            self.t[key] = state
            for action in self.actions[:4]:
                if action in self.graph.getVertex(state).getDirection():
                    key = "%d_%s" % (state, action)
                    self.t[key] = self.graph.getVertex(
                        state).getNeighbor(action)
        # print(self.t)
        self.gamma = 0.8  # 折扣因子
        self.viewer = None
        self.state = None

    def set_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)

    def getTerminal(self):
        return self.terminate_states

    def getGamma(self):
        return self.gamma

    def getStates(self):
        return self.states

    def getValidAction(self, n):
        return list(self.graph.getVertex(n).getDirection()) + ['stop']

    def getAction(self):
        return self.actions

    def getTerminate_states(self):
        return self.terminate_states

    def setState(self, s):
        self.state = s

    def random_action(self):
        valid_action = self.getValidAction(self.state)
        return random.sample(valid_action, 1)[0]

    # gym中显示的方格与obstacle_mask索引关系，以3*4(l*w)为例
    # (3,0) (3,1) (3,2)
    # (2,0) (2,1) (2,2)
    # (1,0) (1,1) (1,2)
    # (0,0) (0,1) (0,2)
    def generate_obstacle(self, num):
        self.obstacle_mask = np.zeros(
            (self.graph.w, self.graph.l))  # 注意array的行列顺序与gym中的顺序有所区分
        obstacle_list = random.sample(range(len(self.states)), num)
        for i in obstacle_list:
            row = i // self.graph.l
            col = i % self.graph.l
            self.obstacle_mask[row][col] = 1

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        self.obstacle_mask = cv2.dilate(self.obstacle_mask, kernel)
        self.obstacles = []

        for i in range(self.graph.w):
            for j in range(self.graph.l):
                if self.obstacle_mask[i][j] == 1:
                    obstacle = rendering.make_polygon(
                        [(0, 0), (20, 0), (20, 20), (0, 20)])
                    obstacletrans = rendering.Transform(
                        translation=(100+j*20, 100+i*20))
                    obstacle.add_attr(obstacletrans)
                    obstacle.set_color(0, 0, 0)
                    self.obstacles.append(obstacle)
                    self.viewer.add_geom(obstacle)
                    continue
                neighs = list(self.graph.getVertex(
                    i*self.graph.l+j).getConnections())
                for n in neighs:
                    row = n // self.graph.l
                    col = n % self.graph.l
                    if self.obstacle_mask[row][col] == 1:
                        self.graph.getVertex(i*self.graph.l+j).delNeighbor(n)

    def step(self, action):  # 更新state
        # 系统当前状态
        state = self.state
        # if state in self.terminate_states:
        #     return state, 0, True, {}
        print(self.graph.getVertex(state).getDirection())
        key = "%d_%s" % (state, action)

        # 状态转移
        assert (key in self.t)
        next_state = self.t[key]

        self.state = next_state

        is_terminal = False

        if next_state in self.terminate_states:
            is_terminal = True

        if key not in self.rewards:
            r = 0.0
        else:
            r = self.rewards[key]

        return next_state, r, is_terminal, {}

    def reset(self):
        # self.state = self.states[int(random.random() * len(self.states))]
        self.state = 0
        return self.state

    def render(self, mode='human'):
        screen_width = 600 * self.scale
        screen_height = 600 * self.scale

        if self.viewer is None:
            self.viewer = rendering.Viewer(screen_width, screen_height)

            # 创建网格世界
            self.lines = []
            for i in range(21):
                line = rendering.Line((100, 20*i+100), (500, 20*i+100))
                line.set_color(0, 0, 0)
                self.lines.append(line)
            for i in range(21):
                line = rendering.Line((20*i+100, 100), (20*i+100, 500))
                line.set_color(0, 0, 0)
                self.lines.append(line)

            # 创建机器人
            self.robot = rendering.make_circle(6)
            self.robotrans = rendering.Transform()
            self.robot.add_attr(self.robotrans)
            self.robot.set_color(0, 1, 0)

            for i in self.lines:
                self.viewer.add_geom(i)

            self.viewer.add_geom(self.robot)
            self.generate_obstacle(8)

        if self.state is None:
            return None

        self.robotrans.set_translation(self.x[self.state], self.y[self.state])

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()


if __name__ == '__main__':
    g = Rect_Graph(20, 20)
    env = School(g)
    env.reset()
    env.set_seed(4)
    for i in range(1000):
        env.render()
        action = env.random_action()
        env.step(action)
        time.sleep(1)
