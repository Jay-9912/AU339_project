# import sys
# sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import enum
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


class Agent():
    def __init__(self, init_state, id):
        self.state = init_state
        self.id = id
        self.guard_nearby = []

    def render(self):
        self.viz = rendering.make_circle(6)
        self.trans = rendering.Transform()
        self.viz.add_attr(self.trans)
        self.viz.set_color(0, 1, 0)

    def set_location(self, x, y):
        self.trans.set_translation(x, y)

    def get_state(self):
        return self.state

    def update_state(self, next_state):
        self.state = next_state

    def get_action(self, env):
        # 需要根据当前位置、附近保安位置和策略、小偷位置综合决策
        return self.random_action(env)

    def getValidAction(self, n, env):
        return list(env.graph.getVertex(n).getDirection()) + ['stop']

    def random_action(self, env):
        valid_action = self.getValidAction(self.state, env)
        return random.sample(valid_action, 1)[0]

    def clear_guard_nearby(self):
        self.guard_nearby = []

    def add_guard_nearby(self, id):
        self.guard_nearby.append(id)


class Thief():
    def __init__(self, init_state, id):
        self.state = init_state
        self.id = id
        self.escape = 0

    def render(self):
        self.viz = rendering.make_circle(6)
        self.trans = rendering.Transform()
        self.viz.add_attr(self.trans)
        self.viz.set_color(1, 0, 0)

    def set_location(self, x, y):
        self.trans.set_translation(x, y)

    def get_state(self):
        return self.state

    def update_state(self, next_state):
        self.state = next_state

    def __del__(self):
        if self.escape == 0:
            print("Thief No.%d is caught by guard"%(self.id))
        else:
            print("Thief No.%d escapes"%(self.id))

class School(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 1
    }

    def __init__(self, graph, agent_num, perception_range, thief_num, theft_time, obstacle_num):
        self.set_seed(4)
        self.graph = graph  # rect_graph
        self.grid_size = 20
        self.states = range(self.graph.l * self.graph.w)  # 状态空间
        self.agent_num = agent_num
        self.thief_num = thief_num
        self.obstacle_num = obstacle_num
        self.perception_range =perception_range
        self.theft_time = theft_time
        self.agent_list = []
        self.thief_list = {}
        self.screen_width = 1200
        self.screen_height = 800
        self.loss = 0
        self.x_min = int(self.screen_width / 2 -
                         self.grid_size / 2 * self.graph.l)
        self.x_max = int(self.screen_width / 2 +
                         self.grid_size / 2 * self.graph.l)
        self.y_min = int(self.screen_height / 2 -
                         self.grid_size / 2 * self.graph.w)
        self.y_max = int(self.screen_height / 2 +
                         self.grid_size / 2 * self.graph.w)
        self.x = list(
            range(self.x_min + int(self.grid_size / 2), self.x_max,
                  self.grid_size)) * self.graph.w  # 每一小格20*20像素
        self.y = []
        for i in range(self.y_min + int(self.grid_size / 2), self.y_max,
                       self.grid_size):
            self.y = self.y + [i] * self.graph.l

        self.actions = ['n', 's', 'w', 'e', 'stop']  # 上下左右以及不动五个动作

        self.t = dict()  # 状态转移的数据格式为字典
        self.viewer = None
        self.obstacle_mask = None
        self.init_env()
        self.init_state = None
        # self.init_state = self.GetInitState() # 需要初始化每个保安的初始位置  df part
        if self.init_state is None:  # 随机初始化位置，仅适用于4个保安的情况，否则报错
            self.init_state = [
                0, self.graph.l - 1,
                self.graph.l * self.graph.w - self.graph.l,
                self.graph.l * self.graph.w - 1
            ]
        self.state = self.init_state
        self.init_agent()

    def init_env(self):
        # 建立状态转移关系
        for state in range(self.graph.l * self.graph.w):
            key = "%d_stop" % (state)
            self.t[key] = state
            for action in self.actions[:4]:
                if action in self.graph.getVertex(state).getDirection():
                    key = "%d_%s" % (state, action)
                    self.t[key] = self.graph.getVertex(state).getNeighbor(
                        action)
        self.generate_obstacle(self.obstacle_num)

    def init_agent(self):
        # 初始化agent
        for num in range(self.agent_num):
            agent = Agent(self.state[num], num)
            self.agent_list.append(agent)

    def generate_thief(self):
        invalid_loc = self.obstacle_mask.copy()
        invalid_loc = invalid_loc.flatten()
        for agent in self.agent_list:
            invalid_loc[agent.get_state()] = 1
        valid_loc = np.where(invalid_loc == 0)[0]
        loc_list = np.random.choice(valid_loc,
                                    size=self.thief_num,
                                    replace=False)
        for id, loc in enumerate(loc_list):
            thief = Thief(loc_list[id], id)
            thief.render()
            self.viewer.add_geom(thief.viz)
            thief.set_location(self.x[loc], self.y[loc])
            self.thief_list[id] = thief
            
    def cal_loss(self):
        keys = list(self.thief_list.keys())
        self.loss += len(keys)
        for k in keys:
            self.thief_list[k].escape = 1 # 逃跑成功
            self.viewer.geoms.remove(self.thief_list[k].viz) # 删除图形显示
            self.thief_list.pop(k) # 删除键

    def set_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)

    def getStates(self):
        return self.states

    def getAction(self):
        pass

    def setState(self, s):
        self.state = s

    def getRandomAction(self):
        action = [agent.random_action(self) for agent in self.agent_list]
        return action

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
        # 删除和障碍物相连的边
        for i in range(self.graph.w):
            for j in range(self.graph.l):
                if self.obstacle_mask[i][j] == 1:
                    neighs = list(
                        self.graph.getVertex(i * self.graph.l +
                                             j).getConnections())
                    for n in neighs:
                        row = n // self.graph.l
                        col = n % self.graph.l
                        self.graph.getVertex(i * self.graph.l +
                                             j).delNeighbor(n)
                    continue
                neighs = list(
                    self.graph.getVertex(i * self.graph.l +
                                         j).getConnections())
                for n in neighs:
                    row = n // self.graph.l
                    col = n % self.graph.l
                    if self.obstacle_mask[row][col] == 1:
                        self.graph.getVertex(i * self.graph.l +
                                             j).delNeighbor(n)

    def cal_state_dist(self, s1, s2):
        row1 = s1 // self.graph.l
        col1 = s1 % self.graph.l
        row2 = s2 // self.graph.l
        col2 = s2 % self.graph.l
        return abs(row1 - row2) + abs(col1 - col2)

    def step(self, action):  # 更新state
        # 系统当前状态
        state = self.state

        next_state_list = []
        for id in range(self.agent_num):
            key = "%d_%s" % (state[id], action[id])

            # 状态转移
            assert (key in self.t)
            next_state = self.t[key]
            self.agent_list[id].update_state(next_state)
            next_state_list.append(next_state)

        self.state = next_state_list

        # 感知附近的其他保安
        for agent in self.agent_list:
            agent.clear_guard_nearby()
        for i in range(self.agent_num):
            for j in range(i + 1, self.agent_num):
                if self.cal_state_dist(self.agent_list[i].get_state(),
                                       self.agent_list[j].get_state()) < self.perception_range:
                    self.agent_list[i].add_guard_nearby(j)
                    self.agent_list[j].add_guard_nearby(i)
        # for agent in self.agent_list:
        #     print(agent.id, agent.guard_nearby)

        # 是否抓到小偷
        for agent in self.agent_list:
            keys = list(self.thief_list.keys())
            for k in keys:
                if self.cal_state_dist(agent.get_state(), self.thief_list[k].get_state()) <= 1:
                    self.viewer.geoms.remove(self.thief_list[k].viz) # 删除图形显示
                    self.thief_list.pop(k) # 删除键

        return next_state_list

    def reset(self):

        self.state = self.init_state
        return self.state

    def render(self, mode='human'):

        if self.viewer is None:
            self.viewer = rendering.Viewer(self.screen_width,
                                           self.screen_height)
            # 创建网格世界
            self.lines = []
            for i in range(self.graph.w + 1):
                line = rendering.Line(
                    (self.x_min, self.grid_size * i + self.y_min),
                    (self.x_max, self.grid_size * i + self.y_min))
                line.set_color(0, 0, 0)
                self.lines.append(line)
            for i in range(self.graph.l + 1):
                line = rendering.Line(
                    (self.grid_size * i + self.x_min, self.y_min),
                    (self.grid_size * i + self.x_min, self.y_max))
                line.set_color(0, 0, 0)
                self.lines.append(line)

            # 渲染保安
            for agent in self.agent_list:
                agent.render()
                self.viewer.add_geom(agent.viz)

            for i in self.lines:
                self.viewer.add_geom(i)
            self.obstacles = []

            for i in range(self.graph.w):
                for j in range(self.graph.l):
                    if self.obstacle_mask[i][j] == 1:
                        obstacle = rendering.make_polygon([
                            (0, 0), (self.grid_size, 0),
                            (self.grid_size, self.grid_size),
                            (0, self.grid_size)
                        ])
                        obstacletrans = rendering.Transform(
                            translation=(self.x_min + j * self.grid_size,
                                         self.y_min + i * self.grid_size))
                        obstacle.add_attr(obstacletrans)
                        obstacle.set_color(0, 0, 0)
                        self.obstacles.append(obstacle)
                        self.viewer.add_geom(obstacle)

        if self.state is None:
            return None
        for agent in self.agent_list:
            state = agent.get_state()
            # print('state:', state)
            agent.set_location(self.x[state], self.y[state])

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()


if __name__ == '__main__':
    g = Rect_Graph(40, 36)
    env = School(g, 4, 5, 6, 10, 8)
    env.reset()

    for i in range(1000):
        # 假设2s时出现小偷，10s偷完，这里需要根据需要设置合适的小偷产生模型
        if i == 2: 
            env.generate_thief()
        if i == 10: 
            env.cal_loss()
            print('loss:', env.loss)
        # if i ==3 :
        #     del env.thief_list[1]
        env.render()
        action = env.getRandomAction()  # list
        # print(env.state)
        env.step(action)
        time.sleep(1)
