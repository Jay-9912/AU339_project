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
import copy

logger = logging.getLogger(__name__)

GRAPH_LENGTH = 40
GRAPH_WIDTH = 36

# gym中显示的方格与序号关系，以3*4(l*w)为例
# 9 10 11
# 6  7  8
# 3  4  5
# 0  1  2


class Agent(): # 保安类
    def __init__(self, init_state, id, res):
        self.state = init_state # 初始state
        self.id = id 
        self.guard_nearby = [] # 附近的保安
        self.observe_range = 3 # 观察范围
        self.res = res # 分辨率

    def change(self): # 改变颜色
        self.viz.set_color(1, 0, 0)

    def render(self):
        self.viz = rendering.Image('guard.jpg', self.res - 1, self.res - 1) # 添加卡通形象
        self.trans = rendering.Transform() 
        self.viz.add_attr(self.trans)
        self.sight_list = {} 
        self.trans_list = {}
        for i in range(-self.observe_range, self.observe_range + 1): # 渲染观察范围
            for j in range(
                    abs(i) - self.observe_range,
                    self.observe_range - abs(i) + 1):
                if i == 0 and j == 0:
                    continue
                sight = rendering.make_polygon([(0, 0), (self.res, 0),
                                                (self.res, self.res),
                                                (0, self.res)])
                trans = rendering.Transform()
                sight.add_attr(trans)
                self.trans_list[(i, j)] = trans
                sight.set_color_rgbd(112 / 255.0, 10 / 255.0, 245 / 255.0, 0.2)
                self.sight_list[(i, j)] = sight

    def set_location(self, x, y): # 设置保安的位置
        self.trans.set_translation(x, y)
        for key in self.sight_list.keys(): # 相应地更新观察范围的位置
            row = self.state // GRAPH_LENGTH
            col = self.state % GRAPH_LENGTH
            row_sight = row + key[0]
            col_sight = col + key[1]
            self.trans_list[key].set_translation(
                x + key[1] * self.res - self.res / 2,
                y + key[0] * self.res - self.res / 2)
            if row_sight < 0 or row_sight >= GRAPH_WIDTH or col_sight < 0 or col_sight >= GRAPH_LENGTH: # 消除超出地图范围的观察范围
                self.sight_list[key].set_color_rgbd(112 / 255.0, 10 / 255.0,
                                                    245 / 255.0, 0)
            else:
                self.sight_list[key].set_color_rgbd(112 / 255.0, 10 / 255.0,
                                                    245 / 255.0, 0.2)

    def get_state(self): # 获得保安状态
        return self.state

    def update_state(self, next_state): # 更新状态
        self.state = next_state

    def get_action(self, env):
        # 需要根据当前位置、附近保安位置和策略、小偷位置综合决策
        return self.random_action(env)

    def getValidAction(self, n, env): # 获得合法action
        return list(env.graph.getVertex(n).getDirection()) + ['stop']

    def random_action(self, env): # 随即策略
        valid_action = self.getValidAction(self.state, env)
        return random.sample(valid_action, 1)[0]

    def clear_guard_nearby(self): # 清除附近的保安列表
        self.guard_nearby = []

    def add_guard_nearby(self, id): # 添加附近的保安
        self.guard_nearby.append(id)


class Thief(): # 小偷类
    def __init__(self, init_state, id, res):
        self.state = init_state # 初始状态
        self.id = id
        self.escape = 0 # 是否逃跑成功
        self.res = res # 分辨率

    def render(self):
        self.viz = rendering.Image('thief.jpg', self.res - 1, self.res - 1) # 渲染小偷
        self.trans = rendering.Transform()
        self.viz.add_attr(self.trans)

    def set_location(self, x, y): # 设置位置
        self.trans.set_translation(x, y)

    def get_state(self): # 获得状态
        return self.state

    def update_state(self, next_state): # 更新状态
        self.state = next_state

    def __del__(self): # 析构
        if self.escape == 0:
            print("Thief No.%d is caught by guard" % (self.id))
        else:
            print("Thief No.%d escapes" % (self.id))


class School(gym.Env): # 学校类
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 1
    }

    def __init__(self, graph, agent_num, perception_range, thief_num,
                 theft_time, obstacle_num):
        self.set_seed(4)
        self.graph = graph  # rect_graph
        self.grid_size = 20 # 网格大小
        self.states = range(self.graph.l * self.graph.w)  # 状态空间
        self.time_flow = np.zeros((self.graph.w, self.graph.l)) 
        self.detect_value = 1
        self.detected_graph = np.zeros((self.graph.w, self.graph.l))
        self.prob_distrib = np.zeros((self.graph.w, self.graph.l))

        self.agent_num = agent_num # 保安个数
        self.thief_num = thief_num # 小偷个数
        self.obstacle_num = obstacle_num # 障碍物个数
        self.perception_range = perception_range # 保安对讲机通信范围
        self.theft_time = theft_time # 完成偷盗所需时间
        self.agent_list = [] # 保安列表
        self.thief_list = {} # 小偷列表, key: thief id, value: thief类
        self.screen_width = 1200 # 可视化界面大小
        self.screen_height = 800
        self.loss = 0 # 损失
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
                  self.grid_size)) * self.graph.w  
        self.y = []
        for i in range(self.y_min + int(self.grid_size / 2), self.y_max,
                       self.grid_size):
            self.y = self.y + [i] * self.graph.l

        self.actions = ['n', 's', 'w', 'e', 'stop']  # 上下左右以及不动五个动作

        self.t = dict()  # 状态转移的数据格式为字典
        self.viewer = None # 可视化界面类
        self.obstacle_mask = None # 障碍物mask
        self.init_env()
        self.init_state = None
        # self.init_state = self.GetInitState() # 需要初始化每个保安的初始位置 
        if self.init_state is None:  # 随机初始化位置
            self.init_state = [
                0, self.graph.l - 1,
                self.graph.l * self.graph.w - self.graph.l,
                self.graph.l * self.graph.w - 1
            ]
        self.state = self.init_state
        self.init_agent()

    def init_env(self): # 初始化环境
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
            agent = Agent(self.state[num], num, self.grid_size)
            self.agent_list.append(agent)

    def generate_thief(self): # 随机产生小偷
        invalid_loc = self.obstacle_mask.copy() # 去除障碍物位置
        invalid_loc = invalid_loc.flatten()
        for agent in self.agent_list:
            invalid_loc[agent.get_state()] = 1
        valid_loc = np.where(invalid_loc == 0)[0]
        loc_list = np.random.choice(valid_loc,
                                    size=self.thief_num,
                                    replace=False)
        for id, loc in enumerate(loc_list):
            thief = Thief(loc_list[id], id, self.grid_size)
            thief.render() # 渲染小偷
            self.viewer.add_geom(thief.viz)
            thief.set_location(self.x[loc], self.y[loc])
            self.thief_list[id] = thief

    def cal_loss(self): # 计算损失
        keys = list(self.thief_list.keys())
        self.loss += len(keys)
        for k in keys:
            self.thief_list[k].escape = 1  # 逃跑成功
            self.viewer.geoms.remove(self.thief_list[k].viz)  # 删除图形显示
            self.thief_list.pop(k)  # 删除键

    def cal_weight(self, me, neighbor, current_time):
        xm = me.get_state() // self.graph.l
        ym = me.get_state() % self.graph.l
        xn = neighbor.getId() // self.graph.l
        yn = neighbor.getId() % self.graph.l
        detect_means = np.mean(self.detected_graph) # 是否需要用平均频次来作为参考

        thief_inc = self.prob_distrib[xn, yn] - self.prob_distrib[xm, ym] # 概率分布增量
        
        time_delta = 0
        cnt = 0
        cnt_list = []
        cnt_list.append(me.get_state())
        for next_neighbor in list(self.graph.getVertex(neighbor.getId()).getConnections()):
            cnt_list.append(next_neighbor)
            for nnext_neighbor in list(self.graph.getVertex(next_neighbor).getConnections()):
                if nnext_neighbor not in cnt_list:
                    xnn = nnext_neighbor // self.graph.l
                    ynn = nnext_neighbor % self.graph.l
                    time_delta += current_time - self.time_flow[xnn, ynn] # 时间间隔
                    cnt += 1
                    cnt_list.append(nnext_neighbor)
        time_delta /= max(cnt,1)

        detect_need = 0 # invert
        cnt = 0.5
        cnt_list = []
        cnt_list.append(me.get_state())
        cnt_list.append(neighbor.getId())
        detect_need += 0.5*self.detected_graph[xn,yn]
        for next_neighbor in list(self.graph.getVertex(neighbor.getId()).getConnections()):
            if next_neighbor not in cnt_list:
                xnn = next_neighbor // self.graph.l
                ynn = next_neighbor % self.graph.l
                detect_need += 0.5*self.detected_graph[xnn, ynn]
                cnt += 0.5
                cnt_list.append(next_neighbor)
            for nnext_neighbor in list(self.graph.getVertex(next_neighbor).getConnections()):
                if nnext_neighbor not in cnt_list:
                    xnn = nnext_neighbor // self.graph.l
                    ynn = nnext_neighbor % self.graph.l
                    detect_need += 0.3*self.detected_graph[xnn, ynn]
                    cnt += 0.3
                    cnt_list.append(nnext_neighbor)

        nearby_repulse = 0# negative value
        for nearby_agent_id in me.guard_nearby:
            nearby_agent = self.agent_list[nearby_agent_id]
            xna = nearby_agent.get_state() // self.graph.l
            yna = nearby_agent.get_state() % self.graph.l
            nearby_repulse -= ((xn - xm)*(xna - xn) + (yn - ym)*(yna - yn)) # 利用向量乘法

        #print(1000*thief_inc, 2*time_delta, -1*(detect_need- cnt*detect_means), 50*nearby_repulse)
        weight = 1000*thief_inc + 2*time_delta - 1*(detect_need - cnt*detect_means) + 50*nearby_repulse

        return weight
    
    def getAction(self, iteration):
        action = []
        #env.graph.getVertex(n).getDirection()
        for agent in self.agent_list:
            chosen = {}
            node = self.graph.getVertex(agent.get_state())
            neighbors = list(node.getConnections())
            for n in neighbors:
                weight = self.cal_weight(agent, self.graph.getVertex(n), iteration)
                chosen[n] = weight
            #print(chosen)
            choice = max(chosen, key=chosen.get)
            direction = node.connectedTo[choice][0]
            action.append(direction)
        return action

    def set_seed(self, seed): # 设置随机数种子
        random.seed(seed)
        np.random.seed(seed)

    def getStates(self): # 获得状态空间
        return self.states
    
    def detect(self, iteration):
        detect_list=[]
        for agent in self.agent_list:
            detect_list.append(agent.get_state())
            row = agent.get_state() // self.graph.l
            col = agent.get_state() % self.graph.l
            self.time_flow[row][col] = iteration
            self.detected_graph[row][col] = self.detected_graph[row][col] + self.detect_value 
            neighbors = list(self.graph.getVertex(agent.get_state()).getConnections())
            for n in neighbors:
                detect_list.append(n)
                row = n // self.graph.l
                col = n % self.graph.l
                self.time_flow[row][col] = iteration
                self.detected_graph[row][col] = self.detected_graph[row][col] + self.detect_value 
                next_neighbors = list(self.graph.getVertex(n).getConnections()) 
                for m in next_neighbors:
                    if m in detect_list:
                        continue
                    detect_list.append(m)
                    row = m // self.graph.l
                    col = m % self.graph.l
                    self.time_flow[row][col] = iteration
                    self.detected_graph[row][col] = self.detected_graph[row][col] + self.detect_value 
        #感知机器人附近是否有其他机器人
        for agent in self.agent_list:
            agent.clear_guard_nearby()
        for i in range(self.agent_num):
            for j in range(i + 1, self.agent_num):
                if self.cal_state_dist(self.agent_list[i].get_state(),
                                       self.agent_list[j].get_state()) < self.perception_range:

                    self.agent_list[i].add_guard_nearby(j)
                    self.agent_list[j].add_guard_nearby(i)
        
    def setState(self, s): # 设置状态
        self.state = s

    def getRandomAction(self): # 随机策略
        action = [agent.random_action(self) for agent in self.agent_list]
        return action

    # gym中显示的方格与obstacle_mask索引关系，以3*4(l*w)为例
    # (3,0) (3,1) (3,2)
    # (2,0) (2,1) (2,2)
    # (1,0) (1,1) (1,2)
    # (0,0) (0,1) (0,2)

    def generate_obstacle(self, num): # 随机产生障碍物
        self.obstacle_mask = np.zeros(
            (self.graph.w, self.graph.l))  # 注意array的行列顺序与gym中的顺序有所区分
        obstacle_list = random.sample(range(len(self.states)), num)
        for i in obstacle_list:
            row = i // self.graph.l
            col = i % self.graph.l
            self.obstacle_mask[row][col] = 1

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        self.obstacle_mask = cv2.dilate(self.obstacle_mask, kernel) # 膨胀障碍物
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

    def cal_state_dist(self, s1, s2): # 计算manhattan distance
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
                if self.cal_state_dist(self.agent_list[i].get_state(
                ), self.agent_list[j].get_state()) < self.perception_range:
                    self.agent_list[i].add_guard_nearby(j)
                    self.agent_list[j].add_guard_nearby(i)
        # for agent in self.agent_list:
        #     print(agent.id, agent.guard_nearby)

        # 是否抓到小偷
        for agent in self.agent_list:
            keys = list(self.thief_list.keys())
            for k in keys:
                if self.cal_state_dist(agent.get_state(),
                                       self.thief_list[k].get_state()) <= 2:
                    self.viewer.geoms.remove(self.thief_list[k].viz)  # 删除图形显示
                    self.thief_list.pop(k)  # 删除键
                    self.generate_probability_distribution()

        return next_state_list

    def reset(self): # 重置环境

        self.state = self.init_state
        return self.state

    def render(self, mode='human'): # 渲染环境

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
                for key in agent.sight_list.keys():
                    self.viewer.add_geom(agent.sight_list[key])

            for i in self.lines:
                self.viewer.add_geom(i)
            
            # 渲染障碍物
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
        for agent in self.agent_list: # 设置保安位置
            state = agent.get_state()
            # print('state:', state)
            agent.set_location(self.x[state], self.y[state])

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self): # 关闭可视化界面
        if self.viewer:
            self.viewer.close()
    
    def generate_probability_distribution(self):
        # 获取小偷概率分布图（从左上角开始的）
        # kong
        self.prob_mask = copy.deepcopy(self.obstacle_mask)
        self.prob_mask *= -1
        self.prob_mask += 1
        k_size = 11
        gauss_kernel = cv2.getGaussianKernel(k_size, 1)
        gauss_kernel = np.dot(gauss_kernel, gauss_kernel.T)
        gauss_kernel[gauss_kernel < 0.001] = 0.001

        self.prob_distrib = np.zeros((self.graph.w, self.graph.l))
        keys = list(self.thief_list.keys())
        for key in keys:
            thief = self.thief_list[key]
            i = thief.get_state()
            x0 = i // self.graph.l
            y0 = i % self.graph.l
            x1 = max(0, x0 - (k_size // 2))
            x2 = min(self.graph.w - 1, x0 + (k_size // 2))
            y1 = max(0, y0 - (k_size // 2))
            y2 = min(self.graph.l - 1, y0 + (k_size // 2))
            x3 = max(0, (k_size // 2) - x0)
            x4 = min(k_size - 1, self.graph.w - 1 - x0 + (k_size // 2))
            y3 = max(0, (k_size // 2) - y0)
            y4 = min(k_size - 1, self.graph.l - 1 - y0 + (k_size // 2))
            self.prob_distrib[x1:x2, y1:y2] += gauss_kernel[x3:x4, y3:y4]

        self.prob_distrib = self.prob_distrib * self.prob_mask


if __name__ == '__main__':
    g = Rect_Graph(40, 36)
    env = School(g, 4, 5, 6, 10, 8)
    env.reset()

    for i in range(1000):
        env.detect(i)
        print('times:', i)
        if i % 50 == 10:
            env.generate_thief()
            env.generate_probability_distribution()
        if i % 50 == 35: 
            env.cal_loss()
            print('loss:', env.loss)
        # if i ==3 :
        #     del env.thief_list[1]
        env.render()
        action = env.getAction(i)
        #action = env.getRandomAction()  # list
        # print(env.state)
        env.step(action)
        time.sleep(1)

