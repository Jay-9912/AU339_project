import logging
import random
import gym
import time
from Graph import *

logger = logging.getLogger(__name__)

class School(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 1
    }

    def __init__(self, graph):
        self.graph = graph
        self.scale = 1
        self.states = range(self.graph.l*self.graph.w) # 状态空间

        self.x = [150,250,350,450] * 4
        self.y=[150] * 4 + [250] * 4 + [350] * 4 + [450] * 4

        self.terminate_states = dict()  # 终止状态为字典格式
        # self.terminate_states[11] = 1
        # self.terminate_states[12] = 1
        # self.terminate_states[15] = 1

        self.actions = ['n','s','w','e','stop']  # 上下左右以及不动五个动作

        self.rewards = dict();        # 回报的数据结构为字典
        # self.rewards['8_s'] = -1.0
        # self.rewards['13_w'] = -1.0
        # self.rewards['7_s'] = -1.0
        # self.rewards['10_e'] = -1.0
        # self.rewards['14_4'] = 1.0

        self.t = dict();             # 状态转移的数据格式为字典
        for state in range(self.graph.l*self.graph.w):

            key = "%d_stop"%(state)
            self.t[key] = state
            for action in self.actions[:4]:
                if action in self.graph.getVertex(state).getDirection():
                    key = "%d_%s"%(state, action)
                    self.t[key] = self.graph.getVertex(state).getNeighbor(action)
        # print(self.t)
        self.gamma = 0.8         #折扣因子
        self.viewer = None
        self.state = None

    def _seed(self, seed=None):
        self.np_random, seed = random.seeding.np_random(seed)
        return [seed]

    def getTerminal(self):
        return self.terminate_states

    def getGamma(self):
        return self.gamma

    def getStates(self):
        return self.states

    def getAction(self):
        return self.actions

    def getTerminate_states(self):
        return self.terminate_states

    def setState(self,s):
        self.state = s

    def random_action(self):
        valid_action = list(self.graph.getVertex(self.state).getDirection()) + ['stop']
        return random.sample(valid_action, 1)[0]

    def step(self, action): # 更新state
        # 系统当前状态
        state = self.state
        # if state in self.terminate_states:
        #     return state, 0, True, {}
        key = "%d_%s"%(state, action)   

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

        return next_state, r, is_terminal,{}

    def reset(self):
        # self.state = self.states[int(random.random() * len(self.states))]
        self.state = 12
        return self.state

    def render(self, mode='human'):
        from gym.envs.classic_control import rendering
        screen_width = 600 * self.scale
        screen_height = 600 * self.scale

        if self.viewer is None:
            self.viewer = rendering.Viewer(screen_width, screen_height)

            #创建网格世界
            self.lines = []
            for i in range(5):
                line = rendering.Line((100,100*i+100), (500, 100*i+100))
                line.set_color(0,0,0)
                self.lines.append(line)
            for i in range(5):
                line = rendering.Line((100*i+100,100), (100*i+100, 500))
                line.set_color(0,0,0)
                self.lines.append(line)
            # self.line1 = rendering.Line((100,100),(500,100))
            # self.line2 = rendering.Line((100, 200), (500, 200))
            # self.line3 = rendering.Line((100, 300), (500, 300))
            # self.line4 = rendering.Line((100, 400), (500, 400))
            # self.line5 = rendering.Line((100, 500), (500, 500))
            # self.line6 = rendering.Line((100, 100), (100, 500))
            # self.line7 = rendering.Line((200, 100), (200, 500))
            # self.line8 = rendering.Line((300, 100), (300, 500))
            # self.line9 = rendering.Line((400, 100), (400, 500))
            # self.line10 = rendering.Line((500, 100), (500, 500))

            # #创建石柱
            # self.shizhu = rendering.make_circle(40)
            # self.circletrans = rendering.Transform(translation=(250,350))
            # self.shizhu.add_attr(self.circletrans)
            # self.shizhu.set_color(0.8,0.6,0.4)

            # #创建第一个火坑
            # self.fire1 = rendering.make_circle(40)
            # self.circletrans = rendering.Transform(translation=(450, 250))
            # self.fire1.add_attr(self.circletrans)
            # self.fire1.set_color(1, 0, 0)

            # #创建第二个火坑
            # self.fire2 = rendering.make_circle(40)
            # self.circletrans = rendering.Transform(translation=(150, 150)) # 左下角
            # self.fire2.add_attr(self.circletrans)
            # self.fire2.set_color(1, 0, 0)

            # #创建宝石
            # self.diamond = rendering.make_circle(40)
            # self.circletrans = rendering.Transform(translation=(450, 150))
            # self.diamond.add_attr(self.circletrans)
            # self.diamond.set_color(0, 0, 1)

            #创建机器人
            self.robot= rendering.make_circle(30)
            self.robotrans = rendering.Transform()
            self.robot.add_attr(self.robotrans)
            self.robot.set_color(0, 1, 0)

            # self.line1.set_color(0, 0, 0)
            # self.line2.set_color(0, 0, 0)
            # self.line3.set_color(0, 0, 0)
            # self.line4.set_color(0, 0, 0)
            # self.line5.set_color(0, 0, 0)
            # self.line6.set_color(0, 0, 0)
            # self.line7.set_color(0, 0, 0)
            # self.line8.set_color(0, 0, 0)
            # self.line9.set_color(0, 0, 0)
            # self.line10.set_color(0, 0, 0)

            for i in self.lines:
                self.viewer.add_geom(i)
            # self.viewer.add_geom(self.line1)
            # self.viewer.add_geom(self.line2)
            # self.viewer.add_geom(self.line3)
            # self.viewer.add_geom(self.line4)
            # self.viewer.add_geom(self.line5)
            # self.viewer.add_geom(self.line6)
            # self.viewer.add_geom(self.line7)
            # self.viewer.add_geom(self.line8)
            # self.viewer.add_geom(self.line9)
            # self.viewer.add_geom(self.line10)
            # self.viewer.add_geom(self.shizhu)
            # self.viewer.add_geom(self.fire1)
            # self.viewer.add_geom(self.fire2)
            # self.viewer.add_geom(self.diamond)
            self.viewer.add_geom(self.robot)

        if self.state is None: 
            return None
    
        self.robotrans.set_translation(self.x[self.state], self.y[self.state])

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()

if __name__ == '__main__':
    g = Rect_Graph(4,4)
    env = School(g)
    env.reset()
    for i in range(1000):
        env.render()
        action = env.random_action()
        env.step(action)
        time.sleep(1)
    # while True:
    #     i=0
    #env.close()
    # env = gym.make('CartPole-v0')
    # env.reset()
    # for i in range(1000):
    #     env.render()
    #     env.step(env.action_space.sample()) # take a random action