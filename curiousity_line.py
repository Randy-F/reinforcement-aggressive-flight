import numpy as np
import time, math, random
from timer import Timer
from dtaidistance import dtw


class CuriousityMoudle:
    """
    A simple FIFO experience replay buffer for TD3 agents.
    """

    def __init__(self, reward_scale=4, dist_max=1):
        #beta = 12 / 希望reward取到max时dist的值

        self.reward_scale = reward_scale 
        self.beta = 12 / dist_max
        pass

    def get_curious_reward(self, _state_buf, _cur_state):

        if _state_buf: #判断非空，第一次时存储buff为空
            dists = []
            # state_buf, cur_state = np.array(_state_buf), np.array(_cur_state)
            for _state in _state_buf:
                length = min(len(_state[0]), len(_cur_state[0]))
                dist = dtw.distance_fast(_state[1, :length], _cur_state[1, :length]) / length
                # dist = np.linalg.norm(_state[:length] - cur_state[:length], ord=1) / length
                dists.append(dist)
            min_dist = min(dists)
            max_dist = max(dists)
            # print(min_dist)
            return 1/(1+math.exp((-self.beta*min_dist + 6))) * self.reward_scale
        else:
            return 0

if __name__ == '__main__':
    # y1 = []
    # for _ in range(600):
    #     y = np.arange(0.0, 4.0, 0.025) + 0.01
    #     y1.append(y)
    # y2 = np.arange(0.0, 4.0, 0.025)
    y1 = []
    for _ in range(200):
        y = np.array([0.0, 1.0, 2.0 ,3.0])
        y1.append(y)
    y2 = y1

    curiousity = CuriousityMoudle(reward_scale=4)
    timer = Timer()
    timer.beginCount()
    r1 = curiousity.get_curious_reward(_state_buf=y1, _cur_state=y2)
    print(r1)
    print(timer.secondsDiff())