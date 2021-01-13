"""
Note: This is a updated version from my previous code,
for the target network, I use moving average to soft replace target parameters instead using assign function.
By doing this, it has 20% speed up on my machine (CPU).
Deep Deterministic Policy Gradient (DDPG), Reinforcement Learning.
DDPG is Actor Critic based algorithm.
Pendulum example.
View more on my tutorial page: https://morvanzhou.github.io/tutorials/
Using:
tensorflow 1.0
gym 0.8.0
"""
import setup_path 
import airsim

import numpy as np
import time
import csv, math, random
from curiousity_line import CuriousityMoudle

###############################  airsim env ##################################
class AggEnv():
    def __init__(self, client, init_pos, window_state, time_factor=2.0, ctrl_duaration=0.04, a_bound=0.7):
        self.client = client
        self.init_pos = init_pos
        self.window_state = window_state
        self.time_factor = time_factor
        self.ctrl_duaration = ctrl_duaration
        self.a_bound = a_bound
        self.pre_in_middle = True    # 上一次是否在中间，用类成员变量保存
        self.middle_count = 0
        self.base_a = 0
        self.passcount = 0
        self.suc_count = 0   # 多个任务中成功的次数
        self.curiousity = CuriousityMoudle(reward_scale=4)
        self.pos_error1, self.pos_error2 = 0, 0

    def init(self):
        self.client.reset()
        time.sleep(1)
        self.client.confirmConnection()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        self.client.takeoffAsync().join()
        self.client.moveToPositionAsync(self.init_pos[0], self.init_pos[1], self.init_pos[2], 5).join()

    def reset(self):
        self.client.reset()
        # self.client.confirmConnection()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        # self.client.takeoffAsync().join()
        self.client.moveToPositionAsync(self.init_pos[0], self.init_pos[1], self.init_pos[2], 5).join()
        self.pre_in_middle = True     # 重置在中间的记录
        self.passcount, self.base_a, self.middle_count, self.suc_count = 0, 0, 0, 0

    def step(self, action):
        self.client.moveByAngleZAsync(-0.05, action, self.init_pos[2], 0, self.ctrl_duaration)
        time.sleep(self.ctrl_duaration / self.time_factor)
        s_ = self.getState()
        collision_info = self.client.simGetCollisionInfo()
        return s_, collision_info

    def getState(self): 
        _state = self.client.simGetGroundTruthKinematics()
        pos = _state.position
        pos = [pos.x_val, pos.y_val, pos.z_val]
        vel = _state.linear_velocity 
        vel = [vel.x_val, vel.y_val, vel.z_val]
        angle = self.quad2euler(_state.orientation)
        state = [pos[0], pos[1], angle[1], vel[1]]
        # # state = pos + angle + window_state
        return np.array(state)

    def is_in_middle(self, x):
        return math.fabs(x) < 0.2

    # 检测中间穿越
    # def compute_reward_done(self, quad_state, pre_state, collision_info, force_done):        
    #     done, reward = False, 0

    #     # 抖动惩罚
    #     angle_diff_weight = 0.3
    #     angle_diff = quad_state[2]-pre_state[2]
    #     angle_diff_reward = -math.fabs(angle_diff)
    #     reward = angle_diff_reward * angle_diff_weight

    #     if (quad_state[0] > 8 or math.fabs(quad_state[1]) > 3):
    #         done = True

    #     in_middle = self.is_in_middle(quad_state[1])
    #     if ((in_middle is True) and (self.pre_in_middle is False)):
    #         self.middle_count = self.middle_count + 1
    #         if ((self.middle_count == 1 and quad_state[3] > 0 and quad_state[0] >2) or (self.middle_count == 2 and quad_state[3] < 0 and quad_state[0] >4)):
    #             reward = 5
    #     self.pre_in_middle = in_middle
        


    #     return reward, done

    #  检测关键点速度
    def compute_reward_done(self, quad_state, pre_state, collision_info, force_done, replay_buffer=None, is_curiosity=False):        
        done, reward = False, 0

        reward_factor = 2
        beta = 1
        pos_weight, vel_weight, angle_weight = 2, 5, 4

        # print(quad_state)
        pos_error = quad_state[1] - self.window_state[1]
        
        reward, done = 0.0, False
        # print("x:", quad_state[0])
        pos_reward, angle_reward = 0, 0
        if (quad_state[0] > 5 or math.fabs(quad_state[1]) > 3):
            done = True

        # 两点狭缝式约束
        # if (self.passcount == 0 and quad_state[0] > 2.5): 
        #     self.passcount = self.passcount + 1
        #     if math.fabs(pos_error) < 2:
        #         angle_error = quad_state[2] - math.pi/6
        #         pos_reward = (math.exp(-beta * math.fabs(pos_error)) - 0.5)
        #         angle_reward = (math.exp(-beta * math.fabs(angle_error)) - 0.5) 
        # elif (self.passcount == 1 and quad_state[0] > 6): 
        #     self.passcount = self.passcount + 1
        #     if math.fabs(pos_error) < 2:
        #         angle_error = quad_state[2] + math.pi/6
        #         pos_reward = math.exp(-beta * math.fabs(pos_error)) - 0.5
        #         angle_reward = (math.exp(-beta * math.fabs(angle_error)) - 0.5) 
        # reward = (pos_reward*pos_weight + angle_reward*angle_weight) * reward_factor


        #  四点约束
        # if (self.passcount == 0 and quad_state[0] > 2): 
        #     self.passcount = self.passcount + 1
        #     if math.fabs(quad_state[1]+1.6) < 1:
        #         self.suc_count = self.suc_count + 1
        #         pos_reward = math.exp(-beta * math.fabs(quad_state[1]+1.6)) - 0.5
        #         reward = (pos_reward*pos_weight) * reward_factor + 3
        # elif (self.passcount == 1 and quad_state[0] > 2.5): 
        #     self.passcount = self.passcount + 1
        #     if self.suc_count == 1:
        #         if math.fabs(pos_error) < 1:
        #             self.suc_count = self.suc_count + 1
        #             pos_reward = math.exp(-beta * math.fabs(pos_error)) - 0.5
        #             vel_reward = 2 if quad_state[3] > 0 else 0
        #             reward = (pos_reward*pos_weight + vel_reward*vel_weight) * reward_factor + 3
        #         else:
        #             reward = 0
        # elif (self.passcount == 2 and quad_state[0] > 5): 
        #     self.passcount = self.passcount + 1
        #     if self.suc_count == 2:
        #         if math.fabs(quad_state[1]-1.6) < 1:
        #             self.suc_count = self.suc_count + 1
        #             pos_reward = math.exp(-beta * math.fabs(quad_state[1]-1.6)) - 0.5
        #             reward = (pos_reward*pos_weight) * reward_factor + 3
        #         else:
        #             reward = 0
        # elif (self.passcount == 3 and quad_state[0] > 6): 
        #     self.passcount = self.passcount + 1
        #     if self.suc_count == 3:
        #         if math.fabs(pos_error) < 1:
        #             self.suc_count = self.suc_count + 1
        #             pos_reward = math.exp(-beta * math.fabs(pos_error)) - 0.5
        #             vel_reward = 2 if (quad_state[3] < 0) else 0
        #             reward = (pos_reward*pos_weight + vel_reward*vel_weight) * reward_factor + 3
        #         else:
        #             reward = 0

        #  两点速度约束
        # if (self.passcount == 0 and quad_state[0] > 2.5): 
        #     self.passcount = self.passcount + 1
        #     if math.fabs(pos_error) < 1 and quad_state[3] > 0:
        #         self.suc_count = self.suc_count + 1
        #         pos_reward = math.exp(-beta * math.fabs(pos_error)) - 0.5
        #         vel_reward = -math.exp(-beta * quad_state[3]) + 1
        #         reward = (pos_reward*pos_weight + vel_reward*vel_weight) * reward_factor + 3
        # elif (self.passcount == 1 and quad_state[0] > 6): 
        #     self.passcount = self.passcount + 1
        #     if self.suc_count == 1:
        #         if math.fabs(pos_error) < 1 and quad_state[3] < 0:
        #             self.suc_count = self.suc_count + 1
        #             pos_reward = math.exp(-beta * math.fabs(pos_error) - 0.5)
        #             vel_reward = -math.exp(beta * quad_state[3]) + 1
        #             reward = (pos_reward*pos_weight + vel_reward*vel_weight) * reward_factor + 3
        #         else:
        #             reward = 0

        if (self.passcount == 0 and quad_state[0] > 1.5): 
            self.passcount = self.passcount + 1
            if math.fabs(pos_error+0.15) < 0.3 and quad_state[3] > 0:
                self.suc_count = self.suc_count + 1
                pos_reward = math.exp(-beta * math.fabs(pos_error)) - 0.5
                vel_reward = -math.exp(-beta * quad_state[3]) + 1
                reward = (pos_reward*pos_weight + vel_reward*vel_weight) * reward_factor + 3
                self.pos_error1 = pos_error

        elif (self.passcount == 1 and quad_state[0] > 3): 
            self.passcount = self.passcount + 1
            if self.suc_count == 1:
                if math.fabs(pos_error-0.3) < 0.3 and quad_state[3] < 0:
                    self.suc_count = self.suc_count + 1
                    pos_reward = math.exp(-beta * math.fabs(pos_error) - 0.5)
                    vel_reward = -math.exp(beta * quad_state[3]) + 1
                    reward = (pos_reward*pos_weight + vel_reward*vel_weight) * reward_factor + 3
                    self.pos_error2 = pos_error
                else:
                    reward = 0

        if is_curiosity is True:
            replay_buffer.store_traj(quad_state, done)
            if done is True:
                reward_curiousity = self.curiousity.get_curious_reward(replay_buffer.trajs_buf[:-1], replay_buffer.trajs_buf[-1])
                reward += reward_curiousity
        # replay_buffer.store_traj(quad_state, done)
        # if done is True:
        #     reward_curiousity = self.curiousity.get_curious_reward(replay_buffer.trajs_buf[:-1], replay_buffer.trajs_buf[-1])
        #     reward += reward_curiousity
        #  上次成功的
        # if (self.passcount == 0 and quad_state[0] > 2.5): 
        #     self.passcount = self.passcount + 1
        #     if math.fabs(pos_error) < 0.4 and quad_state[3] > 0:
        #         self.suc_count = self.suc_count + 1
        #         pos_reward = math.exp(-beta * math.fabs(pos_error)) - 0.5
        #         vel_reward = -math.exp(-beta * quad_state[3]) + 0.5
        #         reward = (pos_reward*pos_weight + vel_reward*vel_weight) * reward_factor + 3
        # elif (self.passcount == 1 and quad_state[0] > 6): 
        #     self.passcount = self.passcount + 1
        #     if self.suc_count == 1:
        #         if math.fabs(pos_error) < 0.4 and quad_state[3] < 0:
        #             self.suc_count = self.suc_count + 1
        #             pos_reward = math.exp(-beta * math.fabs(pos_error) - 0.5)
        #             vel_reward = -math.exp(beta * quad_state[3]) + 0.5
        #             reward = (pos_reward*pos_weight + vel_reward*vel_weight) * reward_factor + 3
        #         else:
        #             reward = 0


        return reward, done, self.pos_error1, self.pos_error2

    def action_sample(self, act_limit, time_count):
        # print("time_count", time_count)
        if (time_count-1) % 3 == 0:
            # print("in")
            self.base_a = random.uniform(-act_limit, act_limit)
        return self.base_a + random.uniform(-act_limit/10, act_limit/10)

        # return random.uniform(-act_limit, act_limit)

    def write_csv(self, data):
        with open('some.csv', 'a', newline='') as f:  # 采用b的方式处理可以省去很多问题
            writer = csv.writer(f)
            writer.writerow(data)

    def quad2euler(self, q):
        z = q.z_val
        y = q.y_val
        x = q.x_val
        w = q.w_val
        ysqr = y * y

        # roll (x-axis rotation)
        t0 = +2.0 * (w*x + y*z)
        t1 = +1.0 - 2.0*(x*x + ysqr)
        roll = math.atan2(t0, t1)

        # pitch (y-axis rotation)
        t2 = +2.0 * (w*y - z*x)
        if (t2 > 1.0):
            t2 = 1
        if (t2 < -1.0):
            t2 = -1.0
        pitch = math.asin(t2)

        # yaw (z-axis rotation)
        t3 = +2.0 * (w*z + x*y)
        t4 = +1.0 - 2.0 * (ysqr + z*z)
        yaw = math.atan2(t3, t4)

        return [pitch, roll, yaw]

    def get_noise_scale(self, step_count):
        scale_factor = 0.001
        return math.atan(1 / ((step_count+1) * scale_factor)) / (math.pi / 2)



