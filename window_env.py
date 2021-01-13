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


###############################  airsim env ##################################
class AggEnv():
    def __init__(self, client, init_pos, window_state, time_factor=2.0, ctrl_duaration=0.04, a_bound=0.7):
        self.client = client
        self.init_pos = init_pos
        self.window_state = window_state
        self.time_factor = time_factor
        self.ctrl_duaration = ctrl_duaration
        self.a_bound = a_bound

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

        # temp_z = 2
        # pos = self.client.getPosition()
        # # self.client.hover()
        # # self.client.moveToPosition(pos.x_val, pos.y_val, temp_z, 5)
        # # time.sleep(1)
        # # # self.client.hover()
        # # self.client.moveToPosition(initX, initY, temp_z, 5)
        # # time.sleep(1)
        # # self.client.hover()
        # self.client.moveToPositionAsync(initX, initY, initZ, 5).join()
        # self.client.hover()
        # time.sleep(1.5)
        # self.client.moveByAngleAsync(0, 0, initZ, 0, 5).join()
        # self.client.hover()
        # time.sleep(1.5)
        # print("return")

    def step(self, action):
        self.client.moveByAngleZAsync(-0.1, action, self.init_pos[2], 0, self.ctrl_duaration)
        time.sleep(self.ctrl_duaration / self.time_factor - 0.01)
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
        # state = pos + angle + window_state
        return np.array(state)

    def action_sample(self, act_limit, time_count):
        # print("time_count", time_count)
        if (time_count-1) % 5 == 0:
            # print("in")
            self.base_a = random.uniform(-act_limit, act_limit)
        # print("base", self.base_a)
        return self.base_a + random.uniform(-act_limit/10, act_limit/10)

    def compute_reward_done(self, quad_state, pre_state, collision_info, force_done):        
        thresh_dist_max = 1
        reward_factor = 4
        beta = 1
        pos_weight = 0.5
        angle_weight = 1.0
        angle_diff_weight = 0

        # print(quad_state)
        angle_error = quad_state[2] - self.window_state[4]
        pos_error = quad_state[1] - self.window_state[1]
        x_dist = quad_state[0] - self.window_state[0]
        angle_diff = quad_state[2]-pre_state[2]
        passWindow = False
        done = False
        # print("x:", quad_state[0])

        if (x_dist > -0.1) or collision_info.has_collided or math.fabs(quad_state[1]) > 3:
            if math.fabs(pos_error) < 2:
                passWindow = True
                done = True
            else:
                passWindow = False
                done = True

        pos_reward = (math.exp(-beta * math.fabs(pos_error)) - 0.5)
        angle_reward = (math.exp(-beta * math.fabs(angle_error)) - 0.5) 
        angle_diff_reward = -math.fabs(angle_diff)
        # print("angle_diff", angle_diff)

        if done and passWindow:
            reward = (pos_reward*pos_weight + angle_reward*angle_weight + angle_diff_reward*angle_diff_weight) * reward_factor
        elif done and (not passWindow):
            reward = -10
        elif not done:
            reward = angle_diff_reward * angle_diff_weight * reward_factor
        
        return reward, done, pos_error, angle_error

    def write_csv(self, data):
        with open('some.csv', 'a', newline='') as f: # 采用b的方式处理可以省去很多问题
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



