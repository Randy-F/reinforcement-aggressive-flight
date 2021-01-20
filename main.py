import numpy as np
import tensorflow as tf
import time
import sys
sys.path.append("../")
from td3_sp import core
from td3_sp.core import get_vars, mlp_actor_critic
from window_env import AggEnv
import time, csv, math, random
import setup_path 
import airsim

class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for TD3 agents.
    """

    def __init__(self, obs_dim, act_dim, size):
        self.obs1_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.obs2_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.acts_buf = np.zeros([size, act_dim], dtype=np.float32)
        self.rews_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs, done):
        self.obs1_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        return dict(obs1=self.obs1_buf[idxs],
                    obs2=self.obs2_buf[idxs],
                    acts=self.acts_buf[idxs],
                    rews=self.rews_buf[idxs],
                    done=self.done_buf[idxs])


"""

TD3 (Twin Delayed DDPG)

"""
def td3(mlp_actor_critic=core.mlp_actor_critic,
        ac_kwargs=dict(), seed=0,
        steps_per_epoch=5000, epochs=250,
        replay_size=int(1e6), gamma=0.99,
        polyak=0.995, pi_lr=5e-4, q_lr=5e-4,
        batch_size=256, start_steps=0,
        act_noise=0.1, target_noise=0.2,
        noise_clip=0.5, policy_delay=2,
        max_ep_len=1000, logger_kwargs=dict(), save_freq=200):

    tf.set_random_seed(seed)
    np.random.seed(seed)

    obs_dim = 4
    act_dim = 1

    # Action limit for clamping: critically, assumes all dimensions share the same bound!
    act_limit = 0.7

    # Share information about action space with policy architecture
    ac_kwargs['action_space'] = act_limit

    # Inputs to computation graph
    x_ph, a_ph, x2_ph, r_ph, d_ph = core.placeholders(obs_dim, act_dim, obs_dim, None, None)

    # Main outputs from computation graph
    with tf.variable_scope('main'):
        pi, q1, q2, q1_pi = mlp_actor_critic(x_ph, a_ph, **ac_kwargs)

    # Target policy network
    with tf.variable_scope('target'):
        pi_targ, _, _, _ = mlp_actor_critic(x2_ph, a_ph, **ac_kwargs)

    # Target Q networks
    with tf.variable_scope('target', reuse=True):

        # Target policy smoothing, by adding clipped noise to target actions
        epsilon = tf.random_normal(tf.shape(pi_targ), stddev=target_noise)
        epsilon = tf.clip_by_value(epsilon, -noise_clip, noise_clip)
        a2 = pi_targ + epsilon
        a2 = tf.clip_by_value(a2, -act_limit, act_limit)

        # Target Q-values, using action from target policy
        _, q1_targ, q2_targ, _ = mlp_actor_critic(x2_ph, a2, **ac_kwargs)

    # Experience buffer
    replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_size)

    # Count variables
    var_counts = tuple(core.count_vars(scope)
                       for scope in ['main/pi',
                                     'main/q1',
                                     'main/q2',
                                     'main'])
    print('\nNumber of parameters: \t pi: %d, \t q1: %d, \t q2: %d, \t total: %d\n' % var_counts)

    # Bellman backup for Q functions, using Clipped Double-Q targets
    min_q_targ = tf.minimum(q1_targ, q2_targ)
    backup = tf.stop_gradient(r_ph + gamma * (1 - d_ph) * min_q_targ)

    # TD3 losses
    pi_loss = -tf.reduce_mean(q1_pi)
    q1_loss = tf.reduce_mean((q1 - backup) ** 2)
    q2_loss = tf.reduce_mean((q2 - backup) ** 2)
    q_loss_sample = (q1 - backup) ** 2 + (q2 - backup) ** 2
    # 为啥这里的loss是加起来的?
    q_loss = q1_loss + q2_loss

    # Separate train ops for pi, q
    pi_optimizer = tf.train.AdamOptimizer(learning_rate=pi_lr)
    q_optimizer = tf.train.AdamOptimizer(learning_rate=q_lr)
    train_pi_op = pi_optimizer.minimize(pi_loss,
                                        var_list=get_vars('main/pi'))
    # 这里的参数,怎么是总的q?
    # 难道这里的字符串只需要匹配就好了?
    train_q_op = q_optimizer.minimize(q_loss,
                                      var_list=get_vars('main/q'))

    # Polyak averaging for target variables
    target_update = tf.group([tf.assign(v_targ, polyak * v_targ + (1 - polyak) * v_main)
                              for v_main, v_targ in zip(get_vars('main'), get_vars('target'))])

    # Initializing targets to match main variables
    target_init = tf.group([tf.assign(v_targ, v_main)
                            for v_main, v_targ in zip(get_vars('main'), get_vars('target'))])

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    sess.run(target_init)
    saver = tf.train.Saver()
    saver.restore(sess, './save/model14200.ckpt')

    def get_action(o, noise_scale):
        a = sess.run(pi, feed_dict={x_ph: o.reshape(1, -1)})[0]
        a += noise_scale * np.random.randn(act_dim)
        return np.clip(a, -act_limit, act_limit)



    with open('some.csv', 'w', newline='') as f:
        f.truncate()
        f.close()

    timeFactor, ctrlDuaration = 2.0, 0.04
    init_pos = (0, 0, 0)
    window_state = (1.5, -0.83, 9.8, 0, 0.50, 0)

    # connect to the AirSim simulator 
    client = airsim.MultirotorClient()
    env = AggEnv(client, init_pos, window_state, timeFactor, ctrlDuaration)

    env.reset()
    r, d, ep_len = 0, False, 0
    s = env.getState()
    s_ = s  # 初始化赋予一样的值，让compute_reward_done形参能传入
    force_done = False

    ep_len, total_step, total_eps, ep_rwd = 0, 0, 0, 0
    # Main loop: collect experience in env and update/log each epoch
    # Main loop: collect experience in env and update/log each epoch
    while True:

        """
        Until start_steps have elapsed, randomly sample actions
        from a uniform distribution for better exploration. Afterwards, 
        use the learned policy. 
        """
        ep_len = ep_len + 1
        total_step = total_step + 1

        """
        Until start_steps have elapsed, randomly sample actions
        from a uniform distribution for better exploration. Afterwards, 
        use the learned policy (with some noise, via act_noise). 
        """
        # s = np.array([0, 0, 0, 0])
        if total_step > start_steps:
            a = get_action(s, 0).tolist()[0]
        else:
            a = env.action_sample(0.4, ep_len)
            

        # Step the env
        s_, collision_info = env.step(a)
        r, d, pe1, pe2 = env.compute_reward_done(s, s_, collision_info, force_done)
        ep_rwd = ep_rwd + r
        # Store experience to replay buffer
        replay_buffer.store(s, a, r, s_, d)
        # print('s, a', s, a)
        # Super critical, easy to overlook step: make sure to update
        # most recent observation!
        s = s_
        env.write_csv([total_eps, s[0], s[1], s[2]]) 

        if d:
            total_eps = total_eps + 1
            if total_eps % save_freq == 0:
                saver.save(sess, "./save/model"+str(total_eps)+".ckpt" )

            print('Action, ep_rwd, Done:', a, ep_rwd, d)
            ctrlRoll = 0  # 控制量也要复位！
            # for j in range(ep_len):
            #     batch = replay_buffer.sample_batch(batch_size)
            #     feed_dict = {x_ph: batch['obs1'],
            #                  x2_ph: batch['obs2'],
            #                  a_ph: batch['acts'],
            #                  r_ph: batch['rews'],
            #                  d_ph: batch['done']
            #                  }
            #     q_step_ops = [q_loss, q1, q2, train_q_op, q_loss_sample]
            #     outs = sess.run(q_step_ops, feed_dict)

            #     if j % policy_delay == 0:
            #         # Delayed policy update
            #         outs = sess.run([pi_loss, train_pi_op, target_update],
            #                         feed_dict)
            env.reset()
            # env.write_csv([pe1, pe2, ep_rwd]) 
            s, r, d, ep_rwd, ep_len = env.getState(), 0, False, 0, 0


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='Hopper-v2')
    parser.add_argument('--hid', type=int, default=256)
    parser.add_argument('--l', type=int, default=3)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=3)
    parser.add_argument('--epochs', type=int, default=250)
    parser.add_argument('--exp_name', type=str, default='td3')
    args = parser.parse_args()

    td3(mlp_actor_critic=core.mlp_actor_critic,
        ac_kwargs=dict(hidden_sizes=[args.hid] * args.l),
        gamma=args.gamma, seed=args.seed, epochs=args.epochs
        )


