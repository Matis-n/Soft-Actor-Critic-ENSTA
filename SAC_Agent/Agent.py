"""
A bit of help from
https://towardsdatascience.com/in-depth-review-of-soft-actor-critic-91448aba63d4
"""

from Nets.Actor import GaussianPolicy as Actor
from Nets.Critic import Critic
from Utils.memory import Memory
import gym
import tensorflow as tf
import numpy as np
import time
from gym.utils.save_video import save_video
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.ndimage import gaussian_filter1d

physical_devices = tf.config.list_physical_devices('CPU')  # GPU

# tf.config.experimental.set_memory_growth(physical_devices[0], True)


class Agent:
    def __init__(self, env, summary_writer, method_alternative,
                total_eps=1e2, batch_size=256, alpha=0.2, max_mem_length=1e6, tau=0.005,
                Q_lr=3e-4, policy_lr=3e-4, alpha_lr=3e-4, gamma=0.99):
        self.env = env
        self.method_alternative = method_alternative
        self.env_name = env.unwrapped.spec.id

        self.total_eps = int(total_eps)
        self.steps = 0
        self.total_scores = []
        self.batch_size = batch_size
        self.tau = tau
        self.memory = Memory(int(max_mem_length))
        self.gamma = gamma

        self.Actor = Actor(env.action_space.shape[0])
        self.Q1 = Critic()
        self.Q2 = Critic()
        self.target_Q1 = Critic()
        self.target_Q1.set_weights(self.Q1.get_weights())
        self.target_Q2 = Critic()
        self.target_Q2.set_weights(self.Q2.get_weights())

        
        self.alpha = tf.Variable(alpha, dtype=tf.float32)

        self.entropy_target = tf.constant(-env.action_space.shape[0], dtype=tf.float32)

        self.Q1_optimizer = tf.keras.optimizers.Adam(Q_lr)
        self.Q2_optimizer = tf.keras.optimizers.Adam(Q_lr)
        self.Actor_optimizer = tf.keras.optimizers.Adam(policy_lr)
        self.alpha_optimizer = tf.keras.optimizers.Adam(alpha_lr)

        self.summary_writer = summary_writer
        self.steps_list = []
        self.Q1_loss_list = []
        self.Q2_loss_list = []
        self.Actor_loss_list = []
        self.Alpha_loss_list = []
        self.Alpha_value_list = []


    def run(self, scale_reward=1, max_steps_per_epoch=1e6):
        for ep in range(self.total_eps):
            print(f'=========== Epoch n°{ep}==============\n')
            total_score = 0
            done = False
            state = self.env.reset()[0]
            current_step = 0
            while (not done) and (current_step < max_steps_per_epoch):
                current_step += 1
                print(f"step {current_step}")
                self.steps +=1
                action, log_pi = self.Actor.sample_action(state[np.newaxis, :])
                action = np.squeeze(action, axis=0)
                # interpolate to get action is form for env's space
                env_action = self.env.action_space.low + (action + 1)/2 * (self.env.action_space.high - self.env.action_space.low)
                next_state, reward, done, info, _ = self.env.step(env_action)
                total_score += reward
                reward *= scale_reward
                print(f"total_score = {total_score}")
                self.memory.add((state, action, reward, next_state, 1 - done))
                state = next_state

                if len(self.memory.buffer) > self.batch_size:  
                    self.learn()

            
            with self.summary_writer.as_default():
                tf.summary.scalar("total score", total_score, self.steps)
            self.total_scores.append(total_score)
        
        self.env.close()
        

        return (self.steps_list, self.total_scores, self.Q1_loss_list, self.Q2_loss_list, self.Actor_loss_list, self.Alpha_loss_list, self.Alpha_value_list)


    def learn(self):
        batch = self.memory.sample(self.batch_size)

        states = np.array([each[0] for each in batch])
        actions = np.array([each[1] for each in batch])
        rewards = np.array([each[2] for each in batch]).reshape(self.batch_size, 1)
        next_states = np.array([each[3] for each in batch])
        dones = np.array([each[4] for each in batch]).reshape(self.batch_size, 1)
        
        # Comptute the Q_expected (neeeded to learn Q action-value fucntion model)
        next_actions, next_log_pi = self.Actor.sample_action(next_states)
        next_Q1 = self.target_Q1([next_states, next_actions])
        next_Q2 = self.target_Q2([next_states, next_actions])
        next_Q_target = tf.minimum(next_Q1, next_Q2) - self.alpha * next_log_pi  # is V_psi equation 3 (make target soft)
        Q_expected = rewards + dones * self.gamma * next_Q_target 

        # Learn Q1, Q2
        with tf.GradientTape() as tape1, tf.GradientTape() as tape2:
            tape1.watch(self.Q1.trainable_variables)
            tape2.watch(self.Q2.trainable_variables)
            Q1_predict = self.Q1([states, actions])
            Q2_predict = self.Q2([states, actions])
            loss_Q1 = tf.reduce_mean((Q_expected - Q1_predict)**2) # equation 7
            loss_Q2 = tf.reduce_mean((Q_expected - Q2_predict)**2) # equation 7
        Q1_gradient = tape1.gradient(loss_Q1, self.Q1.trainable_variables)
        Q2_gradient = tape2.gradient(loss_Q2, self.Q2.trainable_variables)
        self.Q1_optimizer.apply_gradients(zip(Q1_gradient, self.Q1.trainable_variables))
        self.Q2_optimizer.apply_gradients(zip(Q2_gradient, self.Q2.trainable_variables))
        del tape1, tape2

        # Learn policy
        with tf.GradientTape() as actor_tape:
            actor_tape.watch(self.Actor.trainable_variables)
            predicted_actions, predicted_log_pi = self.Actor.sample_action(states) # equation 11
            Q1_predicted = self.Q1([states, predicted_actions])
            Q2_predicted = self.Q2([states, predicted_actions])
            Q_target = tf.minimum(Q1_predicted, Q2_predicted)
            actor_loss = tf.reduce_mean(self.alpha*predicted_log_pi - Q_target) # equation 12
            print(f"actor loss = {actor_loss}")
        actor_gradient = actor_tape.gradient(actor_loss, self.Actor.trainable_variables)
        self.Actor_optimizer.apply_gradients(zip(actor_gradient, self.Actor.trainable_variables))
        del actor_tape


        # Learn alpha temperature
        with tf.GradientTape() as alpha_tape:
            alpha_tape.watch(self.alpha)
            alpha_loss = tf.reduce_mean(self.alpha * (-predicted_log_pi - self.entropy_target))
        alpha_gradient = alpha_tape.gradient(alpha_loss, [self.alpha])
        del alpha_tape
        self.alpha_optimizer.apply_gradients(zip(alpha_gradient, [self.alpha]))
        self.update_targets()

        with self.summary_writer.as_default():
            tf.summary.scalar("Q1 loss", loss_Q1, self.steps)
            tf.summary.scalar("Q2 loss", loss_Q2, self.steps)
            tf.summary.scalar("Actor loss", actor_loss, self.steps)
            tf.summary.scalar("alpha loss", alpha_loss, self.steps)
            tf.summary.scalar("alpha value", self.alpha, self.steps)
        
        self.steps_list.append(self.steps)
        self.Q1_loss_list.append(loss_Q1)
        self.Q2_loss_list.append(loss_Q2)
        self.Actor_loss_list.append(actor_loss)
        self.Alpha_loss_list.append(alpha_loss)
        self.Alpha_value_list.append(self.alpha)



    def update_targets(self):
        self.target_Q1.set_weights([tf.math.multiply(local_weight, self.tau) +
                                    tf.math.multiply(target_weight, 1-self.tau)
                                      for local_weight, target_weight in
                                      zip(self.Q1.get_weights(), self.target_Q1.get_weights())])
        self.target_Q2.set_weights([tf.math.multiply(local_weight, self.tau) +
                                    tf.math.multiply(target_weight, 1-self.tau)
                                      for local_weight, target_weight in
                                      zip(self.Q2.get_weights(), self.target_Q2.get_weights())])
