"""
A bit of help from
https://towardsdatascience.com/in-depth-review-of-soft-actor-critic-91448aba63d4
"""

# tensorboard --logdir logs
from SAC_Agent.Agent import *
from SAC_Agent.Agent_alternative import *
import seaborn as sns

""" 
To clean the logs /!\ don't change the path in rmtree()
import shutil
shutil.rmtree('logs/', ignore_errors=True)
"""
# MountainCarContinuous-v0 : max 999, eps 30
# LunarLanderContinuous-v2 : max 10000, eps 40
# Pendulum-v1 : max 200, eps 30
# BipedalWalker-v3 : max 999, eps 1000
# CarRacing-v2 : max 1000, eps 10

env_name = 'MountainCarContinuous-v0' # 'BipedalWalker-v3' 'CarRacing-v2' #'Pendulum-v1' #'LunarLanderContinuous-v2' #'MountainCarContinuous-v0' # 'BipedalWalker-v3'
env = gym.make(env_name, render_mode="rgb_array_list")

method_alternative = False
total_eps = 20
max_steps_per_epoch = 999

log_dir = 'logs/' + env_name + str(method_alternative) + str(time.time())
summary_writer = tf.summary.create_file_writer(log_dir)
print(f"Alternative method (with V function) : {method_alternative}")

steps_list_scale = []
total_scores_scale = []
Q1_loss_list_scale = []
Q2_loss_list_scale = []
Actor_loss_list_scale = []
Alpha_loss_list_scale = []
Alpha_value_list_scale =[]

for scale_reward in [1, 10, 50]:
    if not method_alternative:
        SAC = Agent(env=env, summary_writer=summary_writer, total_eps=total_eps, method_alternative=method_alternative)
        SAC.run(scale_reward = scale_reward, max_steps_per_epoch=max_steps_per_epoch)
        steps_list, total_scores, Q1_loss_list, Q2_loss_list, Actor_loss_list, Alpha_loss_list, Alpha_value_list = SAC.steps_list, SAC.total_scores, SAC.Q1_loss_list, SAC.Q2_loss_list, SAC.Actor_loss_list, SAC.Alpha_loss_list, SAC.Alpha_value_list
        steps_list_scale.append(steps_list)
        total_scores_scale.append(total_scores)
        Q1_loss_list_scale.append(Q1_loss_list)
        Q2_loss_list_scale.append(Q2_loss_list)
        Actor_loss_list_scale.append(Actor_loss_list)
        Alpha_loss_list_scale.append(Alpha_loss_list)
        Alpha_value_list_scale.append(Alpha_value_list)
        
    else:
        SAC = AgentAlternative(env=env, summary_writer=summary_writer, total_eps=total_eps, method_alternative=method_alternative)
        steps_list, total_scores, Q1_loss_list, Q2_loss_list, Actor_loss_list, Alpha_loss_list, Alpha_value_list = SAC.run(scale_reward=scale_reward, max_steps_per_epoch=max_steps_per_epoch)

sns.set_theme(style="darkgrid", palette="muted", color_codes=True)
fig, axs = plt.subplots(2, 3, figsize=(17,8))


axs[0, 0].plot(np.arange(0, len(total_scores_scale[0]), 1), total_scores_scale[0], color='b', label='1')
axs[0, 0].plot(np.arange(0, len(total_scores_scale[1]), 1), total_scores_scale[1], color='r', label='5')
axs[0, 0].plot(np.arange(0, len(total_scores_scale[2]), 1), total_scores_scale[2], color='g', label='10')
axs[0, 0].set_title('Total score')
axs[0,0].legend()
axs[0, 1].plot(np.arange(0, len(steps_list_scale[0]), 1), Q1_loss_list_scale[0], color='b', label='1')
axs[0, 1].plot(np.arange(0, len(steps_list_scale[1]), 1), Q1_loss_list_scale[1], color='r', label='5')
axs[0, 1].plot(np.arange(0, len(steps_list_scale[2]), 1), Q1_loss_list_scale[2], color='g', label='10')
axs[0, 1].set_title('Q1 loss')
axs[0,1].legend()
axs[0,2].plot(np.arange(0, len(steps_list_scale[0]), 1), Q2_loss_list_scale[0], color='b', label='1')
axs[0,2].plot(np.arange(0, len(steps_list_scale[1]), 1), Q2_loss_list_scale[1], color='r', label='5')
axs[0,2].plot(np.arange(0, len(steps_list_scale[2]), 1), Q2_loss_list_scale[2], color='g', label='10')
axs[0, 2].set_title('Q2 loss')
axs[0,2].legend()
axs[1, 0].plot(np.arange(0, len(steps_list_scale[0]), 1), Actor_loss_list_scale[0], color='b', label='1')
axs[1, 0].plot(np.arange(0, len(steps_list_scale[1]), 1), Actor_loss_list_scale[1], color='r', label='5')
axs[1, 0].plot(np.arange(0, len(steps_list_scale[2]), 1), Actor_loss_list_scale[2], color='g', label='10') 
axs[1, 0].set_title('Actor loss')
axs[1, 0].legend()
axs[1, 1].plot(np.arange(0, len(steps_list_scale[0]), 1), Alpha_loss_list_scale[0], color='b', label='1')
axs[1, 1].plot(np.arange(0, len(steps_list_scale[1]), 1), Alpha_loss_list_scale[1], color='r', label='5')
axs[1, 1].plot(np.arange(0, len(steps_list_scale[2]), 1), Alpha_loss_list_scale[2], color='g', label='10')
axs[1, 1].set_title('Alpha loss')
axs[1, 1].legend()
axs[1, 2].plot(np.arange(0, len(steps_list_scale[0]), 1), Alpha_value_list_scale[0], color='b', label='1')
axs[1, 2].plot(np.arange(0, len(steps_list_scale[1]), 1), Alpha_value_list_scale[1], color='r', label='5')
axs[1, 2].plot(np.arange(0, len(steps_list_scale[2]), 1), Alpha_value_list_scale[2], color='g', label='10')
axs[1, 2].set_title('Alpha value')
axs[1, 2].legend()

plt.savefig(f'graphs/{env_name}_{method_alternative}_scalerewards_scores.png')

