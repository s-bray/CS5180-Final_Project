import numpy as np
import torch
import os
import glob

import SAC  
import utils
from ActionMapping import ActionMappingClass
from SimpleTrackEnv import SimpleTrackEnvClass

def process_state(s):
    return np.reshape(s, [1, -1])

# Parameters
dt = 0.01
state_dim = 31
action_dim = 2
max_action = 1

args = {
    'start_timesteps': 1e4, 
    'eval_freq': 5e3,
    'expl_noise': 0.1, 
    'batch_size': 256,
    'discount': 0.99,
    'tau': 0.005,
    'alpha': 0.2,
}

kwargs = {
    "state_dim": state_dim,
    "action_dim": action_dim,
    "max_action": max_action,
    "discount": args['discount'],
    "tau": args['tau'],
    "alpha": args['alpha'],
    "automatic_entropy_tuning": True
}

policy = SAC.SAC(**kwargs)
replay_buffer = utils.ReplayBuffer(state_dim, action_dim, max_size=int(5e6))

# Action mapping
am = ActionMappingClass()
env = SimpleTrackEnvClass()

# === Checkpoint Resume Setup ===
model_dir = 'models/SAC'
os.makedirs(model_dir, exist_ok=True)
checkpoint_files = sorted(glob.glob(os.path.join(model_dir, "model_*_actor")), 
                          key=lambda x: int(x.split("_")[-2]))

stepcounter = 0
traincounter = 1
savecounter = 1
trainlog = []
start_episode = 0

# Resume if checkpoint exists
if checkpoint_files:
    latest = checkpoint_files[-1]
    savecounter = int(latest.split("_")[-2])
    model_path = os.path.join(model_dir, f"model_{savecounter}")
    print(f"Resuming from checkpoint: {model_path}")
    policy.load(model_path)

    if os.path.exists('results/trainlog.npy'):
        trainlog = np.load('results/trainlog.npy', allow_pickle=True).tolist()
        if trainlog:
            last_entry = trainlog[-1]
            start_episode = int(last_entry[0]) + 1
            stepcounter = int(last_entry[2])
            traincounter = int(last_entry[3])
else:
    print("No checkpoint found. Starting training from scratch.")

# === Training Loop ===
for episode in range(start_episode, 40000): # Edit number of total episodes to run the trainning here
    ob = env.reset()
    ob = process_state(ob)
    done = False
    saved = False
    episode_reward = 0

    for step in range(10000):
        time = step * dt
        stepcounter += 1

        # Action selection
        if stepcounter < args['start_timesteps']:
            action = np.random.uniform(-1, 1, action_dim)
        else:
            action = policy.select_action(ob)

        action_in = am.mapping(env.car.spd, env.car.steer, action[0], action[1])
        next_ob, r, done = env.step(action_in)
        next_ob = process_state(next_ob)

        replay_buffer.add(ob, action, next_ob, r, done)

        ob = next_ob
        episode_reward += r

        if done:
            break

        if stepcounter > args['start_timesteps']:
            policy.train(replay_buffer, args['batch_size'])
            traincounter += 1

        # Save periodically
        if traincounter % 100000 == 0 and not saved:
            savecounter += 1
            model_name = os.path.join(model_dir, f"model_{savecounter}")
            policy.save(model_name)
            print(f"SAC model {savecounter} saved!")
            saved = True

    fail_reason = env.query_fail_reason()
    print(f'Episode: {episode}  Reward: {episode_reward:.1f}  Step: {step}  Counter: {traincounter}  Reason: {fail_reason}')
    trainlog.append([episode, episode_reward, step, traincounter, fail_reason])

    if saved:
        os.makedirs('results', exist_ok=True)
        np.save('results/trainlog_sac.npy', trainlog)
