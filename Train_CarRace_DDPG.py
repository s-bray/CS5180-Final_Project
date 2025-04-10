import numpy as np
import torch
import DDPG
import utils
import os
import glob
import pickle
from ActionMapping import ActionMappingClass
from SimpleTrackEnv import SimpleTrackEnvClass

def process_state(s):
    return np.reshape(s, [1, -1])

# Set parameters
dt = 0.01
state_dim = 31
action_dim = 2
max_action = 1

args = {
    'start_timesteps': 1e4, 
    'eval_freq': 5e3,
    'expl_noise': 0.2, 
    'batch_size': 256,
    'discount': 0.99,
    'tau': 0.005,
    'policy_freq': 1
}

kwargs = {
    "state_dim": state_dim,
    "action_dim": action_dim,
    "max_action": max_action,
    "discount": args['discount'],
    "tau": args['tau'],
    "policy_freq": args['policy_freq'],
}

policy = DDPG.DDPG(**kwargs)
replay_buffer = utils.ReplayBuffer(state_dim, action_dim, max_size=int(5e6))
am = ActionMappingClass()
env = SimpleTrackEnvClass()

# === Check for latest saved model ===
model_dir = 'models/DDPG'
os.makedirs(model_dir, exist_ok=True)
checkpoint_files = sorted(glob.glob(os.path.join(model_dir, "model_*_actor")), 
                          key=lambda x: int(x.split("_")[-2]))

# Initialize variables
stepcounter = 0
traincounter = 1
savecounter = 1
trainlog = []
start_episode = 0
best_reward = float('-inf')

# Load saved model and training state
if checkpoint_files:
    latest = checkpoint_files[-1]
    savecounter = int(latest.split("_")[-2])
    model_path = os.path.join(model_dir, f"model_{savecounter}")
    print(f"Resuming from checkpoint: {model_path}")
    policy.load(model_path)

    # Load optimizer states
    try:
        policy.actor_optimizer.load_state_dict(torch.load(model_path + "_actor_optimizer.pth"))
        policy.critic_optimizer.load_state_dict(torch.load(model_path + "_critic_optimizer.pth"))
        print("Optimizer states loaded.")
    except:
        print("Warning: Could not load optimizer states.")

    # Load training state
    trainlog_path = 'results/trainlog.npy'
    training_state_path = 'results/training_state.npy'
    replay_buffer_path = 'results/replay_buffer.pkl'

    if os.path.exists(trainlog_path):
        trainlog = np.load(trainlog_path, allow_pickle=True).tolist()

    if os.path.exists(training_state_path):
        training_state = np.load(training_state_path, allow_pickle=True).item()
        start_episode = training_state['start_episode']
        stepcounter = training_state['stepcounter']
        traincounter = training_state['traincounter']
        best_reward = training_state['best_reward']
        print(f"Resuming from episode {start_episode}, steps {stepcounter}, train iterations {traincounter}")

    if os.path.exists(replay_buffer_path):
        with open(replay_buffer_path, 'rb') as f:
            replay_buffer = pickle.load(f)
        print("Replay buffer loaded.")

else:
    print("No checkpoint found. Starting training from scratch.")

# === Training Loop ===
for episode in range(start_episode, 25000):
    if episode >= 12500:
        # Reduce this to 6000 - 7000
        decay_progress = (episode - 12500) / 12500
        args['expl_noise'] = max(0.05, 0.2 * (1 - decay_progress))
    
    ob = env.reset()
    ob = process_state(ob)
    done = False
    saved = False
    episode_reward = 0

    for step in range(10000):
        stepcounter += 1

        # Generate action for carA
        if stepcounter < args['start_timesteps']:
            action = np.random.uniform(-1, 1, action_dim)
        else:
            noise = np.random.normal(0, max_action * args['expl_noise'], size=action_dim)
            action = (policy.select_action(ob) + noise).clip(-max_action, max_action)

        # Perform action with action mapping
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

        if traincounter % 100000 == 0 and not saved:
            savecounter += 1
            model_name = os.path.join(model_dir, f"model_{savecounter}")
            policy.save(model_name)
            torch.save(policy.actor_optimizer.state_dict(), model_name + "_actor_optimizer.pth")
            torch.save(policy.critic_optimizer.state_dict(), model_name + "_critic_optimizer.pth")
            with open('results/replay_buffer.pkl', 'wb') as f:
                pickle.dump(replay_buffer, f)
            print(f"Model and buffer saved at training iteration {traincounter}!")
            saved = True

    # Update best reward
    episode_reward_value = float(episode_reward)
    if episode_reward_value > best_reward:
        best_reward = episode_reward_value
        print(f"New best reward: {best_reward:.4f}")
        # Uncomment to save best model
        # model_name = os.path.join(model_dir, "model_best")
        # policy.save(model_name)

    # End of episode logging
    fail_reason = env.query_fail_reason()
    # print(f'Episode: {episode}  Reward: {episode_reward_value:.1f}  Step: {step} Counter: {traincounter} Reason: {fail_reason}   New best reward: {best_reward:.4f}')
    trainlog.append([episode, episode_reward_value, step, traincounter, fail_reason])

    training_state = {
        'start_episode': episode + 1,
        'stepcounter': stepcounter,
        'traincounter': traincounter,
        'best_reward': best_reward
    }
    np.save('results/training_state.npy', training_state)

    print(f'Episode: {episode}  Reward: {episode_reward:.1f}  Steps: {step}  Counter: {traincounter}  Reason: {fail_reason}  Exploration Noise: {args["expl_noise"]:.3f}   New best reward: {best_reward:.4f}')
