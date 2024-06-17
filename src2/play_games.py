import gym
import card_game_env as card_game_env 
from policy import CategoricalPolicy
from network_utils import build_mlp
from config import config_thirteen
import torch
import numpy as np
 
import pdb



env = gym.make('CardGame-v0')
# env.seed(0)
observation_dim = env.observation_space.n
action_dim = env.action_space.n

num_games = 100

PATH = "baseline_rand.pth"
network = build_mlp(observation_dim, action_dim, 1, 64)
network.load_state_dict(torch.load(PATH))
network.eval()
baseline_policy = CategoricalPolicy(network)

PATH = "no_baseline_rand.pth"
network = build_mlp(observation_dim, action_dim, 1, 64)
network.load_state_dict(torch.load(PATH))
network.eval()
no_baseline_policy = CategoricalPolicy(network)

PATH = "ppo_rand.pth"
network = build_mlp(observation_dim, action_dim, 1, 64)
network.load_state_dict(torch.load(PATH))
network.eval()
ppo_policy = CategoricalPolicy(network)

state = env.reset()
wins = [0, 0, 0, 0]

for i in range(num_games):
    # print(f"game {i}")
    while True:
        if env.current_player == 0:
            # print("0")
            action = ppo_policy.act(state[None])[0]
            # action = baseline_policy.act(state[None])[0]
            # action = no_baseline_policy.act(state[None])[0]
        else:
            player_hand_numeric = np.array([env.deck.index(card) for card in env.player_hands[env.current_player]])
            if not env.won_round_must_play:
                valid_actions = np.array([0])
            else:
                valid_actions = np.array([])
            # print(valid_actions)
            must_beat = env.deck.index(env.current_play) if env.current_play else -1
            valid_actions2 = np.array(player_hand_numeric[player_hand_numeric > must_beat]) + 1
            # print("player hand: ", env.player_hands[env.current_player])
            # print(valid_actions2)
            valid_actions = np.concatenate([valid_actions, valid_actions2])
            # print(valid_actions)
            action = np.random.choice(valid_actions)
    
        state, reward, done, _, _ = env.step(int(action))

        if done:
            wins[env.current_player] += 1
            state = env.reset()
            break    

print(wins)
env.close()