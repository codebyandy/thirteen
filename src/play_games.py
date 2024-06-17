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

PATH = "pg_b_rand.pth"
network = build_mlp(observation_dim, action_dim, 1, 64)
network.load_state_dict(torch.load(PATH))
network.eval()
baseline_policy = CategoricalPolicy(network)

PATH = "pg_nb_rand.pth"
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
    print(f"game {i}")
    while True:
        # pdb.set_trace()
        # print(" ")
        # print(f"player {env.current_player}")
    #    if env.current_player != 0:
        # print(env.player_hands[env.current_player])
        # current_play = state[0]
        # cards = state[1:]
        # valid_cards = cards[cards > current_play]
        # if len(valid_cards) > 0:
        #     action = np.random.choice(valid_cards)
        # else:
        #     action = 0
        # print(current_play)
        # print(cards)
        # print(valid_cards)
        # print(action)
        # state, reward, done, _, _  = env.step(int(action))
        
        # else:
        # print(env.current_player)
        if env.current_player == 0:
            print("0")
            action = ppo_policy.act(state[None])[0]
        elif env.current_player == 1:
            print("1")
            action = baseline_policy.act(state[None])[0]
        elif env.current_player == 2:
            print("2")
            action = no_baseline_policy.act(state[None])[0]
        else:
            print("3")
            current_play = state[0]
            cards = state[1:]
            valid_cards = cards[cards > current_play]
            if len(valid_cards) > 0:
                action = np.random.choice(valid_cards)
            else:
                action = 0
        state, reward, done, _, _ = env.step(int(action))
       
        # print(state)
        # print(action)

        if done:
            wins[env.current_player] += 1
            state = env.reset()
            break    

print(wins)
env.close()