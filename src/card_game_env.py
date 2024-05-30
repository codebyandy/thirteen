import gym
from gym import spaces
import numpy as np
import random
from gym.envs.registration import register
import pdb

class CardGameEnv(gym.Env):
    """Custom Environment for a 13-card game"""
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(CardGameEnv, self).__init__()
        self.deck = self.initialize_deck()
        self.player_hands = [[], [], [], []]
        self.player_pass = [0, 0, 0, 0]
        self.current_play = None
        self.current_player = 0
        self.done = False

        # Define action and observation space
        self.action_space = spaces.Discrete(53)  # Assuming 13 cards + pass
        self.observation_space = spaces.Dict({
            'player_hand': spaces.MultiDiscrete([52]*13),  # A hand of up to 13 cards
            'current_play': spaces.MultiDiscrete([52])  # Current play on the table
        })

    def initialize_deck(self):
        # ranks = ['3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A', '2']
        # suits = ['Spades', 'Hearts', 'Clubs', 'Diamonds']
        # return [f"{rank} of {suit}" for suit in suits for rank in ranks]
        ranks = [i for i in range(1, 14)]
        suits = [i for i in range(1, 5)]
        return[(rank, suit) for suit in suits for rank in ranks]

    def deal_cards(self):
        random.shuffle(self.deck)
        self.player_hands = [self.deck[i*13:(i+1)*13] for i in range(4)]
        self.player_hands = [sorted(hand, key=lambda x: (x[0], x[1])) for hand in self.player_hands]

    def step(self, action):
        if action != (0, 0):  # pass
            card = action
            self.current_play = card
            self.player_hands[self.current_player].remove(card)
        else:
            self.player_pass[self.current_player] = 1
            if np.sum(self.player_pass) == 3:
                self.player_pass = [0, 0, 0, 0]
                self.current_play = (0, 0)

        self.current_player = (self.current_player + 1) % 4
        while self.player_pass[self.current_player]:
            self.current_player = (self.current_player + 1) % 4

        done = any(len(hand) == 0 for hand in self.player_hands)
        reward = 1 if done and len(self.player_hands[self.current_player]) == 0 else 0

        obs = self._get_obs()
        return obs, reward, done, None, {}

    def reset(self):
        self.deck = self.initialize_deck()
        self.deal_cards()
        self.current_player = self._get_starting_player()
        self.current_play = None
        self.done = False
        return self._get_obs()

    def _get_starting_player(self):
        for i, hand in enumerate(self.player_hands):
            if (1, 1) in hand:
                return i
        return 0

    def _get_obs(self):
        player_hand_numeric = self.player_hands[self.current_player]
        current_play_numeric = self.current_play
        # player_hand_numeric = [self.deck.index(card) for card in self.player_hands[self.current_player]]
        # current_play_numeric = self.deck.index(self.current_play) if self.current_play else -1
        return {
            'player_hand': player_hand_numeric,
            'current_play': current_play_numeric
        }

    def render(self, mode='human'):
        print(f"Current Player: {self.current_player + 1}")
        print(f"Player Hand: {self.player_hands[self.current_player]}")
        print(f"Current Play: {self.current_play}")
        print(f"Passes: {self.player_pass}")


register(
    id='CardGame-v0',
    entry_point='card_game_env:CardGameEnv',  # Ensure this points to the correct path
    max_episode_steps=100,
)
