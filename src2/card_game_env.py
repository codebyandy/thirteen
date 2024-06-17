import gym
from gym import spaces
import numpy as np
import random
from gym.envs.registration import register
import random
import pdb

class CardGameEnv(gym.Env):
    """Custom Environment for a 13-card game"""

    def __init__(self):
        super(CardGameEnv, self).__init__()
        self.deck = self.initialize_deck() # ordered deck
        self.shuffled_deck = self.deck

        self.player_hands = [[], [], [], []]
        self.player_passes = np.array([0, 0, 0, 0])
        self.won_round_must_play = False
        self.current_player = 0
        self.agent_player_index = 0

        self.current_play = None
        self.previous_cards = [] # cards already playe
        
        # self.done = False

        self.action_space = spaces.Discrete(53) # singles only: 52 cards + pass
        self.observation_space = spaces.Discrete(52 * 4) # 1-hot encoding ea. card (agent, other, play, prev)


    def initialize_deck(self):
        ranks = ['3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A', '2']
        suits = ['Spades', 'Clubs', 'Diamonds', 'Hearts']
        return [f"{rank} of {suit}" for suit in suits for rank in ranks]


    def deal_cards(self):
        random.shuffle(self.shuffled_deck)
        self.player_hands = [self.shuffled_deck[i*13:(i+1)*13] for i in range(4)]
        self.player_hands = [sorted(hand, key=lambda x: (x[0], x[1])) for hand in self.player_hands]
    

    def step(self, action):
        # print("current player: ", self.current_player, " | action: ", action)
        self.won_round_must_play = False
        winner = self.current_player
        
        # action = 1:52 -> PLAY
        if action != 0:  
            card_numeric = action - 1
            card = self.deck[card_numeric]
            if self.current_play:
                self.previous_cards.append(self.current_play) # current card -> previous cards pile
            self.player_hands[self.current_player].remove(card)
            self.current_play = card # update current card

            # next player, skip passes
            self.current_player = (self.current_player + 1) % 4
            while self.player_passes[self.current_player]:
                self.current_player = (self.current_player + 1) % 4

        # action = 0 -> PASS
        else: 
            self.player_passes[self.current_player] = 1
            if np.sum(self.player_passes) == 3:
                self.current_player = np.where(self.player_passes == 0)[0][0]
                self.player_passes = np.array([0, 0, 0, 0])
                self.current_play = None
                self.won_round_must_play = True
    
            else:
                # next player, skip passes
                self.current_player = (self.current_player + 1) % 4
                while self.player_passes[self.current_player]:
                    self.current_player = (self.current_player + 1) % 4

        # pdb.set_trace()
        done = any(len(hand) == 0 for hand in self.player_hands)
        # print([len(hand) for hand in self.player_hands])
        # if done:
        #     print("winner: ", winner)

            # print(f"{self.current_player} won!")
        reward = 1 if done else 0
        # reward = 1 if done and len(self.player_hands[self.current_player]) == 0 else 0
        # if reward:
        #     print(f"{self.current_player} won!")

        obs = self._get_obs()
        # print(self.current_player)
        # print(action)
        # print(obs.sum())
        # if obs.sum() != 52:
        #     pdb.set_trace()
        return obs, reward, done, None, {}


    def reset(self):
        # print("reset")
        self.deck = self.initialize_deck()
        self.deal_cards()
        self.current_player = self._get_starting_player()
        self.current_play = None
        self.previous_cards = []
        # self.done = False
        return self._get_obs()


    def _get_starting_player(self):
        for i, hand in enumerate(self.player_hands):
            if "3 of Spades" in hand:
                return i
        return 0
    
    
    def _get_obs(self): 
        observation_matrix = np.zeros((52, 4))
        
        player_hand_numeric = np.array([self.deck.index(card) for card in self.player_hands[self.current_player]]).astype('int')
        observation_matrix[player_hand_numeric, 0] = 1 # cards in player's hand
        # others_hand_numeric = np.logical_not(np.isin(np.arange(observation_matrix.shape[0]), player_hand_numeric))
        # observation_matrix[others_hand_numeric, 1] = 1 # cards in other player's hands
        # print("     player hand: ", observation_matrix.sum())
    
        if self.current_play:
            observation_matrix[self.deck.index(self.current_play), 2] = 1 # card currently in play
        # print("     current play: ", observation_matrix.sum())
        
        previous_cards_numeric = np.array([self.deck.index(card) for card in self.previous_cards]).astype('int')
        observation_matrix[previous_cards_numeric, 3] = 1 # cards already played
        # print("     previous plays: ", observation_matrix.sum())

        others_hand_numeric = np.where(observation_matrix.sum(axis=1) == 0)[0].astype('int')
        observation_matrix[others_hand_numeric, 1] = 1
        # print("     other hands: ", observation_matrix.sum())

        if observation_matrix.sum() != 52:
            pdb.set_trace()
        
        return observation_matrix
      

    def render(self, mode='human'):
        print(" ")
        print(f"Current Player: {self.current_player + 1}")
        print(f"Player Hand: {self.player_hands[self.current_player]}")
        print(f"Current Play: {self.current_play}")


    def seed(seed):
        random.seed(seed)


register(
    id='CardGame-v0',
    entry_point='card_game_env:CardGameEnv',  # Ensure this points to the correct path
    max_episode_steps=100,
)
