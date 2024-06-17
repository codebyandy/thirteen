import gym
import card_game_env as card_game_env  
import pdb

# Create the environment
env = gym.make('CardGame-v0')

# Reset the environment
obs = env.reset()

# Sample random actions

while True:
    action = (1, 1)
    if env.current_play:
        action = (0, 0)
        for (rank, suit) in env.player_hands[env.current_player]:
            if rank > env.current_play[0] or (rank == env.current_play[0] and suit > env.current_play[1]):
                action = (rank, suit)
                break
    
    print("Action: ", action)
    print(" ")
    
    obs, reward, done, _, _ = env.step(action)

    if done:
        print("Game Over")
        print(" ")
        break

    env.render()
    

env.close()