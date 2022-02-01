import gym

from qlearning_frozen.epsilon import Epsilon
from qlearning_frozen.qtable import QTable
from qlearning_frozen.qtable import train_qtable
from qlearning_frozen.__constants import *

"""
   Frozen lake involves crossing a frozen lake from Start(S) to goal(G) without falling into any holes(H). The agent
    may not always move in the intended direction due to the slippery nature of the frozen lake.
    
   The agent take a 1-element vector for actions.
   The action space is `(dir)`, where `dir` decides direction to move in which can be:
   - 0: LEFT
   - 1: DOWN
   - 2: RIGHT
   - 3: UP
   
   The observation is a value representing the agents current position as current_row * nrows + current_col
   
   **Rewards:**
   Reward schedule:
   - Reach goal(G): +1
   - Reach hole(H): 0
   
   ### Arguments
   ```
   gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=True)
   ```
   `desc`: Used to specify custom map for frozen lake. For example,
       desc=["SFFF",
             "FHFH", 
             "FFFH", 
             "HFFG"].
             
   `map_name`: ID to use any of the preloaded maps.
       "4x4":[
           "SFFF",
           "FHFH",
           "FFFH",
           "HFFG"
           ]
       "8x8": [
           "SFFFFFFF",
           "FFFFFFFF",
           "FFFHFFFF",
           "FFFFFHFF",
           "FFFHFFFF",
           "FHHFFFHF",
           "FHFFHFHF",
           "FFFHFFFG",
       ]
   `is_slippery`: True/False. If True will move in intended direction with
   probability of 1/3 else will move in either perpendicular direction with
   equal probability of 1/3 in both directions.
       For example, if action is left and is_slippery is True, then:
       - P(move left)=1/3
       - P(move up)=1/3
       - P(move down)=1/3
"""


def main():
    # step 1: loading the environment
    env = gym.make("FrozenLake-v1", desc=MAPS['4x4'])

    # step 2: creating the Q-table
    state_size = env.observation_space.n
    action_size = env.action_space.n
    q = QTable(state_size, action_size)

    # step 3: creating de epsilon decay
    e = Epsilon(initial_epsilon=1.0, max_epsilon=1.0, min_epsilon=0.01, decay_rate=0.00005)

    # step 4: Q-table training
    q, rewards = train_qtable(env, q, e, Q_TOTAL_EPISODES, Q_MAX_STEPS)

    print("Score over time {:.4f}".format(sum(rewards) / Q_TOTAL_EPISODES))
    q.print()

    # Play
    env.reset()
    env.render()
    rewards = []

    for episode in range(ACTION_EPISODES):
        state = env.reset()
        step = 0
        total_rewards = 0

        for step in range(ACTION_STEPS):
            action = q.select_action(env, state)

            # new_state -> the new state of the environment
            # reward -> the reward
            # done -> the a boolean flag indicating if the returned state is a terminal state
            # info -> an object with additional information for debugging purposes
            new_state, reward, done, info = env.step(action)

            total_rewards += reward
            state = new_state

            if done:
                break

        rewards.append(total_rewards)

        if episode % 100 == 0:
            print("=======================================")
            print("EPISODE {}".format(episode))
            print("Number of steps: {}".format(step))
            env.render()

    print("Score over time {:.4f}".format(sum(rewards) / 1000))

    env.close()


if __name__ == "__main__":
    main()
