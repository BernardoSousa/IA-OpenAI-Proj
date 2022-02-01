import gym
from gym.envs.toy_text.frozen_lake import generate_random_map

actions = {
    'Left': 0,
    'Down': 1,
    'Right': 2,
    'Up': 3
}

custom_map = [
    'SFFF',
    'FHFH',
    'FFFH',
    'HFFG',
]

MAX_ITERATIONS = 100

if __name__ == '__main__':
    random_map = generate_random_map(size=10, p=1)
    env = gym.make("FrozenLake-v1", desc=custom_map)
    env.reset()
    print("Action space: ", env.action_space)
    print("Observation space: ", env.observation_space)
    env.render()

    for i in range(MAX_ITERATIONS):
        random_action = env.action_space.sample()
        print("\n\n=========================\n  STEP:", i+1, " ACTION:", random_action, "\n=========================\n")
        new_state, reward, done, info = env.step(
            random_action)
        env.render()
        if done:
            break