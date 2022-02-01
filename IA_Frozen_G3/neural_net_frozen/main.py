import gym
import numpy as np
from keras.models import Sequential
from keras.layers import InputLayer
from keras.layers import Dense

from neural_net_frozen.__constants import *
# from example_agent_search_support import *


def main():
    # ag = Agent()

    # step 1: loading the environment
    env = gym.make('FrozenLake-v1')

    # step 2: define the algorithm variables
    discount_factor = 0.95
    eps = 0.5
    eps_decay_factor = 0.999

    # step 3: create the neural network model
    model = Sequential()
    model.add(InputLayer(batch_input_shape=(1, env.observation_space.n)))
    model.add(Dense(env.action_space.n, activation='relu'))
    # model.add(Dense(env.action_space.n, activation='linear'))
    model.compile(loss='mse', optimizer='adam', metrics=['mae'])

    print("Action space: ", env.action_space)
    print("Observation space: ", env.observation_space)
    for i in range(MAX_LEARNING_EPISODES):
        # ag.frozen_lake_reset()
        print("\n===========================\nEPISODE:", i, "\n===========================")

        state = env.reset()
        eps *= eps_decay_factor
        done = False
        while not done:
            """
            state is a number between 0 and 16 (observation_space.n)
            np.identity returns the 16th length vector with the value one corresponding to the state and the rest of
            the values 0. So it activates the neuron that corresponds to the state.
            It return the max of the values returned. The values are positive and negative (confirm)
            The value returned (argmax) corresponds to the action selected by the neural network.
            """
            if np.random.random() < eps:
                action = np.random.randint(0, env.action_space.n)
            else:
                action = np.argmax(model.predict(np.identity(env.observation_space.n)[state:state + 1]))

            # new_state -> the new state of the environment
            # reward -> the reward
            # done -> the a boolean flag indicating if the returned state is a terminal state
            # _ -> an object with additional information for debugging purposes
            new_state, reward, done, _ = env.step(action)
            target = reward + discount_factor * np.max(
                model.predict(np.identity(env.observation_space.n)[new_state:new_state + 1]))
            target_vector = model.predict(
                np.identity(env.observation_space.n)[state:state + 1])[0]
            target_vector[action] = target
            model.fit(
                np.identity(env.observation_space.n)[state:state + 1],
                target_vector.reshape(-1, env.action_space.n),
                epochs=1, verbose=0)
            state = new_state
            print(state)
            # ag.frozen_lake_integration(action)
            env.render()

    input("Press a key to test learned model.")

    # Test the model
    # Back to basis ...
    state = env.reset()
    for i in range(ACTION_EPISODES):
        action = np.argmax(model.predict(np.identity(env.observation_space.n)[state:state + 1]))
        # Esta ação retorna o novo estado, recompensa e o episódio onde terminou e a informação adicional acerca do que
        # se passa no mundo.
        new_state, reward, done, info = env.step(action)
        # print('New State:', new_state)
        # print('Reward:', reward)
        # print('Done:', done)
        # print('Info:', info)
        # ag.frozen_lake_integration(action)
        env.render()
        state = new_state

    input('End')


if __name__ == '__main__':
    main()
