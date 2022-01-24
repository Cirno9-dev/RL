from env.maze_env import Maze
from RL_brain import QLearningTable

score = 0


def update():
    global score

    for episode in range(100):
        # init observation
        observation = env.reset()

        while True:
            # fresh env
            env.render()
            # RL choose action based on observation
            action = RL.choose_action(str(observation))
            # RL take action and get next observation and reward
            observation_, reward, done = env.step(action)
            # RL learns from this transition
            RL.learn(str(observation), action, reward, str(observation_))
            # swap observation
            observation = observation_

            # break while loop when end this episode
            if done:
                if reward > 0:
                    score += 1
                break

    # end of game
    print("game over")
    print("score:", score)
    print(RL.q_table)
    env.destroy()


if __name__ == "__main__":
    env = Maze()
    RL = QLearningTable(actions=list(range(env.n_actions)))

    env.after(100, update)
    env.mainloop()
