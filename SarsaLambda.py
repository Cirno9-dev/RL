from env.maze_env import Maze
from RL_brain import SarsaLambdaTable

score = 0


# Sarsa(0)
def update():
    global score

    for episode in range(100):
        # init observation
        observation = env.reset()

        # RL choose action based on observation
        action = RL.choose_action(str(observation))

        while True:
            # fresh env
            env.render()
            # RL take action and get next observation and reward
            observation_, reward, done = env.step(action)
            # RL choose action based on next observation
            action_ = RL.choose_action(str(observation_))
            # RL learns from this transition
            RL.learn(str(observation), action, reward, str(observation_), action_)
            # print(RL.eligibility_trace)
            # swap observation and action
            observation = observation_
            action = action_

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
    # Sarsa(0)
    # RL = SarsaLambdaTable(actions=list(range(env.n_actions)), trace_decay=0)
    # Sarsa(lambda)
    RL = SarsaLambdaTable(actions=list(range(env.n_actions)), e_greedy=0.8, trace_decay=0.9)

    env.after(100, update)
    env.mainloop()
