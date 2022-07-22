from models.agent import Agent

def run():
    agent = Agent(
            buffer_max_size=500,
            gamma=0.99,
            tau=0.001,
            epochs=500,
            env_stochasticity=0.15,
            experiment_description="env 1 oded code version",
            replay_buffer_sampling_percent=0.7,
            min_buffer_size_for_learn=300)
    agent.play()

if __name__ == '__main__':
    run()
