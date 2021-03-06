from models.agent import Agent

def run():
    agent = Agent(
            buffer_max_size=500,
            gamma=0.99,
            tau=0.001,
            epochs=500,
            env_stochasticity=0,
            experiment_description="EX1 - Highway-Env- with icm sanity",
            replay_buffer_sampling_percent=0.7,
            min_buffer_size_for_learn=30,
            use_icm=True,
            config_name='config1',
            config_3_env_type=None)
    agent.play()

if __name__ == '__main__':
    run()
