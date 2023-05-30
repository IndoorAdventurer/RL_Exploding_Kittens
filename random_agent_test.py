from ekenv import EKAgent, EKTrainer, EKRandomAgent
import numpy as np

def get_random_agents() -> list[EKAgent]:
    num_agents = np.random.randint(2, 6)
    agent = EKRandomAgent()
    return [agent] * num_agents

if __name__ == "__main__":
    num_epochs = 100
    num_training_games = 100
    num_testing_games = 10

    trainer = EKTrainer(get_random_agents, get_random_agents)

    for idx in range(num_epochs):
        
        print(f"---EPOCH-{idx}" + "-" * 20)
        
        trainer.training_loop(num_training_games)
        results = trainer.testing_loop(num_testing_games)
        print(results)
        # Do stuff with the results here!