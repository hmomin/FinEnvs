import isaacgym 
from TD3_agent import TD3AgentMLP
from finevo.environments.isaac_gym_env import IsaacGymEnv
from finevo.environments.isaac_gym_envs.utils.config_utils import (
    get_isaac_gym_env_args
)

def train_TD3_MLP_on_environiment(env_name:str):
    env_args = get_isaac_gym_env_args(env_name)
    num_envs = env_args["num_envs"]
    batch_size = num_envs*16
    max_samples = 100_000_000

    env = IsaacGymEnv(env_name, num_envs, headless=True)
    agent = TD3AgentMLP(env_args, hidden_dims=(256, 128, 64), write_to_csv=True)
    states = env.reset()
    total_samples = 0
    while total_samples < max_samples:
        (actions,  state_action_values_1, state_action_values_2) = agent.step(states)
        (next_states,rewards,dones,_) = env.step(actions)
        (target_actions,state_action_values_target_1,state_action_values_target_2) = \
            agent.target_networks_step(next_states)
        agent.store(states,actions,rewards,dones,next_states,target_actions,
            state_action_values_1,state_action_values_2,state_action_values_target_1,
            state_action_values_target_2)
        states = next_states
        if agent.get_buffer_size() >=batch_size:
            total_samples = agent.train(states)
        
if __name__ == "__main__":
    train_TD3_MLP_on_environiment("Cartpole")