import sys
from finevol.environments.isaac_gym_env import IsaacGymEnv


def create_isaac_vec_environment(env_name: str):
    isaac_env = IsaacGymEnv(env_name, headless=True)
    del isaac_env


if __name__ == "__main__":
    env_name = sys.argv[1]
    create_isaac_vec_environment(env_name)
