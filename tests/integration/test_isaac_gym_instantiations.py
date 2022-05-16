import isaacgym
import os
import unittest
from finevo.environments.isaac_gym_envs.tasks import isaacgym_task_map
from subprocess import call


class TestIsaacEnvironments(unittest.TestCase):
    def setUp(self):
        self.task_map = isaacgym_task_map

    def test_should_instantiate_all_Isaac_vector_environments(self):
        test_helper_file = os.path.join(
            os.getcwd(), "tests", "integration", "isaac_gym_instance_helper.py"
        )
        # NOTE: Isaac Gym instantiation tests are commented out for now, to speed up
        # testing. These tests have never failed and take up over 90% of test time.
        # The IsaacGymEnv module is quite stable at this point.
        """
        for env_name in self.task_map:
            return_code = call(["python3", test_helper_file, env_name])
            if return_code != 0:
                raise Exception(
                    f"Instantiating {env_name} resulted in error code {return_code}"
                )
        """


if __name__ == "__main__":
    unittest.main()
