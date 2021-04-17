# Copyright 2020 Tensorforce Team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from tensorforce import Runner, Agent, Environment
from learning_tensorforce.env import EnvNN

import paths
import pickle as pkl
import mlflow


def main():
    with open(paths.datasets_data_path / "_0" / "timeseries.pkl", "rb") as f:
        data = pkl.load(f)

    mlflow.set_tracking_uri(paths.mlflow_path)
    mlflow.set_experiment("Testing")  #
    mlflow.start_run()

    EnvNN.data = data
    environment = Environment.create(environment=EnvNN)

    agent = Agent.create(
        agent='dqn', environment=environment,
        memory=1000, batch_size=32, start_updating=750, exploration=0.01
    )

    # Initialize the runner
    runner = Runner(agent=agent, environment=environment)

    # Train for 200 episodes
    runner.run(num_episodes=50000)
    runner.close()

    agent.close()
    environment.close()

    mlflow.end_run()

    # plus agent.close() and environment.close() if created separately


if __name__ == '__main__':
    main()
