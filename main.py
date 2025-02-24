import sys
import os
import random
import numpy as np
import argparse
from typing import Union
import csv

from jsonargparse import CLI

from stable_baselines3 import SAC, TD3, DDPG, PPO, A2C, DQN
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm

from CR_LoFi.atcenv.src.atcenv import AtcEnv
from CR_LoFi.atcenv.src.environment_objects.airspace import Airspace
from CR_LoFi.atcenv.src.environment_objects.flight import Aircraft, Flight
from CR_LoFi.atcenv.src.environment_objects.environment import Environment
from CR_LoFi.atcenv.src.observation.observation import Observation
from CR_LoFi.atcenv.src.reward.reward import Reward
from CR_LoFi.atcenv.src.scenarios.scenario import Scenario
import CR_LoFi.atcenv.src.functions as fn

from ..common.callbacks import CSVLoggerCallback, TimestepStoppingCallback


# Dictionary of all stable-baselines3 algorithms you want to support:
ALGORITHMS = {
    "SAC": SAC,
    "TD3": TD3,
    "DDPG": DDPG,
    "DQN": DQN,
    "PPO": PPO,
    "A2C": A2C
}

# You can track which algos are off-policy for buffer saving/loading logic:
OFF_POLICY = {"SAC", "TD3", "DDPG", "DQN"}
EVAL_EPISODES = 10

def main(experiment_name: str,
         environment: Environment,
         model: dict,
         scenario: Scenario,
         airspace: Airspace,
         aircraft: Aircraft,
         observation: Observation,
         reward: Reward,
         baseline: bool,
         pre_training: Union[str, float],
         window: int,
         algorithm: str = "SAC",
         train: bool = False,
         eval: bool = False,
         seed: int = 42):
    
    # Set seed for reproducibility
    random.seed(seed)
    np.random.seed(seed)

    # # --- Baseline scenario logic: just act with zero actions ---
    # if baseline:
    #     env = AtcEnv(environment=environment, scenario=scenario, airspace=airspace,
    #                  aircraft=aircraft, observation=observation, reward=reward)
    #     rewards = []
    #     for i in range(EVAL_EPISODES):
    #         done = truncated = False
    #         obs, info = env.reset()
    #         total_reward = 0
    #         while not (done or truncated):
    #             action = np.zeros(env.action_space.shape)
    #             obs, rew, done, truncated, info = env.step(action)
    #             total_reward += rew
    #         rewards.append(total_reward)

    #     print(f"Baseline Average Reward: {np.mean(rewards):.3f}")
    #     print(f"Baseline Std Dev: {np.std(rewards):.3f}")
    #     env.close()
    #     return
    
    # --- Prepare logging directory ---
    experiment_folder = f"/scratch/amoec/ATC_RL/LoFi-{algorithm}/{algorithm}_{pre_training}"
    log_dir = f"{experiment_folder}/logs"
    os.makedirs(log_dir, exist_ok=True)
    csv_logger = CSVLoggerCallback(log_dir, 'results.csv')

    # --- Build environment ---
    env = AtcEnv(environment=environment, scenario=scenario, airspace=airspace,
                 aircraft=aircraft, observation=observation, reward=reward,
                 seed=seed)
    
    # Map string name to actual SB3 class:
    if algorithm not in ALGORITHMS:
        raise ValueError(f"Unsupported algorithm: {algorithm}")
    AlgoClass = ALGORITHMS[algorithm]
    
    # Extract policy type from model config:
    policy_type = model.get('policy', {}).get('type')
    # Load all relevant hyperparams from the 'model' dict's 'init_args' key:
    algo_params = model.get('init_args', {})

    # Initialize EarlyStopping callback
    early_stopping = TimestepStoppingCallback(target=pre_training, verbose=1)
    callback_list = CallbackList([early_stopping, csv_logger])
    
    # Set network architecture
    policy_kwargs = dict(
        net_arch=[256, 256] # Default architecture used for all algorithms
    )

    # Create the RL model (default hyperparams):
    model_instance = AlgoClass(policy_type, env, policy_kwargs=policy_kwargs)

    # If user wants training:
    if train:
        model_instance.learn(total_timesteps=int(2e6), progress_bar=False, callback=callback_list)
        model_path = f"{experiment_folder}/model"
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        model_instance.save(model_path)

        # If it's an off-policy algorithm, save the replay buffer as well
        if issubclass(AlgoClass, OffPolicyAlgorithm):
            rb_path = f"{model_path}_buffer"
            model_instance.save_replay_buffer(rb_path)
            print(f"Replay buffer saved to: {rb_path}")

        del model_instance

    env.close()

    # --- TEST PHASE ---
    if eval:
        model_path = f"{experiment_folder}/model"
        model_instance = AlgoClass.load(model_path)

        # Recreate env for evaluation (no seed)
        env = AtcEnv(environment=environment, scenario=scenario, airspace=airspace,
                    aircraft=aircraft, observation=observation, reward=reward)

        # Evaluate for a few episodes
        for i in range(EVAL_EPISODES):
            done = truncated = False
            obs, info = env.reset()
            episode_reward = 0
            while not (done or truncated):
                # Predict in deterministic mode
                action, _ = model_instance.predict(obs, deterministic=True)
                obs, rew, done, truncated, info = env.step(action[()])
                episode_reward += rew
        env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pre_training', default='full',
                        help='If "full", no early stopping. Otherwise specify a float (0-100) to do partial training.')
    parser.add_argument('--window', type=int, required=True,
                        help='Window size for the moving average.')
    parser.add_argument('--algorithm', type=str, default='SAC',
                        help='Supported: SAC, TD3, DDPG, PPO, A2C, DQN, etc.')
    parser.add_argument('--train', action='store_true',
                        help='If set, we train the model; otherwise we only load & test.')
    parser.add_argument('--eval', action='store_true',
                        help='If set, evaluate the model after training.')
    parser.add_argument('--seed', type=int, default=42,
                        help='Set random seed for reproducibility.')
    args, unknown = parser.parse_known_args()

    if args.pre_training == 'full':
        pretraining_val = 'full'
    else:
        pretraining_val = float(args.pre_training)
        
    args_list = [
        '--pre_training', str(pretraining_val),
        '--window', str(args.window),
        '--algorithm', args.algorithm,
        '--seed', str(args.seed)
    ]
    
    if args.train:
        args_list.extend(['--train', 'true'])
    if args.eval:
        args_list.extend(['--eval', 'true'])
    
    # Pass these CLI arguments forward to the JSONArgParse CLI wrapper:
    CLI(main, as_positional=False, args=args_list + unknown)