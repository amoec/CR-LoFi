import sys
import os
import random
import numpy as np
import argparse
from typing import Union
import datetime

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

from ..common.callbacks import SafeLogCallback, TimestepStoppingCallback


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
         timeout: datetime.datetime = None,
         train: bool = False,
         eval: bool = False,
         seed: int = 42,
         restart: bool = False):
    
    # Set seed for reproducibility
    random.seed(seed)
    np.random.seed(seed)

    # Determine experiment folder based on baseline flag
    if baseline:
        experiment_folder = f"/scratch/amoec/ATC_RL/baseline/LoFi-{algorithm}/run_{seed}"
    else:
        experiment_folder = f"/scratch/amoec/ATC_RL/LoFi-{algorithm}/{algorithm}_{pre_training}"

    log_dir = f"{experiment_folder}/logs"
    os.makedirs(log_dir, exist_ok=True)
    csv_logger = SafeLogCallback(
        model_path=f"{experiment_folder}/model",
        log_dir=log_dir,
        log_filename="results.csv",
        timeout=timeout,
        save_buffer=True
    )

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
        net_arch=[256, 256] # Default architecture to be used for all algorithms
    )

    # Create or load the RL model based on restart flag
    model_path = f"{experiment_folder}/model"
    
    if restart and os.path.exists(model_path):
        print(f"Restarting training from checkpoint: {model_path}")
        model_instance = AlgoClass.load(model_path, env=env)
        
        # If it's an off-policy algorithm, try to load the replay buffer
        if issubclass(AlgoClass, OffPolicyAlgorithm):
            rb_path = f"{model_path}_buffer"
            if os.path.exists(rb_path):
                model_instance.load_replay_buffer(rb_path)
                print(f"Loaded replay buffer from: {rb_path}")
    else:
        # Create a new model instance
        model_instance = AlgoClass(policy_type, env, policy_kwargs=policy_kwargs)

    # If user wants training:
    if train:
        model_instance.learn(total_timesteps=int(2e6), progress_bar=False, callback=callback_list)
        
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        model_instance.save(model_path)

        # If it's an off-policy algorithm, save the replay buffer as well
        if issubclass(AlgoClass, OffPolicyAlgorithm):
            rb_path = f"{model_path}_buffer"
            model_instance.save_replay_buffer(rb_path)
            print(f"Replay buffer saved to: {rb_path}")

        del model_instance

    env.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pre_training', default='full',
                        help='If "full", no early stopping. Otherwise specify a float (0-100) to do partial training.')
    parser.add_argument('--window', type=int, required=True,
                        help='Window size for the moving average.')
    parser.add_argument('--algorithm', type=str, default='SAC',
                        help='Supported: SAC, TD3, DDPG, PPO, A2C, DQN, etc.')
    parser.add_argument('--timeout', type=datetime.datetime, required=True,
                        help='Timeout datetime for the job.')
    parser.add_argument('--train', action='store_true',
                        help='If set, we train the model; otherwise we only load & test.')
    parser.add_argument('--eval', action='store_true',
                        help='If set, evaluate the model after training.')
    parser.add_argument('--seed', type=int, default=42,
                        help='Set random seed for reproducibility.')
    parser.add_argument('--baseline', action='store_true',
                        help='Flag to indicate baseline full training runs.')
    parser.add_argument('--restart', type=str, default='False',
                        help='Flag to indicate restarting from a checkpoint.')
    args, unknown = parser.parse_known_args()

    if args.pre_training == 'full':
        pretraining_val = 'full'
    else:
        pretraining_val = float(args.pre_training)
    
    restart = args.restart.lower() == 'true'
    
    args_list = [
        '--pre_training', str(pretraining_val),
        '--window', str(args.window),
        '--algorithm', args.algorithm,
        '--seed', str(args.seed),
        '--restart', str(restart)
    ]
    
    if args.train:
        args_list.extend(['--train', 'true'])
    if args.eval:
        args_list.extend(['--eval', 'true'])
    if args.baseline:
        args_list.extend(['--baseline', 'true'])
    
    # Pass these CLI arguments forward to the JSONArgParse CLI wrapper:
    CLI(main, as_positional=False, args=args_list + unknown)