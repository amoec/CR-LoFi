import sys
import os
import random
import numpy as np
import argparse
from typing import Union
import csv

from jsonargparse import CLI

from stable_baselines3 import SAC, TD3, DDPG, PPO, A2C, DQN
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm

from CR_LoFi.atcenv.src.atcenv import AtcEnv
from CR_LoFi.atcenv.src.environment_objects.airspace import Airspace
from CR_LoFi.atcenv.src.environment_objects.flight import Aircraft, Flight
from CR_LoFi.atcenv.src.environment_objects.environment import Environment
from CR_LoFi.atcenv.src.observation.observation import Observation
from CR_LoFi.atcenv.src.reward.reward import Reward
from CR_LoFi.atcenv.src.scenarios.scenario import Scenario
import CR_LoFi.atcenv.src.functions as fn


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


class PerformanceStoppingCallback(BaseCallback):
    """
    Stop training if the total episode reward (averaged over a moving window)
    crosses a threshold percentage, based on the min and max of episode rewards
    from a reference CSV log.
    """

    def __init__(self, training_amount, window, algo, verbose=0):
        super(PerformanceStoppingCallback, self).__init__(verbose)
        self.training_amount = float(training_amount) if training_amount != 'full' else 'full'
        self.window = int(window)
        self.verbose = verbose

        if self.training_amount != 'full':
            # Load only the 'total_reward' column (index 2) from CSV
            self.reward_data_full = np.loadtxt(
                f'/scratch/amoec/ATC_RL/LoFi-{algo}/{algo}_full/logs/results.csv',
                delimiter=',',
                skiprows=1,  # skip header
                usecols=2    # column index for total_reward
            )
            self.len_full = len(self.reward_data_full)
            # min_loss: average of last 5% of the full training logs
            self.min_loss = np.mean(np.abs(self.reward_data_full[-int(self.len_full * 0.05):]))
            # max_loss: average of first 5% of the full training logs
            self.max_loss = np.mean(np.abs(self.reward_data_full[:int(self.len_full * 0.05)]))
        else:
            self.min_loss = None
            self.max_loss = None
            
        self.check = max(self.window, self.len_full * 0.05)
        # List to store final episode rewards
        self.reward_history = []
        # Accumulator for the current episode’s rewards
        self.episode_reward = 0.0

    def _on_step(self) -> bool:
        """
        Called at every environment step. We add the step’s reward to
        self.episode_reward. Once the environment is 'done', we record
        the total for that episode and do our early-stopping check.
        """
        if self.training_amount == 'full':
            # User wants full training with no early stop
            return True

        # 'rewards' is a single-element list when n_envs = 1
        rewards = self.locals.get("rewards", [0.0])
        dones = self.locals.get("dones", [False])

        # Since n_envs=1, just grab index 0
        reward = rewards[0]
        done = dones[0]

        # Accumulate the reward
        self.episode_reward += reward

        # If the episode ended
        if done:
            # Record final episode reward
            self.reward_history.append(self.episode_reward)
            # Reset for the next episode
            self.episode_reward = 0.0

            # Perform early-stopping checks if we have enough episodes
            if len(self.reward_history) > self.check:
                moving_avg = np.mean(self.reward_history[-self.check:])

                # Make sure we have valid min_loss/max_loss and a non-zero range
                if (self.min_loss is not None) and (self.max_loss is not None):
                    denom = self.max_loss - self.min_loss
                    if denom != 0:
                        pct_train = np.round((self.max_loss - abs(moving_avg)) / denom * 100, 2)

                        if self.verbose:
                            print(
                                f"Moving Avg of last {self.window} episodes: {moving_avg:.2f}, "
                                f"Min: {self.min_loss:.2f}, Max: {self.max_loss:.2f}, "
                                f"pct_train: {pct_train:.2f}%"
                            )

                        # Check threshold
                        if pct_train >= self.training_amount and abs(moving_avg) < self.max_loss:
                            if self.verbose:
                                print(f"Early stopping triggered at {pct_train:.2f}%")
                            return False

        # If we didn't stop, continue
        return True
    
class TimestepStoppingCallback(BaseCallback):
    """
    Stop training after a certain percentage of total timesteps have been reached."""
    def __init__(self, target, total_timesteps=2e6, verbose=0):
        super(TimestepStoppingCallback, self).__init__(verbose)
        self.total_timesteps = total_timesteps
        self.timesteps = 0
        self.target = target
        
    def _on_step(self) -> bool:
        if self.target == 'full':
            # User wants full training with no early stop
            return True
        
        self.timesteps += 1
        pct = self.timesteps / self.total_timesteps * 100
        
        if pct >= self.target:
            return False
        
        return True

class CSVLoggerCallback(BaseCallback):
    def __init__(self, log_dir, filename, verbose=0):
        super().__init__(verbose)
        os.makedirs(log_dir, exist_ok=True)
        self.log_path = os.path.join(log_dir, filename)
        self.headers = [
            "timesteps",
            "episode",
            "reward"
        ]
        self.current_episode = 0
        self.buffer = []  # buffer for metrics
        self.flush_frequency = 1000  # flush every 1000 episodes
        self._write_header()
    
    def _write_header(self):
        if not os.path.exists(self.log_path) or os.path.getsize(self.log_path) == 0:
            with open(self.log_path, "w") as f:
                writer = csv.writer(f)
                writer.writerow(self.headers)
        
    def _flush_buffer(self):
        if self.buffer:
            with open(self.log_path, "a") as f:
                writer = csv.writer(f)
                for metrics in self.buffer:
                    writer.writerow([metrics[h] for h in self.headers])
            self.buffer = []
    
    def _on_step(self) -> bool:
        step_reward = self.locals.get("rewards", [0])[0]
        done = self.locals.get("dones", [False])[0]
        
        # Only increment episode count when an episode is done
        if done:
            self.current_episode += 1
        
        metrics = {
            "timesteps": self.num_timesteps,
            "episode": self.current_episode,
            "reward": step_reward
        }
        
        # Store metric in buffer
        self.buffer.append(metrics)
        
        # Flush buffer every `flush_frequency` episodes or if done on final steps (optional)
        if done and self.current_episode % self.flush_frequency == 0:
            self._flush_buffer()
                    
        return True
    
    def _on_training_end(self):
        # Flush buffer one last time
        self._flush_buffer()

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

    # --- Baseline scenario logic: just act with zero actions ---
    if baseline:
        env = AtcEnv(environment=environment, scenario=scenario, airspace=airspace,
                     aircraft=aircraft, observation=observation, reward=reward)
        rewards = []
        for i in range(EVAL_EPISODES):
            done = truncated = False
            obs, info = env.reset()
            total_reward = 0
            while not (done or truncated):
                action = np.zeros(env.action_space.shape)
                obs, rew, done, truncated, info = env.step(action)
                total_reward += rew
            rewards.append(total_reward)

        print(f"Baseline Average Reward: {np.mean(rewards):.3f}")
        print(f"Baseline Std Dev: {np.std(rewards):.3f}")
        env.close()
        return
    
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