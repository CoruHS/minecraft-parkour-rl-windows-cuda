import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import csv
import sys
import subprocess
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PARKOUR_ENV_PATH = PROJECT_ROOT / "python" / "ParkourEnv"
sys.path.insert(0, str(PARKOUR_ENV_PATH))

from env_cnn import (
    ParkourRL,
    RUNCLIENT_LOG_PATH,
    _can_connect_to_minecraft,
    start_minecraft_client,
    wait_for_minecraft_socket,
)
from model import ParkourCNN

CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints_cnn"
EPISODE_LOG_PATH = PROJECT_ROOT / "training_logs" / "episodes_cnn.csv"
EPISODE_CHECKPOINT_INTERVAL = 50
STEP_CHECKPOINT_INTERVAL = 50_000

class RolloutBuffer:
    def __init__(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []

    def add(self, state, action, log_prob, reward, value, done):
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)

    def clear(self):
        self.__init__()

    def __len__(self):
        return len(self.states)


def copy_obs(obs):
    return {
        "frame": obs["frame"].copy(),
        "mlp": obs["mlp"].copy(),
    }


def compute_gae(rewards, values, dones, gamma = 0.99, lam = 0.95):
    advantages = []
    gae = 0

    for i in reversed(range(len(rewards))):
        if i == len(rewards) - 1:
            next_value = 0.0
        else:
            next_value = values[i+1]

        delta = rewards[i] + gamma * next_value * (1-dones[i]) - values[i]
        gae = delta + gamma * lam * (1-dones[i]) * gae
        advantages.insert(0,gae)

    advantages = torch.FloatTensor(advantages)
    returns = advantages + torch.FloatTensor(values)
    return advantages, returns

def ppo_update(model, optimizer, buffer, device, epochs = 4, clip_epsilon= 0.2, batch_size = 64):
    if len(buffer) == 0:
        return

    frames = torch.as_tensor(np.stack([state["frame"] for state in buffer.states]), dtype=torch.float32, device=device)
    mlp_states = torch.as_tensor(np.stack([state["mlp"] for state in buffer.states]), dtype=torch.float32, device=device)
    actions = torch.as_tensor(buffer.actions, dtype=torch.long, device=device)
    old_log_probs = torch.stack(buffer.log_probs).detach().to(device)
    values = [v.item() for v in buffer.values]

    advantages, returns = compute_gae(buffer.rewards, values, buffer.dones)
    advantages = advantages.to(device)
    returns = returns.to(device)

    # normalize advantages (helps stability)
    advantages = (advantages - advantages.mean())/(advantages.std(unbiased=False)+1e-8)

    #run multiple passes over the same data

    for _ in range(epochs):
        indices = np.arange(len(buffer))
        np.random.shuffle(indices)

        for start in range(0,len(buffer),batch_size):
            end = start + batch_size
            batch_idx = torch.as_tensor(indices[start:end], dtype=torch.long, device=device)

            batch_frames = frames[batch_idx]
            batch_mlp_states = mlp_states[batch_idx]
            batch_actions = actions[batch_idx]
            batch_old_log_probs = old_log_probs[batch_idx]
            batch_advantages = advantages[batch_idx]
            batch_returns = returns[batch_idx]

            # current predictions
            logits, values_pred = model(batch_frames, batch_mlp_states)
            dist = torch.distributions.Categorical(logits=logits)
            new_log_probs = dist.log_prob(batch_actions)
            entropy = dist.entropy().mean()

            #PPO clipped objective

            ratio = torch.exp(new_log_probs - batch_old_log_probs)
            clipped_ratio = torch.clamp(ratio, 1-clip_epsilon, 1+clip_epsilon)
            actor_loss = -torch.min(ratio * batch_advantages, clipped_ratio * batch_advantages).mean()

            critic_loss = nn.MSELoss()(values_pred.squeeze(-1), batch_returns)

            loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),0.5)
            optimizer.step()


def save_checkpoint(model, filename):
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_path = CHECKPOINT_DIR / filename
    torch.save(model.state_dict(), checkpoint_path)
    torch.save(model.state_dict(), CHECKPOINT_DIR / "latest.pt")
    return checkpoint_path


def format_duration(seconds):
    if seconds is None:
        return "unknown"

    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)

    if hours > 0:
        return f"{hours}h {minutes}m {seconds}s"
    if minutes > 0:
        return f"{minutes}m {seconds}s"
    return f"{seconds}s"


def get_timing(start_time, steps_done, total_timesteps):
    elapsed_seconds = time.time() - start_time
    steps_per_second = steps_done / elapsed_seconds if elapsed_seconds > 0 else 0.0
    remaining_steps = max(0, total_timesteps - steps_done)
    eta_seconds = remaining_steps / steps_per_second if steps_per_second > 0 else None
    return elapsed_seconds, steps_per_second, eta_seconds


def log_episode(
    episode,
    steps_done,
    episode_steps,
    episode_reward,
    reason,
    elapsed_seconds,
    steps_per_second,
    eta_seconds,
):
    EPISODE_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    write_header = not EPISODE_LOG_PATH.exists()

    with EPISODE_LOG_PATH.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "episode",
                "steps_done",
                "episode_steps",
                "episode_reward",
                "reason",
                "elapsed_seconds",
                "steps_per_second",
                "eta_seconds",
            ],
        )
        if write_header:
            writer.writeheader()
        writer.writerow(
            {
                "episode": episode,
                "steps_done": steps_done,
                "episode_steps": episode_steps,
                "episode_reward": episode_reward,
                "reason": reason,
                "elapsed_seconds": elapsed_seconds,
                "steps_per_second": steps_per_second,
                "eta_seconds": eta_seconds,
            }
        )


def train():
    env = None
    process = None
    log_file = None

    try:
        if _can_connect_to_minecraft(timeout=0.5):
            print("Existing Minecraft RL socket detected; using the running client.")
        else:
            process, log_file = start_minecraft_client()
            print(f"Started Minecraft with ./gradlew runClient. Log: {RUNCLIENT_LOG_PATH}")
            print("Waiting for Minecraft socket...")
            wait_for_minecraft_socket(process)

        env = ParkourRL() # no need to add any parameters.
        num_actions = len(env.action_table)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Training on device: {device}"
              + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))

        model = ParkourCNN(num_actions=num_actions, stack_size=env.stack_size).to(device)
        optimizer = optim.AdamW(model.parameters(),lr = 0.0003, weight_decay =0.01)
        buffer = RolloutBuffer()

        rollout_steps = 2048
        total_timesteps = 200_000
        steps_done = 0
        episode = 0
        episode_reward = 0.0
        episode_steps = 0
        last_checkpoint_episode = 0
        last_step_checkpoint = 0
        training_start_time = time.time()

        obs, info = env.reset()

        while steps_done < total_timesteps:
            # ---- collect experience ----
            for _ in range(rollout_steps):
                action, log_prob, value = model.get_action(obs)
                next_obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                buffer.add(copy_obs(obs), action, log_prob, reward, value, float(done))

                obs = next_obs
                episode_reward += reward
                episode_steps += 1
                steps_done += 1

                if done:
                    episode += 1
                    reason = "terminated" if terminated else "truncated"
                    elapsed_seconds, steps_per_second, eta_seconds = get_timing(
                        training_start_time,
                        steps_done,
                        total_timesteps,
                    )
                    log_episode(
                        episode,
                        steps_done,
                        episode_steps,
                        episode_reward,
                        reason,
                        elapsed_seconds,
                        steps_per_second,
                        eta_seconds,
                    )
                    print(
                        f"Episode {episode}  reward={episode_reward:.2f}  "
                        f"episode_steps={episode_steps}  total_steps={steps_done}  "
                        f"elapsed={format_duration(elapsed_seconds)}  "
                        f"speed={steps_per_second:.2f} steps/s  "
                        f"eta={format_duration(eta_seconds)}"
                    )
                    episode_reward = 0.0
                    episode_steps = 0
                    obs, info = env.reset()

                    if (
                        episode % EPISODE_CHECKPOINT_INTERVAL == 0
                        and episode != last_checkpoint_episode
                    ):
                        path = save_checkpoint(model, f"checkpoint_episode_{episode}.pt")
                        print(f"Saved episode checkpoint: {path}")
                        last_checkpoint_episode = episode

                if steps_done >= total_timesteps:
                    break

            # ---- learn from collected experience ----
            ppo_update(model, optimizer, buffer, device)
            buffer.clear()
            elapsed_seconds, steps_per_second, eta_seconds = get_timing(
                training_start_time,
                steps_done,
                total_timesteps,
            )
            print(
                f"Progress {steps_done}/{total_timesteps}  "
                f"elapsed={format_duration(elapsed_seconds)}  "
                f"speed={steps_per_second:.2f} steps/s  "
                f"eta={format_duration(eta_seconds)}"
            )

            if steps_done - last_step_checkpoint >= STEP_CHECKPOINT_INTERVAL:
                path = save_checkpoint(model, f"checkpoint_step_{steps_done}.pt")
                print(f"Saved timestep checkpoint: {path}")
                last_step_checkpoint = steps_done
    finally:
        if env is not None:
            env.close()
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
        if log_file is not None:
            log_file.close()

if __name__ == "__main__":
    train()
