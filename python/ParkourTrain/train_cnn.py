import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import csv
import json
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
# Each training invocation gets its own log directory so CSVs from different runs
# never get appended together (the old single-file layout silently concatenated runs).
RUN_ID = time.strftime("%Y%m%d_%H%M%S")
RUN_LOG_DIR = PROJECT_ROOT / "training_logs" / f"run_{RUN_ID}"
EPISODE_LOG_PATH = RUN_LOG_DIR / "episodes_cnn.csv"
EVAL_LOG_PATH = RUN_LOG_DIR / "eval_cnn.csv"
EPISODE_CHECKPOINT_INTERVAL = 50
STEP_CHECKPOINT_INTERVAL = 50_000
EVAL_INTERVAL_STEPS = 10_000
EVAL_EPISODES = 5
BEST_MEAN_CHECKPOINT = "best_mean.pt"
BEST_WORST_CHECKPOINT = "best_worst.pt"

# PPO entropy bonus coefficient. Lower = sharper / less exploratory policy.
# 0.001 let the policy collapse to top_prob~0.87 and argmax to a single-action sequence;
# 0.01 keeps enough spread that argmax tracks the actual best action per state.
ENTROPY_COEF = 0.01
# Camera curriculum: mask yaw/pitch actions during the first N training steps so the agent
# learns clean forward/jump movement before camera control enters the action space.
# Set to 0 to disable. Eval is NEVER masked.
MASK_CAMERA_ACTIONS_INITIAL_STEPS = 50_000
CAMERA_ACTION_IDS = {14, 15, 16, 17, 18, 19}
# Finite (not -inf) so masked actions get ~0 probability without producing NaN entropy.
MASKED_LOGIT_VALUE = -1e9
# Flat reward penalty per camera (yaw/pitch) action, applied by the env. 0.0 disables it.
# (No-op when using the MINIMAL_ACTION_TABLE which has no camera actions.)
CAMERA_ACTION_PENALTY = 0.02
# Per-platform landing bonus. Gives PPO a clean discrete signal each time the agent
# lands on a platform farther along the course than ever before this episode.
# Strong enough to dominate noise but small relative to the goal bonus (+50).
PLATFORM_REWARD = 2.0
PLATFORM_Z_STEP = 1.0

class RolloutBuffer:
    def __init__(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []
        self.masked = []

    def add(self, state, action, log_prob, reward, value, done, masked=False):
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)
        self.masked.append(masked)

    def clear(self):
        self.__init__()

    def __len__(self):
        return len(self.states)


def copy_obs(obs):
    return {
        "frame": obs["frame"].copy(),
        "mlp": obs["mlp"].copy(),
    }


def eval_diagnostics(model, obs):
    """Greedy action for one observation plus diagnostics: (action, top_prob, entropy)."""
    with torch.no_grad():
        device = next(model.parameters()).device
        frame = torch.as_tensor(obs["frame"], dtype=torch.float32, device=device).unsqueeze(0)
        mlp = torch.as_tensor(obs["mlp"], dtype=torch.float32, device=device).unsqueeze(0)
        logits, _ = model(frame, mlp)
        probs = torch.softmax(logits, dim=-1)
        action = int(torch.argmax(logits, dim=-1).item())
        top_prob = float(probs.max().item())
        entropy = float(torch.distributions.Categorical(probs=probs).entropy().item())
    return action, top_prob, entropy


def choose_action_deterministic(model, obs):
    action, _, _ = eval_diagnostics(model, obs)
    return action


def format_action_counts(action_counts):
    total = sum(action_counts) or 1
    parts = [
        f"{action_id}={count} ({100.0 * count / total:.0f}%)"
        for action_id, count in enumerate(action_counts)
        if count > 0
    ]
    return ", ".join(parts) if parts else "(none)"


def choose_action_sampled_masked(model, obs, masked_action_ids):
    """Sample a training action with the given action IDs masked out.

    Returns (action, log_prob, value) under the masked distribution, so the stored
    log_prob matches the policy that actually acted. Training-only; eval is never masked.
    """
    with torch.no_grad():
        device = next(model.parameters()).device
        frame = torch.as_tensor(obs["frame"], dtype=torch.float32, device=device).unsqueeze(0)
        mlp = torch.as_tensor(obs["mlp"], dtype=torch.float32, device=device).unsqueeze(0)
        logits, value = model(frame, mlp)
        if masked_action_ids:
            cols = torch.as_tensor(sorted(masked_action_ids), dtype=torch.long, device=device)
            logits[:, cols] = MASKED_LOGIT_VALUE
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action).squeeze()
    return action.item(), log_prob, value.squeeze()


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
    # Per-step flag: was the camera-action mask active when this transition was collected?
    # Masked transitions must be re-masked below so the PPO ratio/entropy match the policy
    # that actually acted (otherwise the entropy bonus inflates the masked actions).
    masked_flags = torch.as_tensor(buffer.masked, dtype=torch.bool, device=device)
    camera_cols = torch.as_tensor(sorted(CAMERA_ACTION_IDS), dtype=torch.long, device=device)

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
            batch_masked = masked_flags[batch_idx]

            # current predictions
            logits, values_pred = model(batch_frames, batch_mlp_states)
            # Re-apply the camera mask for transitions collected under the curriculum mask,
            # so recomputed log-probs/entropy come from the same distribution that acted.
            if batch_masked.any():
                logits = logits.clone()
                rows = batch_masked.nonzero(as_tuple=True)[0]
                logits[rows.unsqueeze(1), camera_cols.unsqueeze(0)] = MASKED_LOGIT_VALUE
            dist = torch.distributions.Categorical(logits=logits)
            new_log_probs = dist.log_prob(batch_actions)
            entropy = dist.entropy().mean()

            #PPO clipped objective

            ratio = torch.exp(new_log_probs - batch_old_log_probs)
            clipped_ratio = torch.clamp(ratio, 1-clip_epsilon, 1+clip_epsilon)
            actor_loss = -torch.min(ratio * batch_advantages, clipped_ratio * batch_advantages).mean()

            critic_loss = nn.MSELoss()(values_pred.squeeze(-1), batch_returns)

            loss = actor_loss + 0.5 * critic_loss - ENTROPY_COEF * entropy

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),0.5)
            optimizer.step()


def save_checkpoint(model, filename, update_latest=True):
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_path = CHECKPOINT_DIR / filename
    torch.save(model.state_dict(), checkpoint_path)
    if update_latest:
        torch.save(model.state_dict(), CHECKPOINT_DIR / "latest.pt")
    return checkpoint_path


def evaluate_policy(model, env, episodes=EVAL_EPISODES):
    num_actions = len(env.action_table)
    rewards = []
    action_counts = [0] * num_actions
    top_probs = []
    entropies = []
    was_training = model.training
    model.eval()

    try:
        for _ in range(episodes):
            obs, info = env.reset()
            episode_reward = 0.0

            while True:
                action, top_prob, entropy = eval_diagnostics(model, obs)
                action_counts[action] += 1
                top_probs.append(top_prob)
                entropies.append(entropy)
                obs, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward
                if terminated or truncated:
                    break

            rewards.append(episode_reward)
    finally:
        if was_training:
            model.train()

    return {
        "mean": float(np.mean(rewards)),
        "worst": float(np.min(rewards)),
        "best": float(np.max(rewards)),
        "action_counts": action_counts,
        "mean_top_prob": float(np.mean(top_probs)) if top_probs else 0.0,
        "mean_entropy": float(np.mean(entropies)) if entropies else 0.0,
    }


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


def log_eval(steps_done, stats, saved_best_mean, saved_best_worst, elapsed_seconds):
    EVAL_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    write_header = not EVAL_LOG_PATH.exists()

    with EVAL_LOG_PATH.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "steps_done",
                "mean_reward",
                "worst_reward",
                "best_reward",
                "saved_best_mean",
                "saved_best_worst",
                "mean_top_prob",
                "mean_entropy",
                "action_counts",
                "elapsed_seconds",
            ],
        )
        if write_header:
            writer.writeheader()
        writer.writerow(
            {
                "steps_done": steps_done,
                "mean_reward": stats["mean"],
                "worst_reward": stats["worst"],
                "best_reward": stats["best"],
                "saved_best_mean": int(saved_best_mean),
                "saved_best_worst": int(saved_best_worst),
                "mean_top_prob": stats["mean_top_prob"],
                "mean_entropy": stats["mean_entropy"],
                "action_counts": json.dumps(stats["action_counts"]),
                "elapsed_seconds": elapsed_seconds,
            }
        )


def train():
    env = None
    process = None
    log_file = None

    print(f"Run id: {RUN_ID}")
    print(f"Logging to: {RUN_LOG_DIR}")

    try:
        if _can_connect_to_minecraft(timeout=0.5):
            print("Existing Minecraft RL socket detected; using the running client.")
        else:
            process, log_file = start_minecraft_client()
            print(f"Started Minecraft with ./gradlew runClient. Log: {RUNCLIENT_LOG_PATH}")
            print("Waiting for Minecraft socket...")
            wait_for_minecraft_socket(process)

        env = ParkourRL(
            camera_action_penalty=CAMERA_ACTION_PENALTY,
            platform_reward=PLATFORM_REWARD,
            platform_z_step=PLATFORM_Z_STEP,
        )
        num_actions = len(env.action_table)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Training on device: {device}"
              + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))

        mlp_input_size = env.obs_shape["mlp"][1]
        model = ParkourCNN(
            num_actions=num_actions,
            mlp_input_size=mlp_input_size,
            stack_size=env.stack_size,
        ).to(device)
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
        next_eval_step = EVAL_INTERVAL_STEPS
        best_mean_reward = -float("inf")
        best_worst_reward = -float("inf")
        training_start_time = time.time()
        # Inference timing probe: average ms per action selection across the current rollout.
        # Useful for confirming whether Python (CNN forward) is the bottleneck vs. the game tick.
        infer_time_total = 0.0
        infer_time_count = 0

        obs, info = env.reset()

        while steps_done < total_timesteps:
            # ---- collect experience ----
            for _ in range(rollout_steps):
                camera_masked = steps_done < MASK_CAMERA_ACTIONS_INITIAL_STEPS
                infer_start = time.perf_counter()
                if camera_masked:
                    action, log_prob, value = choose_action_sampled_masked(model, obs, CAMERA_ACTION_IDS)
                else:
                    action, log_prob, value = model.get_action(obs)
                infer_time_total += time.perf_counter() - infer_start
                infer_time_count += 1
                next_obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                buffer.add(copy_obs(obs), action, log_prob, reward, value, float(done), camera_masked)

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
            avg_infer_ms = (
                1000.0 * infer_time_total / infer_time_count if infer_time_count else 0.0
            )
            print(
                f"Progress {steps_done}/{total_timesteps}  "
                f"elapsed={format_duration(elapsed_seconds)}  "
                f"speed={steps_per_second:.2f} steps/s  "
                f"infer={avg_infer_ms:.1f} ms/step  "
                f"eta={format_duration(eta_seconds)}"
            )
            infer_time_total = 0.0
            infer_time_count = 0

            if steps_done - last_step_checkpoint >= STEP_CHECKPOINT_INTERVAL:
                path = save_checkpoint(model, f"checkpoint_step_{steps_done}.pt")
                print(f"Saved timestep checkpoint: {path}")
                last_step_checkpoint = steps_done

            if steps_done >= next_eval_step:
                stats = evaluate_policy(model, env)
                mean_reward = stats["mean"]
                worst_reward = stats["worst"]
                best_reward = stats["best"]
                print(
                    f"Eval mean={mean_reward:.2f}, "
                    f"worst={worst_reward:.2f}, best={best_reward:.2f}  "
                    f"top_prob={stats['mean_top_prob']:.3f}  "
                    f"entropy={stats['mean_entropy']:.3f}"
                )
                print(f"  action counts -> {format_action_counts(stats['action_counts'])}")

                saved_best_mean = mean_reward > best_mean_reward
                if saved_best_mean:
                    best_mean_reward = mean_reward
                    path = save_checkpoint(model, BEST_MEAN_CHECKPOINT, update_latest=False)
                    print(f"Saved best mean checkpoint: {path}")

                saved_best_worst = worst_reward > best_worst_reward
                if saved_best_worst:
                    best_worst_reward = worst_reward
                    path = save_checkpoint(model, BEST_WORST_CHECKPOINT, update_latest=False)
                    print(f"Saved best worst checkpoint: {path}")

                eval_elapsed = time.time() - training_start_time
                log_eval(steps_done, stats, saved_best_mean, saved_best_worst, eval_elapsed)

                while next_eval_step <= steps_done:
                    next_eval_step += EVAL_INTERVAL_STEPS
                obs, info = env.reset()
                episode_reward = 0.0
                episode_steps = 0
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
