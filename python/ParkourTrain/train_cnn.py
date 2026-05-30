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
    COURSES,
    ParkourRL,
    RUNCLIENT_LOG_PATH,
    _can_connect_to_minecraft,
    make_multihead_action,
    start_minecraft_client,
    wait_for_minecraft_socket,
)
from model import ParkourCNN, ParkourCNNMultiHead

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
EVAL_EPISODES = 10
BEST_MEAN_CHECKPOINT = "best_mean.pt"
BEST_WORST_CHECKPOINT = "best_worst.pt"

# PPO entropy bonus coefficient. Lower = sharper / less exploratory policy.
# 0.001 let the policy collapse to top_prob~0.87 and argmax to a single-action sequence;
# 0.01 keeps enough spread that argmax tracks the actual best action per state.
# Multi-head now has THREE independent Bernoulli heads (W, Jump, Sprint), so max joint
# entropy is 3*ln(2)=2.08 — same coef spreads thinner per head, hence the higher base.
ENTROPY_COEF_CATEGORICAL = 0.01
ENTROPY_COEF_MULTIHEAD = 0.015
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
# Multi-head policy switch: independent Bernoulli heads for W and Jump instead of
# a single Categorical over ACTION_TABLE. The agent learns to hold-or-release W
# independently of when to jump. Sprint auto-tied to W in env.
# Set False to fall back to the categorical ACTION_TABLE path (kept intact for
# the diagonal/varied-height courses planned later).
USE_MULTI_HEAD = True
ENTROPY_COEF = ENTROPY_COEF_MULTIHEAD if USE_MULTI_HEAD else ENTROPY_COEF_CATEGORICAL
# 8 joint buckets for multi-head action counts: index = w + 2*jump + 4*sprint.
# Sprint-without-W is a no-op in MC, so buckets 4 (ctrl alone) and 6 (ctrl+jip) behave
# like stand / jip respectively; kept distinct for diagnostics.
MULTIHEAD_BUCKET_LABELS = [
    "stand",        # 0: (w0 j0 s0)
    "walk",         # 1: (w1 j0 s0)
    "jip",          # 2: (w0 j1 s0)
    "walk-jump",    # 3: (w1 j1 s0)
    "ctrl-noop",    # 4: (w0 j0 s1)
    "sprint",       # 5: (w1 j0 s1)
    "ctrl-jip",     # 6: (w0 j1 s1)
    "sprint-jump",  # 7: (w1 j1 s1)
]
MULTIHEAD_NUM_BUCKETS = len(MULTIHEAD_BUCKET_LABELS)

# Warm-start: load the solved 1-block policy and continue, instead of training from
# scratch. The sprint_head is new (not in the 2-head checkpoint) so it loads fresh via
# strict=False; cnn/mlp/shared/w_head/jump_head/critic transfer directly.
WARM_START = True
WARM_START_CHECKPOINT = CHECKPOINT_DIR / "best_mean.pt"
# The warm-started policy has collapsed entropy (~0.46). Boost the entropy coef at the
# start to reopen exploration for the bigger 2/3-block gaps, then linearly decay back to
# the base coef over this many steps. Only active when a warm-start actually loaded.
WARM_START_ENTROPY_COEF = 0.03
WARM_START_ENTROPY_DECAY_STEPS = 150_000

# Additive course curriculum: phase difficulty in by steps_done, but never drop a course
# once it's in the pool (cumulative -> no catastrophic forgetting). 1block is omitted from
# phase 0 because the warm-start already solves it; concentrating phase 0 on 2block alone
# gives the freshly-initialized sprint head a clean signal to discover sprint-jump. Each
# phase is a (steps_threshold, [course names]) entry, lowest threshold first. The pool used
# is the last entry whose threshold <= steps_done. Within a phase, random resets hold one
# lane for COURSE_BLOCK_SIZE consecutive resets (block sampling).
CURRICULUM_PHASES = [
    (0,        ["2block"]),
    (50_000,   ["1block", "2block"]),
    (100_000,  ["1block", "2block", "3block"]),
    (150_000,  ["1block", "2block", "3block", "mixed"]),
]
COURSE_BLOCK_SIZE = 5


def current_phase_courses(steps_done):
    """Return (phase_index, [course names]) for the active curriculum phase."""
    phase_index = 0
    for i, (threshold, _names) in enumerate(CURRICULUM_PHASES):
        if steps_done >= threshold:
            phase_index = i
        else:
            break
    return phase_index, CURRICULUM_PHASES[phase_index][1]

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


def eval_diagnostics_multihead(model, obs):
    """Multi-head eval action + diagnostics.

    SAMPLES from the per-head Bernoullis instead of taking argmax. Argmax per
    head can only ever produce the diagonal corners of the joint grid, which makes
    off-diagonal actions like jip=(w0,j1,s0) structurally unreachable at eval even
    when the policy assigns them real probability. Sampling here matches what the
    policy actually does during PPO rollouts, so eval reward becomes comparable to
    rollout reward.

    Returns (bucket, top_prob, entropy, (w_act, jump_act, sprint_act)) where
    bucket = w + 2*jump + 4*sprint (0..7) follows the sampled action. top_prob and
    entropy still describe the policy distribution, not the realized sample.
    """
    with torch.no_grad():
        device = next(model.parameters()).device
        frame = torch.as_tensor(obs["frame"], dtype=torch.float32, device=device).unsqueeze(0)
        mlp = torch.as_tensor(obs["mlp"], dtype=torch.float32, device=device).unsqueeze(0)
        w_logits, jump_logits, sprint_logits, _ = model(frame, mlp)
        w_p = torch.sigmoid(w_logits)
        jump_p = torch.sigmoid(jump_logits)
        sprint_p = torch.sigmoid(sprint_logits)
        w_dist = torch.distributions.Bernoulli(probs=w_p)
        jump_dist = torch.distributions.Bernoulli(probs=jump_p)
        sprint_dist = torch.distributions.Bernoulli(probs=sprint_p)
        w_act = int(w_dist.sample().item())
        jump_act = int(jump_dist.sample().item())
        sprint_act = int(sprint_dist.sample().item())
        # Joint top-prob: prob of the most-likely joint action under the policy.
        w_top = float(max(w_p.item(), 1.0 - w_p.item()))
        jump_top = float(max(jump_p.item(), 1.0 - jump_p.item()))
        sprint_top = float(max(sprint_p.item(), 1.0 - sprint_p.item()))
        top_prob = w_top * jump_top * sprint_top
        entropy = float((w_dist.entropy() + jump_dist.entropy() + sprint_dist.entropy()).item())
        bucket = w_act + 2 * jump_act + 4 * sprint_act
    return bucket, top_prob, entropy, (w_act, jump_act, sprint_act)


def choose_action_deterministic(model, obs):
    action, _, _ = eval_diagnostics(model, obs)
    return action


def format_action_counts(action_counts, labels=None):
    total = sum(action_counts) or 1
    parts = []
    for action_id, count in enumerate(action_counts):
        if count == 0:
            continue
        label = labels[action_id] if labels and action_id < len(labels) else str(action_id)
        parts.append(f"{label}={count} ({100.0 * count / total:.0f}%)")
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

def ppo_update(model, optimizer, buffer, device, epochs=4, clip_epsilon=0.2, batch_size=64, multi_head=False, entropy_coef=None):
    if len(buffer) == 0:
        return
    if entropy_coef is None:
        entropy_coef = ENTROPY_COEF

    frames = torch.as_tensor(np.stack([state["frame"] for state in buffer.states]), dtype=torch.float32, device=device)
    mlp_states = torch.as_tensor(np.stack([state["mlp"] for state in buffer.states]), dtype=torch.float32, device=device)
    if multi_head:
        # actions stored as (w, jump, sprint) tuples; stack into [N, 3] float tensor for Bernoulli.log_prob
        actions = torch.as_tensor(np.asarray(buffer.actions, dtype=np.float32), dtype=torch.float32, device=device)
    else:
        actions = torch.as_tensor(buffer.actions, dtype=torch.long, device=device)
    old_log_probs = torch.stack(buffer.log_probs).detach().to(device)
    values = [v.item() for v in buffer.values]
    # Per-step flag: was the camera-action mask active when this transition was collected?
    # Masked transitions must be re-masked below so the PPO ratio/entropy match the policy
    # that actually acted (otherwise the entropy bonus inflates the masked actions).
    # Multi-head path has no camera actions and never sets masked=True, so this is a no-op there.
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

            if multi_head:
                w_logits, jump_logits, sprint_logits, values_pred = model(batch_frames, batch_mlp_states)
                w_dist = torch.distributions.Bernoulli(logits=w_logits)
                jump_dist = torch.distributions.Bernoulli(logits=jump_logits)
                sprint_dist = torch.distributions.Bernoulli(logits=sprint_logits)
                w_actions = batch_actions[:, 0]
                jump_actions = batch_actions[:, 1]
                sprint_actions = batch_actions[:, 2]
                # Joint log-prob = sum of independent per-head log-probs (heads are independent).
                new_log_probs = (
                    w_dist.log_prob(w_actions)
                    + jump_dist.log_prob(jump_actions)
                    + sprint_dist.log_prob(sprint_actions)
                )
                entropy = (w_dist.entropy() + jump_dist.entropy() + sprint_dist.entropy()).mean()
            else:
                logits, values_pred = model(batch_frames, batch_mlp_states)
                # Re-apply the camera mask for transitions collected under the curriculum mask,
                # so recomputed log-probs/entropy come from the same distribution that acted.
                if batch_masked.any() and camera_cols.numel() > 0:
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

            loss = actor_loss + 0.5 * critic_loss - entropy_coef * entropy

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),0.5)
            optimizer.step()


def save_checkpoint(model, filename, update_latest=True, metadata=None):
    """Save with optional metadata wrapper. New format: {"state_dict": ..., "meta": {...}}.
    Loaders should accept both raw state_dicts (old) and the wrapped dict (new).
    """
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_path = CHECKPOINT_DIR / filename
    payload = {"state_dict": model.state_dict(), "meta": metadata or {}}
    torch.save(payload, checkpoint_path)
    if update_latest:
        torch.save(payload, CHECKPOINT_DIR / "latest.pt")
    return checkpoint_path


def evaluate_policy(model, env, episodes=EVAL_EPISODES, multi_head=False, course=None):
    """Run `episodes` greedy(=sampled) eval episodes. If `course` is given, every
    episode resets onto that specific course (per-course eval); otherwise reset uses
    the env's own course-selection policy."""
    if multi_head:
        num_buckets = MULTIHEAD_NUM_BUCKETS
    else:
        num_buckets = len(env.action_table)
    rewards = []
    action_counts = [0] * num_buckets
    top_probs = []
    entropies = []
    was_training = model.training
    model.eval()

    try:
        for _ in range(episodes):
            obs, info = env.reset(course=course)
            episode_reward = 0.0

            while True:
                if multi_head:
                    bucket, top_prob, entropy, (w_act, jump_act, sprint_act) = eval_diagnostics_multihead(model, obs)
                    action_counts[bucket] += 1
                    step_action = make_multihead_action(w_act, jump_act, sprint_act)
                else:
                    bucket, top_prob, entropy = eval_diagnostics(model, obs)
                    action_counts[bucket] += 1
                    step_action = bucket
                top_probs.append(top_prob)
                entropies.append(entropy)
                obs, reward, terminated, truncated, info = env.step(step_action)
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


def current_entropy_coef(steps_done, warm_started):
    """Entropy coef for this update. When warm-started, start boosted and decay
    linearly to the base coef over WARM_START_ENTROPY_DECAY_STEPS to reopen
    exploration that the loaded (collapsed) policy lost. Base coef otherwise."""
    if not (warm_started and WARM_START):
        return ENTROPY_COEF
    frac = min(1.0, steps_done / max(1, WARM_START_ENTROPY_DECAY_STEPS))
    return WARM_START_ENTROPY_COEF * (1.0 - frac) + ENTROPY_COEF * frac


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


def log_eval(steps_done, course, stats, saved_best_mean, saved_best_worst, elapsed_seconds):
    """Append one eval row. `course` is a course name, or "ALL" for the cross-course
    aggregate. The saved_best flags are only meaningful on the ALL row (checkpointing
    is driven by the cross-course average)."""
    EVAL_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    write_header = not EVAL_LOG_PATH.exists()

    with EVAL_LOG_PATH.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "steps_done",
                "course",
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
                "course": course,
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

        # Multi-course training via an additive curriculum (see CURRICULUM_PHASES): the
        # sampling pool of courses grows by difficulty over training, but never drops a
        # course, so the policy must keep reading each gap from the pixels and can't forget
        # easier ones. All courses are registered on the env so eval can score each one
        # separately (see below) even before it enters the sampling pool.
        eval_courses = list(COURSES.values())
        env = ParkourRL(
            camera_action_penalty=CAMERA_ACTION_PENALTY,
            platform_reward=PLATFORM_REWARD,
            platform_z_step=PLATFORM_Z_STEP,
            action_repeat=2 if USE_MULTI_HEAD else 5,
            courses=eval_courses,
            random_courses=True,
        )
        print(f"Courses ({len(eval_courses)}): {', '.join(c.name for c in eval_courses)}  "
              f"(additive curriculum; pool set per phase)")
        num_actions = len(env.action_table)
        # Filter camera mask IDs to the current action table. MINIMAL_ACTION_TABLE has no
        # camera actions, so this becomes an empty set and the curriculum is a no-op.
        global CAMERA_ACTION_IDS
        CAMERA_ACTION_IDS = {i for i in CAMERA_ACTION_IDS if i < num_actions}

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Training on device: {device}"
              + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))

        mlp_input_size = env.obs_shape["mlp"][1]
        if USE_MULTI_HEAD:
            model = ParkourCNNMultiHead(
                mlp_input_size=mlp_input_size,
                stack_size=env.stack_size,
            ).to(device)
            print("Policy: ParkourCNNMultiHead (independent W + Jump + Sprint Bernoulli heads)")
        else:
            model = ParkourCNN(
                num_actions=num_actions,
                mlp_input_size=mlp_input_size,
                stack_size=env.stack_size,
            ).to(device)
            print(f"Policy: ParkourCNN (categorical over {num_actions} ACTION_TABLE entries)")

        # Warm-start from the solved 1-block policy. strict=False because the new
        # sprint_head isn't in the 2-head checkpoint (loads fresh); everything else
        # transfers. warm_started gates the entropy boost below.
        warm_started = False
        if USE_MULTI_HEAD and WARM_START and WARM_START_CHECKPOINT.exists():
            raw = torch.load(WARM_START_CHECKPOINT, map_location=device)
            state_dict = raw["state_dict"] if isinstance(raw, dict) and "state_dict" in raw else raw
            result = model.load_state_dict(state_dict, strict=False)
            warm_started = True
            print(f"Warm-started from {WARM_START_CHECKPOINT}")
            if result.missing_keys:
                print(f"  fresh (not in checkpoint): {sorted(result.missing_keys)}")
            if result.unexpected_keys:
                print(f"  ignored (not in model): {sorted(result.unexpected_keys)}")
            print(f"  entropy boost {WARM_START_ENTROPY_COEF} -> {ENTROPY_COEF} over "
                  f"{WARM_START_ENTROPY_DECAY_STEPS} steps")
        elif WARM_START and USE_MULTI_HEAD:
            print(f"Warm-start requested but {WARM_START_CHECKPOINT} not found; training from scratch.")

        checkpoint_meta = {
            "model_type": "multi_head" if USE_MULTI_HEAD else "categorical",
            "mlp_input_size": mlp_input_size,
            "stack_size": env.stack_size,
            "num_actions": num_actions,
        }
        optimizer = optim.AdamW(model.parameters(),lr = 0.0003, weight_decay =0.01)
        buffer = RolloutBuffer()

        rollout_steps = 2048
        total_timesteps = 1_000_000
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

        # Apply the starting curriculum phase before the first reset.
        active_phase_idx, phase_names = current_phase_courses(steps_done)
        env.set_sampling_courses(phase_names, block_size=COURSE_BLOCK_SIZE)
        print(f"Curriculum phase {active_phase_idx}: sampling {phase_names} "
              f"(hold each lane {COURSE_BLOCK_SIZE} resets)")

        obs, info = env.reset()

        while steps_done < total_timesteps:
            # ---- collect experience ----
            for _ in range(rollout_steps):
                # Camera curriculum mask is categorical-only; multi-head has no camera actions.
                camera_masked = (not USE_MULTI_HEAD) and steps_done < MASK_CAMERA_ACTIONS_INITIAL_STEPS
                infer_start = time.perf_counter()
                if USE_MULTI_HEAD:
                    action, log_prob, value = model.get_action(obs)  # action = (w_bit, jump_bit)
                    step_input = make_multihead_action(*action)
                elif camera_masked:
                    action, log_prob, value = choose_action_sampled_masked(model, obs, CAMERA_ACTION_IDS)
                    step_input = action
                else:
                    action, log_prob, value = model.get_action(obs)
                    step_input = action
                infer_time_total += time.perf_counter() - infer_start
                infer_time_count += 1
                next_obs, reward, terminated, truncated, info = env.step(step_input)
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

                    # Advance the additive curriculum if we've crossed a phase boundary.
                    new_phase_idx, phase_names = current_phase_courses(steps_done)
                    if new_phase_idx != active_phase_idx:
                        active_phase_idx = new_phase_idx
                        env.set_sampling_courses(phase_names, block_size=COURSE_BLOCK_SIZE)
                        print(f"--> Curriculum advanced to phase {active_phase_idx} "
                              f"at step {steps_done}: sampling {phase_names}")
                    obs, info = env.reset()

                    if (
                        episode % EPISODE_CHECKPOINT_INTERVAL == 0
                        and episode != last_checkpoint_episode
                    ):
                        path = save_checkpoint(model, f"checkpoint_episode_{episode}.pt", metadata=checkpoint_meta)
                        print(f"Saved episode checkpoint: {path}")
                        last_checkpoint_episode = episode

                if steps_done >= total_timesteps:
                    break

            # ---- learn from collected experience ----
            entropy_coef = current_entropy_coef(steps_done, warm_started)
            ppo_update(model, optimizer, buffer, device, multi_head=USE_MULTI_HEAD, entropy_coef=entropy_coef)
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
                f"ent_coef={entropy_coef:.4f}  "
                f"eta={format_duration(eta_seconds)}"
            )
            infer_time_total = 0.0
            infer_time_count = 0

            if steps_done - last_step_checkpoint >= STEP_CHECKPOINT_INTERVAL:
                path = save_checkpoint(model, f"checkpoint_step_{steps_done}.pt", metadata=checkpoint_meta)
                print(f"Saved timestep checkpoint: {path}")
                last_step_checkpoint = steps_done

            if steps_done >= next_eval_step:
                # Per-course eval: score each course separately so we can see which
                # difficulties are solved vs. struggling. Checkpointing is driven by the
                # cross-course aggregate (mean of per-course means / min of per-course
                # worsts) because we want one policy that's good at ALL courses.
                labels = MULTIHEAD_BUCKET_LABELS if USE_MULTI_HEAD else None
                eval_elapsed = time.time() - training_start_time
                per_course = {}
                for course in eval_courses:
                    cstats = evaluate_policy(model, env, multi_head=USE_MULTI_HEAD, course=course)
                    per_course[course.name] = cstats
                    print(
                        f"Eval[{course.name}] mean={cstats['mean']:.2f}, "
                        f"worst={cstats['worst']:.2f}, best={cstats['best']:.2f}  "
                        f"top_prob={cstats['mean_top_prob']:.3f}  entropy={cstats['mean_entropy']:.3f}"
                    )
                    print(f"    action counts -> {format_action_counts(cstats['action_counts'], labels=labels)}")
                    log_eval(steps_done, course.name, cstats, False, False, eval_elapsed)

                # Cross-course aggregate.
                mean_reward = float(np.mean([s["mean"] for s in per_course.values()]))
                worst_reward = float(np.min([s["worst"] for s in per_course.values()]))
                best_reward = float(np.max([s["best"] for s in per_course.values()]))
                agg_counts = [
                    sum(s["action_counts"][i] for s in per_course.values())
                    for i in range(len(next(iter(per_course.values()))["action_counts"]))
                ]
                agg_stats = {
                    "mean": mean_reward,
                    "worst": worst_reward,
                    "best": best_reward,
                    "action_counts": agg_counts,
                    "mean_top_prob": float(np.mean([s["mean_top_prob"] for s in per_course.values()])),
                    "mean_entropy": float(np.mean([s["mean_entropy"] for s in per_course.values()])),
                }
                print(
                    f"Eval[ALL] mean={mean_reward:.2f} (avg of course means), "
                    f"worst={worst_reward:.2f} (min), best={best_reward:.2f} (max)"
                )

                saved_best_mean = mean_reward > best_mean_reward
                if saved_best_mean:
                    best_mean_reward = mean_reward
                    path = save_checkpoint(model, BEST_MEAN_CHECKPOINT, update_latest=False, metadata=checkpoint_meta)
                    print(f"Saved best mean checkpoint: {path}")

                saved_best_worst = worst_reward > best_worst_reward
                if saved_best_worst:
                    best_worst_reward = worst_reward
                    path = save_checkpoint(model, BEST_WORST_CHECKPOINT, update_latest=False, metadata=checkpoint_meta)
                    print(f"Saved best worst checkpoint: {path}")

                # Always-save eval checkpoint so we never lose intermittent breakthroughs
                # (last run hit best=59 at step 141k but that snapshot wasn't kept).
                eval_ckpt = save_checkpoint(
                    model, f"checkpoint_eval_{steps_done}.pt", update_latest=False, metadata=checkpoint_meta,
                )
                print(f"Saved eval checkpoint: {eval_ckpt}")

                log_eval(steps_done, "ALL", agg_stats, saved_best_mean, saved_best_worst, eval_elapsed)

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
