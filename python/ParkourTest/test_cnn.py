import argparse
import subprocess
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PARKOUR_ENV_PATH = PROJECT_ROOT / "python" / "ParkourEnv"
PARKOUR_TRAIN_PATH = PROJECT_ROOT / "python" / "ParkourTrain"
sys.path.insert(0, str(PARKOUR_ENV_PATH))
sys.path.insert(0, str(PARKOUR_TRAIN_PATH))

from env_cnn import (
    COURSES,
    LAN_PORT_PATH,
    RUNCLIENT_LOG_PATH,
    ParkourRL,
    _can_connect_to_minecraft,
    _lan_port_from_info,
    make_multihead_action,
    start_minecraft_client,
    wait_for_minecraft_socket,
)
from model import ParkourCNN, ParkourCNNMultiHead


def find_latest_checkpoint():
    checkpoint_dir = PROJECT_ROOT / "checkpoints_cnn"
    for filename in ("best_worst.pt", "best_mean.pt", "latest.pt"):
        checkpoint_path = checkpoint_dir / filename
        if checkpoint_path.exists():
            return checkpoint_path

    candidates = list(checkpoint_dir.glob("*.pt"))

    unique_candidates = list(dict.fromkeys(path.resolve() for path in candidates))
    if not unique_candidates:
        return None

    return max(unique_candidates, key=lambda path: path.stat().st_mtime)


def choose_action(model, obs, sample=False, temperature=1.0):
    """Pick a categorical action. sample=False -> argmax; sample=True -> sample(logits/temp)."""
    with torch.no_grad():
        frame = torch.as_tensor(obs["frame"], dtype=torch.float32).unsqueeze(0)
        mlp_state = torch.as_tensor(obs["mlp"], dtype=torch.float32).unsqueeze(0)
        logits, _ = model(frame, mlp_state)
        if sample:
            dist = torch.distributions.Categorical(logits=logits / max(temperature, 1e-6))
            return int(dist.sample().item())
        return int(torch.argmax(logits, dim=-1).item())


def choose_action_multihead(model, obs, sample=False, temperature=1.0):
    """Pick a (w_bit, jump_bit, sprint_bit) tuple from the multi-head model.

    sample=False -> argmax per head. sample=True -> Bernoulli sample with logits/temp.
    """
    with torch.no_grad():
        frame = torch.as_tensor(obs["frame"], dtype=torch.float32).unsqueeze(0)
        mlp_state = torch.as_tensor(obs["mlp"], dtype=torch.float32).unsqueeze(0)
        w_logits, jump_logits, sprint_logits, _ = model(frame, mlp_state)
        if sample:
            t = max(temperature, 1e-6)
            w_dist = torch.distributions.Bernoulli(logits=w_logits / t)
            jump_dist = torch.distributions.Bernoulli(logits=jump_logits / t)
            sprint_dist = torch.distributions.Bernoulli(logits=sprint_logits / t)
            w = int(w_dist.sample().item())
            j = int(jump_dist.sample().item())
            s = int(sprint_dist.sample().item())
        else:
            w = int((w_logits > 0).item())
            j = int((jump_logits > 0).item())
            s = int((sprint_logits > 0).item())
    return w, j, s


def _unwrap_checkpoint(raw):
    """Accept either a raw state_dict (legacy) or {state_dict, meta} (new)."""
    if isinstance(raw, dict) and "state_dict" in raw and isinstance(raw["state_dict"], dict):
        return raw["state_dict"], raw.get("meta", {})
    return raw, {}


def _detect_multi_head(state_dict, meta):
    if meta.get("model_type") == "multi_head":
        return True
    if meta.get("model_type") == "categorical":
        return False
    return any(k.startswith("w_head.") or k.startswith("jump_head.") for k in state_dict.keys())


def load_model(checkpoint_path, num_actions, mlp_input_size, stack_size):
    raw = torch.load(checkpoint_path, map_location="cpu")
    state_dict, meta = _unwrap_checkpoint(raw)
    multi_head = _detect_multi_head(state_dict, meta)
    if multi_head:
        model = ParkourCNNMultiHead(mlp_input_size=mlp_input_size, stack_size=stack_size)
        # strict=False so an older 2-head (W+Jump) checkpoint still loads into the
        # current 3-head model (sprint_head stays at init). New checkpoints load fully.
        result = model.load_state_dict(state_dict, strict=False)
        if result.missing_keys:
            print(f"Warning: checkpoint missing {sorted(result.missing_keys)} (using fresh init for those)")
        if result.unexpected_keys:
            print(f"Warning: checkpoint had unused keys {sorted(result.unexpected_keys)}")
    else:
        model = ParkourCNN(
            num_actions=num_actions,
            mlp_input_size=mlp_input_size,
            stack_size=stack_size,
        )
        model.load_state_dict(state_dict)
    model.eval()
    return model, multi_head


def parse_args():
    parser = argparse.ArgumentParser(description="Run a trained Minecraft parkour policy.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Path to a .pt checkpoint. Defaults to best_worst, best_mean, latest, then newest.",
    )
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=1000000)
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Sample from action probabilities instead of using argmax.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature (only used with --sample). 1.0 = raw probs, "
             "0.5 = sharpened toward top action, >1 = flattened.",
    )
    parser.add_argument(
        "--no-sample-multihead",
        action="store_true",
        help="Force argmax for multi-head checkpoints (debugging only — argmax "
             "can't reach off-diagonal action buckets like jip=(0,1,0)).",
    )
    parser.add_argument(
        "--course",
        choices=sorted(COURSES.keys()),
        default="1block",
        help="Which course to run (start/goal lane in the world). Default: 1block.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    checkpoint_path = args.checkpoint or find_latest_checkpoint()
    if checkpoint_path is None:
        raise FileNotFoundError(
            "No checkpoint_*.pt file found. Train first or pass --checkpoint path/to/file.pt"
        )
    checkpoint_path = checkpoint_path.resolve()

    process = None
    log_file = None
    env = None

    try:
        if _can_connect_to_minecraft(timeout=0.5):
            print("Existing Minecraft RL socket detected; using the running client.")
        else:
            process, log_file = start_minecraft_client()
            print(f"Started Minecraft with ./gradlew runClient. Log: {RUNCLIENT_LOG_PATH}")
            print("Waiting for Minecraft socket...")
            wait_for_minecraft_socket(process)

        # Build env first with default repeat, then rebuild if checkpoint is multi-head
        # so the test cadence matches what training used. The selected course fixes the
        # start/goal lane for every reset.
        course = COURSES[args.course]
        env = ParkourRL(max_steps=args.max_steps, courses=[course])
        model, multi_head = load_model(
            checkpoint_path,
            len(env.action_table),
            env.obs_shape["mlp"][1],
            env.stack_size,
        )
        if multi_head and env.action_repeat != 2:
            env.close()
            env = ParkourRL(max_steps=args.max_steps, action_repeat=2, courses=[course])
        print(f"Loaded checkpoint: {checkpoint_path} ({'multi-head' if multi_head else 'categorical'})")
        print(f"Course: {args.course}  start={course.start.tolist()}  goal={course.goal.tolist()}")

        obs, info = env.reset()
        lan_port = _lan_port_from_info(info)
        if lan_port is not None:
            print(f"LAN direct connect: localhost:{lan_port}")
        else:
            print(f"LAN port not available yet. Port file: {LAN_PORT_PATH}")

        for episode in range(1, args.episodes + 1):
            episode_reward = 0.0
            steps = 0

            while True:
                # Multi-head argmax-per-head is structurally broken (can't reach
                # off-diagonal buckets like jip=(0,1)). Force sampling unless the
                # user explicitly asks for argmax via --no-sample-multihead.
                multihead_sample = args.sample or (multi_head and not args.no_sample_multihead)
                if multi_head:
                    w, j, s = choose_action_multihead(
                        model, obs, sample=multihead_sample, temperature=args.temperature,
                    )
                    action_display = f"(w={w}, j={j}, s={s})"
                    step_input = make_multihead_action(w, j, s)
                else:
                    action = choose_action(
                        model, obs, sample=args.sample, temperature=args.temperature,
                    )
                    action_display = str(action)
                    step_input = action
                obs, reward, terminated, truncated, info = env.step(step_input)
                episode_reward += reward
                steps += 1
                pos = info["packet"]["position"]

                print(
                    f"\repisode={episode} step={steps} action={action_display} "
                    f"reward={reward:.3f} total={episode_reward:.3f} "
                    f"position=({pos['x']:.2f}, {pos['y']:.2f}, {pos['z']:.2f})",
                    end="",
                    flush=True,
                )

                if terminated or truncated:
                    reason = "terminated" if terminated else "truncated"
                    print(f"\nEpisode {episode} ended: {reason}, reward={episode_reward:.3f}")
                    if episode < args.episodes:
                        obs, info = env.reset()
                    break
    except KeyboardInterrupt:
        print("\nStopping test run.")
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
    main()
