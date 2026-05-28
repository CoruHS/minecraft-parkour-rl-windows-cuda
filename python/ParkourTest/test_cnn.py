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
    LAN_PORT_PATH,
    RUNCLIENT_LOG_PATH,
    ParkourRL,
    _can_connect_to_minecraft,
    _lan_port_from_info,
    start_minecraft_client,
    wait_for_minecraft_socket,
)
from model import ParkourCNN


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
    """Pick an action. sample=False -> argmax (temperature ignored).
    sample=True -> categorical sample after dividing logits by `temperature`.
    Lower temperature sharpens toward argmax; higher temperature flattens toward uniform.
    """
    with torch.no_grad():
        frame = torch.as_tensor(obs["frame"], dtype=torch.float32).unsqueeze(0)
        mlp_state = torch.as_tensor(obs["mlp"], dtype=torch.float32).unsqueeze(0)
        logits, _ = model(frame, mlp_state)
        if sample:
            dist = torch.distributions.Categorical(logits=logits / max(temperature, 1e-6))
            return int(dist.sample().item())
        return int(torch.argmax(logits, dim=-1).item())


def load_model(checkpoint_path, num_actions, mlp_input_size, stack_size):
    model = ParkourCNN(
        num_actions=num_actions,
        mlp_input_size=mlp_input_size,
        stack_size=stack_size,
    )
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return model


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

        env = ParkourRL(max_steps=args.max_steps)
        model = load_model(
            checkpoint_path,
            len(env.action_table),
            env.obs_shape["mlp"][1],
            env.stack_size,
        )
        print(f"Loaded checkpoint: {checkpoint_path}")

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
                action = choose_action(
                    model, obs, sample=args.sample, temperature=args.temperature,
                )
                obs, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward
                steps += 1
                pos = info["packet"]["position"]

                print(
                    f"\repisode={episode} step={steps} action={action} "
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
