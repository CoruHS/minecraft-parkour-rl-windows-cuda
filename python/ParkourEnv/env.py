import numpy as np
import gymnasium as gym
import time
import json
import os
import socket
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MINECRAFT_HOST = "127.0.0.1"
MINECRAFT_PORT = 5005
MINECRAFT_WORLD_FOLDER = "ParkourRL"
RUNCLIENT_LOG_PATH = PROJECT_ROOT / "run" / "parkour_env_runclient.log"
LAN_PORT_PATH = PROJECT_ROOT / "run" / "parkour_lan_port.txt"
SOCKET_START_TIMEOUT_SECONDS = 180

YAW_STEP = 2.0
PITCH_STEP = 2.0


def _action(
    forward=False,
    back=False,
    left=False,
    right=False,
    jump=False,
    sprint=False,
    yaw_delta=0.0,
    pitch_delta=0.0,
):
    return {
        "forward": forward,
        "back": back,
        "left": left,
        "right": right,
        "jump": jump,
        "sprint": sprint,
        "yaw_delta": yaw_delta,
        "pitch_delta": pitch_delta,
    }


ACTION_TABLE = [
    # 0: release every key / no-op
    _action(),
    # 1: careful forward movement
    _action(forward=True),
    # 2: main parkour movement
    _action(forward=True, sprint=True),
    # 3: main jump action for gaps
    _action(forward=True, jump=True, sprint=True),
    # 4-5: small path correction while moving
    _action(forward=True, left=True),
    _action(forward=True, right=True),
    # 6-7: fast path correction while moving
    _action(forward=True, left=True, sprint=True),
    _action(forward=True, right=True, sprint=True),
    # 8-9: diagonal sprint jumps
    _action(forward=True, left=True, jump=True, sprint=True),
    _action(forward=True, right=True, jump=True, sprint=True),
    # 10-13: recovery actions
    _action(jump=True),
    _action(back=True),
    _action(left=True),
    _action(right=True),
    # 14-15: camera-only yaw correction
    _action(yaw_delta=-YAW_STEP),
    _action(yaw_delta=YAW_STEP),
    # 16-17: turn while keeping speed
    _action(forward=True, sprint=True, yaw_delta=-YAW_STEP),
    _action(forward=True, sprint=True, yaw_delta=YAW_STEP),
    # 18-19: pitch control for the later CNN version
    _action(pitch_delta=-PITCH_STEP),
    _action(pitch_delta=PITCH_STEP),
]

# parameters
x = 0.0
y = 1.0
z = 0.0
base_yaw = 0.0
base_pitch = 0.0
chosen_height = 0.0
goal_x = 0.0
goal_y = 0.0
goal_z = 28.0
RESET_POSITION_TOLERANCE = 0.75
RESET_WAIT_PACKETS = 80


def _can_connect_to_minecraft(timeout=1.0):
    try:
        with socket.create_connection((MINECRAFT_HOST, MINECRAFT_PORT), timeout=timeout):
            return True
    except OSError:
        return False


def _gpu_offload_env():
    # On hybrid-GPU laptops (Intel iGPU + NVIDIA dGPU) Linux defaults rendering
    # to the integrated GPU. These env vars route OpenGL/Vulkan to the NVIDIA
    # card via PRIME render offload. No-op on machines without NVIDIA PRIME.
    env = os.environ.copy()
    env.setdefault("__NV_PRIME_RENDER_OFFLOAD", "1")
    env.setdefault("__GLX_VENDOR_LIBRARY_NAME", "nvidia")
    env.setdefault("__VK_LAYER_NV_optimus", "NVIDIA_only")
    return env


def start_minecraft_client():
    RUNCLIENT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    LAN_PORT_PATH.unlink(missing_ok=True)
    log_file = RUNCLIENT_LOG_PATH.open("a", encoding="utf-8")
    process = subprocess.Popen(
        ["./gradlew", "runClient"],
        cwd=PROJECT_ROOT,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        env=_gpu_offload_env(),
    )
    return process, log_file


def wait_for_minecraft_socket(process=None, timeout_seconds=SOCKET_START_TIMEOUT_SECONDS):
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        if process is not None and process.poll() is not None:
            raise RuntimeError(
                f"Minecraft exited before opening the socket. Check {RUNCLIENT_LOG_PATH}"
            )
        if _can_connect_to_minecraft(timeout=1.0):
            return
        time.sleep(1.0)

    raise TimeoutError(
        f"Minecraft socket did not open on {MINECRAFT_HOST}:{MINECRAFT_PORT} "
        f"within {timeout_seconds} seconds. Check {RUNCLIENT_LOG_PATH}"
    )


def _lan_port_from_packet(packet):
    lan_port = packet.get("lan_port", -1)
    try:
        lan_port = int(lan_port)
    except (TypeError, ValueError):
        return None
    return lan_port if lan_port > 0 else None


def _lan_port_from_file():
    try:
        lan_port = int(LAN_PORT_PATH.read_text(encoding="utf-8").strip())
    except (FileNotFoundError, TypeError, ValueError):
        return None
    return lan_port if lan_port > 0 else None


def _lan_port_from_info(info):
    return _lan_port_from_packet(info["packet"]) or _lan_port_from_file()


class ParkourRL(gym.Env):
    def __init__(self,
                 log_path: str = "run/logs/latest.log",
                 stack_size: int = 4,
                 action_repeat: int = 2,
                 tickrate: float = 80.0,
                 goal_position=None,
                 goal_x: float = goal_x,
                 goal_y: float = goal_y,
                 goal_z: float = goal_z,
                 fall_y=None,
                 action_table: list = ACTION_TABLE,
                 max_steps=1000):
        # Initilize basically all the variables the code will utilize(yes theres that many).
        # More variables will be added later for CNN + MLP implementation
        self.log_path = log_path
        self.stack_size = stack_size
        self.action_repeat = action_repeat
        self.tickrate = tickrate
        self.fall_y = fall_y
        self.state_stack = np.zeros((self.stack_size, 14), dtype=np.float32)
        self.last_position = None
        self.action_table = action_table
        self.action_space = gym.spaces.Discrete(len(self.action_table))
        self.max_steps = max_steps
        self.elapsed_steps = 0
        self.last_distance_to_goal = None
        self.host = MINECRAFT_HOST
        self.port = MINECRAFT_PORT
        self.socket = socket.create_connection((self.host, self.port), timeout=10)
        self.reader = self.socket.makefile('r', encoding="utf-8")
        self.writer = self.socket.makefile("w", encoding="utf-8")
        # I will choose and initialize a goal coordinate myself
        if goal_position is None:
            goal_position = (goal_x, goal_y, goal_z)
        self.goal_position = np.asarray(goal_position, dtype=np.float32)
        self.obs_shape = (self.stack_size, 14)
        self.fall_y = chosen_height
        self.goal_radius = 1.5

        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.stack_size, 14),
            dtype=np.float32,
        )

    def reset(self, seed=None):
        # Reset the player back to the start everytime a certain condition(height < fall_y) is activated
        # first three zeroes indicate coordinate. the next 2 zeroes indicate pitch and yaw
        # this command can change so its a place holder for now.
        super().reset(seed=seed)
        self.elapsed_steps = 0
        self.state_stack = np.zeros((self.stack_size, 14), dtype=np.float32)

        self._send_action({
            "command": f"tp @p {x} {y} {z} {base_yaw} {base_pitch}",
            "forward": False,
            "back": False,
            "left": False,
            "right": False,
            "jump": False,
            "sprint": False,
            "yaw_delta": 0.0,
            "pitch_delta": 0.0,
        })

        target_position = np.array([x, y, z], dtype=np.float32)
        packet = None
        current_position = None

        for _ in range(RESET_WAIT_PACKETS):
            packet = self._wait_for_telemetry()
            current_position = self._position_from_packet(packet)
            if np.linalg.norm(current_position - target_position) <= RESET_POSITION_TOLERANCE:
                break
        else:
            raise RuntimeError(
                f"Reset did not reach target position {target_position.tolist()}; "
                f"latest position was {current_position.tolist()}"
            )

        obs_14 = self._obs_from_packet(packet)

        self.state_stack = np.repeat(obs_14[np.newaxis, :], self.stack_size, axis=0)
        self.last_position = current_position
        self.last_distance_to_goal = np.linalg.norm(current_position - self.goal_position)

        return self.state_stack.copy(), {"packet": packet}

    def step(self, actionid):
        # step() makes the environment play the game, calculate rewards, and choose actions until reset() is called.

        action = self.action_table[int(actionid)]
        self._send_action(action)

        reward = 0.0
        packet = None

        for _ in range(self.action_repeat):
            packet = self._wait_for_telemetry()
            reward += self._compute_reward(packet)

        mlp_state = self._obs_from_packet(packet)
        obs = self._update_stack(mlp_state)

        self.elapsed_steps += 1
        terminated = self._is_terminated(packet)
        truncated = self.elapsed_steps >= self.max_steps

        info = {
            "packet": packet,
            "action": action,
        }

        return obs, float(reward), terminated, truncated, info

    def close(self):
        # stops everything.
        self.writer.close()
        self.reader.close()
        self.socket.close()

    def _obs_from_packet(self, packet):
        return np.asarray(packet["mlp_state"], dtype=np.float32)

    def _position_from_packet(self, packet):
        pos = packet["position"]
        return np.array([pos["x"], pos["y"], pos["z"]], dtype=np.float32)

    def _update_stack(self, obs):
        # updates stack
        self.state_stack = np.concatenate([self.state_stack[1:], obs[np.newaxis, :]],
            axis=0,
        )
        return self.state_stack.copy()

    def _compute_reward(self, packet):
        # calculate the reward based on the distance of the known last position and goal position.
        current_position = self._position_from_packet(packet)
        current_distance = np.linalg.norm(current_position - self.goal_position)
        reward = self.last_distance_to_goal - current_distance
        reward -= 0.01

        self.last_distance_to_goal = current_distance
        self.last_position = current_position

        return reward

    def _is_terminated(self, packet):
        # check if episode is over --> returns True or False
        current_position = self._position_from_packet(packet)

        fell = current_position[1] < self.fall_y
        reached_goal = np.linalg.norm(current_position - self.goal_position) < self.goal_radius

        return fell or reached_goal

    def _send_action(self, action):
        # sends action to minecraft based on action table
        self.writer.write(json.dumps(action) + "\n")
        self.writer.flush()

    def _wait_for_telemetry(self):
        # returns the data from each line in console per tick.
        while True:
            line = self.reader.readline()
            if not line:
                raise RuntimeError("No line detected")
            packet = json.loads(line)
            if "mlp_state" in packet:
                return packet

# test environment
def main():
    process = None
    log_file = None
    env = None

    world_path = PROJECT_ROOT / "run" / "saves" / MINECRAFT_WORLD_FOLDER
    level_dat_path = world_path / "level.dat"
    if not level_dat_path.exists():
        print(f"Warning: expected world save does not exist yet: {level_dat_path}")
        print("Create that singleplayer world once before relying on auto-open.")

    try:
        if _can_connect_to_minecraft(timeout=0.5):
            print("Existing Minecraft RL socket detected; using the running client.")
        else:
            process, log_file = start_minecraft_client()
            print(f"Started Minecraft with ./gradlew runClient. Log: {RUNCLIENT_LOG_PATH}")
            print("Waiting for Minecraft socket...")
            wait_for_minecraft_socket(process)

        print("Socket is ready. Waiting for world telemetry...")
        env = ParkourRL()
        obs, info = env.reset()
        episode = 1
        episode_reward = 0.0
        last_lan_port = _lan_port_from_info(info)
        print(f"Environment reset. Observation shape: {obs.shape}")
        print(f"Player position: {info['packet']['position']}")
        if last_lan_port is not None:
            print(f"LAN server port: {last_lan_port}")
            print(f"Direct connect address: localhost:{last_lan_port}")
            print(f"LAN port file: {LAN_PORT_PATH}")
        else:
            print("LAN server port is not available yet; waiting for telemetry.")
        print("Running random actions. Press Ctrl+C to stop this launcher.")

        while True:
            action_id = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action_id)
            episode_reward += reward
            lan_port = _lan_port_from_info(info)
            if lan_port is not None and lan_port != last_lan_port:
                last_lan_port = lan_port
                print(f"\nLAN server port: {last_lan_port}")
                print(f"Direct connect address: localhost:{last_lan_port}")
                print(f"LAN port file: {LAN_PORT_PATH}")
            pos = info["packet"]["position"]
            lan_display = last_lan_port if last_lan_port is not None else "pending"
            print(
                f"\repisode={episode} step={env.elapsed_steps} action={action_id} "
                f"reward={reward:.3f} total={episode_reward:.3f} "
                f"lan={lan_display} position=({pos['x']:.2f}, {pos['y']:.2f}, {pos['z']:.2f})",
                end="",
                flush=True,
            )

            if terminated or truncated:
                reason = "fell/reached goal" if terminated else "max steps"
                print(f"\nEpisode {episode} ended: {reason}. Resetting...")
                obs, info = env.reset()
                last_lan_port = _lan_port_from_info(info) or last_lan_port
                episode += 1
                episode_reward = 0.0
    except KeyboardInterrupt:
        print("\nStopping launcher.")
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
