import numpy as np
import gymnasium as gym
import time
import json
import os
import random
import socket
import subprocess
from pathlib import Path
import base64


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MINECRAFT_HOST = "127.0.0.1"
MINECRAFT_PORT = 5005
MINECRAFT_WORLD_FOLDER = "ParkourRL"
RUNCLIENT_LOG_PATH = PROJECT_ROOT / "run" / "parkour_env_runclient.log"
LAN_PORT_PATH = PROJECT_ROOT / "run" / "parkour_lan_port.txt"
SOCKET_START_TIMEOUT_SECONDS = 180

YAW_STEP = 2.0
PITCH_STEP = 2.0

FRAME_W = 84
FRAME_H = 84
FRAME_C = 3


def wrap_degrees(angle):
    return (angle + 180.0) % 360.0 - 180.0


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


FULL_ACTION_TABLE = [
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

# Minimal action set for straight-line 1-block-gap parkour. Drops camera, diagonals,
# and crucially the sprint+jump combo (which overshoots 1-block gaps - it clears
# ~3-4 blocks of horizontal distance). Smaller search space = much faster convergence.
# Switch to FULL_ACTION_TABLE for varied/diagonal/long-gap courses later.
MINIMAL_ACTION_TABLE = [
    _action(),                                  # 0: no-op (brake)
    _action(forward=True),                      # 1: walk forward
    _action(forward=True, sprint=True),         # 2: sprint forward (no jump)
    _action(forward=True, jump=True),           # 3: walk-jump (right size for 1-block gaps)
    _action(jump=True),                         # 4: jump in place
    _action(back=True),                         # 5: walk back (brake)
]

# Default points at the minimal set; train_cnn.py can override via action_table=...
ACTION_TABLE = MINIMAL_ACTION_TABLE


def make_multihead_action(w_bit, jump_bit, sprint_bit=0):
    """Build the action dict the multi-head policy emits.

    Sprint is now its own independent bit (ctrl key), not tied to W. This is what
    lets one policy span courses of different gap sizes: walk-jump clears short
    gaps, sprint-jump clears long ones, so the agent picks jump distance per gap.
    Sprint without forward is a no-op in vanilla MC (you only sprint while moving
    forward), so no special handling is needed for the (sprint=1, w=0) combo.
    No camera here; add yaw/pitch heads later for diagonals.
    """
    return _action(
        forward=bool(w_bit),
        sprint=bool(sprint_bit),
        jump=bool(jump_bit),
    )


class Course:
    """A single parkour course: a start pose and a goal, all in the same world.

    Different courses live at different x-lanes of the same Minecraft world, so
    switching courses is just teleporting to a different start/goal. progress_dir
    and center_x are precomputed so reset/reward can swap courses for free.
    """

    def __init__(self, name, start, goal, yaw=0.0, pitch=0.0, fall_y=0.0):
        self.name = name
        self.start = np.asarray(start, dtype=np.float32)
        self.goal = np.asarray(goal, dtype=np.float32)
        self.yaw = float(yaw)
        self.pitch = float(pitch)
        self.fall_y = float(fall_y)
        delta = self.goal - self.start
        horizontal = np.array([delta[0], 0.0, delta[2]], dtype=np.float32)
        distance = float(np.linalg.norm(horizontal))
        self.progress_dir = (
            horizontal / distance if distance > 1e-6 else np.zeros(3, dtype=np.float32)
        )
        # Lane center for the lane-deviation penalty is the course's own x.
        self.center_x = float(self.start[0])


# Registry of the courses that live in the world. All run in +z on their own x-lane
# (so yaw stays 0). Add 4-block / varied-height / diagonal courses here later.
COURSES = {
    "1block": Course("1block", start=(0, 1, 0), goal=(0, 1, 98)),
    "2block": Course("2block", start=(-5, 1, 0), goal=(-5, 1, 99)),
    "3block": Course("3block", start=(-11, 1, 0), goal=(-11, 1, 100)),
    "mixed": Course("mixed", start=(-19, 1, 0), goal=(-19, 1, 98)),
}

# parameters
x = 0
y = 1.0
z = 0
base_yaw = 0.0
base_pitch = 0.0
chosen_height = 0.0
goal_x = 0
goal_y = 1.0
goal_z = 98.0
RESET_POSITION_TOLERANCE = 0.75
# Tickrate-agnostic: at 100 TPS this is ~2 s wall-clock; at 40 TPS ~5 s. Older value (80)
# was tuned for 40 TPS and was too tight at higher client tickrates because the integrated
# server still runs at 20 TPS, so the tp round-trip eats a bigger fraction of the budget.
RESET_WAIT_PACKETS = 200
# Re-send tp every N packets if the player still isn't at the target. Handles two cases:
#   1) the first tp packet got swallowed or processed after the player had already left,
#   2) /tp does not zero player velocity in vanilla 1.20.1, so a player falling at
#      terminal velocity gets teleported but immediately falls again.
RESET_TP_RESEND_EVERY = 10


def _can_connect_to_minecraft(timeout=1.0):
    try:
        with socket.create_connection((MINECRAFT_HOST, MINECRAFT_PORT), timeout=timeout):
            return True
    except OSError:
        return False


def _gpu_offload_env():
    # On hybrid-GPU laptops (Intel iGPU + NVIDIA dGPU) Linux defaults rendering
    # to the integrated GPU. These env vars route OpenGL/Vulkan to the NVIDIA
    # card via PRIME render offload so Minecraft renders (and reads pixels back)
    # on the fast discrete GPU. No-op on machines without NVIDIA PRIME.
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
                 action_repeat: int = 5,
                 tickrate: float = 60.0,
                 goal_position=None,
                 goal_x: float = goal_x,
                 goal_y: float = goal_y,
                 goal_z: float = goal_z,
                 fall_y=None,
                 center_x: float = 0.0,
                 target_yaw: float = 0.0,
                 target_pitch: float = 0.0,
                 progress_weight: float = 2.0,
                 lane_penalty: float = 0.05,
                 yaw_penalty: float = 0.01,
                 pitch_penalty: float = 0.005,
                 time_penalty: float = 0.005,
                 fall_penalty: float = 5.0,
                 goal_bonus: float = 50.0,
                 camera_action_penalty: float = 0.0,
                 platform_reward: float = 0.0,
                 platform_z_step: float = 1.0,
                 action_table: list = ACTION_TABLE,
                 courses=None,
                 random_courses: bool = False,
                 max_steps=1000):
        # Initilize basically all the variables the code will utilize(yes theres that many).
        # More variables will be added later for CNN + MLP implementation
        self.log_path = log_path
        self.stack_size = stack_size
        self.action_repeat = action_repeat
        self.tickrate = tickrate
        # Optional env-level fall_y override; when None each course supplies its own.
        self.fall_y_override = fall_y
        self.state_stack = np.zeros((self.stack_size, 14), dtype=np.float32)
        self.last_position = None
        self.action_table = action_table
        self.action_space = gym.spaces.Discrete(len(self.action_table))
        self.max_steps = max_steps
        self.elapsed_steps = 0
        self.last_distance_to_goal = None
        self.progress_weight = progress_weight
        self.lane_penalty = lane_penalty
        self.yaw_penalty = yaw_penalty
        self.pitch_penalty = pitch_penalty
        self.time_penalty = time_penalty
        self.fall_penalty = fall_penalty
        self.goal_bonus = goal_bonus
        self.camera_action_penalty = camera_action_penalty
        # Discrete bonus each time the player lands on a platform farther along z than
        # ever before in this episode. Gives PPO a clean per-platform credit signal that
        # the continuous progress reward alone doesn't provide. 0.0 disables.
        self.platform_reward = platform_reward
        self.platform_z_step = platform_z_step
        self.furthest_landed_z = None
        # Whether the most recent action moved the camera (yaw/pitch); set in step().
        self.last_action_is_camera = False
        self.host = MINECRAFT_HOST
        self.port = MINECRAFT_PORT
        self.socket = socket.create_connection((self.host, self.port), timeout=10)
        self.reader = self.socket.makefile('r', encoding="utf-8")
        self.writer = self.socket.makefile("w", encoding="utf-8")
        # Course set. Default (courses=None) = a single course built from the legacy
        # start globals + goal params, so existing single-course callers are unchanged.
        if courses is None:
            if goal_position is None:
                goal_position = (goal_x, goal_y, goal_z)
            default_fall_y = chosen_height if fall_y is None else fall_y
            courses = [Course(
                "default",
                start=(x, y, z),
                goal=goal_position,
                yaw=base_yaw,
                pitch=base_pitch,
                fall_y=default_fall_y,
            )]
        self.courses = list(courses)
        self.courses_by_name = {c.name: c for c in self.courses}
        self.random_courses = random_courses
        # Sampling pool: the subset of self.courses that random resets draw from. It stays
        # = self.courses by default, but a curriculum can shrink/grow it via
        # set_sampling_courses() without touching self.courses (which eval still indexes by
        # name for explicit per-course resets).
        self.sampling_courses = list(self.courses)
        # Block sampling: hold one randomly-chosen lane for course_block_size consecutive
        # resets before rolling a new one, so the agent gets repeated tries on the same gap
        # instead of being teleported to a fresh lane on every fall.
        self.course_block_size = 1
        self._block_course = None        # the lane currently being repeated
        self._block_resets_left = 0      # resets remaining before a new lane is rolled
        # Apply the first course so start/goal/progress_dir/center_x are populated
        # before the first reset (reset re-selects per its course argument / randomness).
        self._apply_course(self.courses[0])
        self.obs_shape = {
            "frame": (self.stack_size * FRAME_C, FRAME_H, FRAME_W),
            "mlp": (self.stack_size, 14),
        }
        self.goal_radius = 1.5
        self.frame_stack = np.zeros(self.obs_shape["frame"], dtype=np.float32)


        self.observation_space = gym.spaces.Dict({
            "frame": gym.spaces.Box(0, 1, shape=self.obs_shape["frame"], dtype=np.float32),
            "mlp": gym.spaces.Box(-np.inf, np.inf, shape=self.obs_shape["mlp"], dtype=np.float32),
        })

    def _resolve_course(self, course):
        """Accept a Course, a course name, or None; return a Course in this env."""
        if course is None:
            return None
        if isinstance(course, Course):
            return course
        try:
            return self.courses_by_name[course]
        except KeyError:
            raise ValueError(
                f"Unknown course {course!r}; known courses: {list(self.courses_by_name)}"
            )

    def set_sampling_courses(self, courses, block_size=None):
        """Set which courses random resets draw from (used by the training curriculum).

        `courses` is a list of Course/name entries that must already be registered in this
        env (so eval can still resolve them by name). Resetting the pool clears the current
        block so the next reset rolls a fresh lane from the new pool. `block_size`, if given,
        updates how many consecutive resets reuse a chosen lane.
        """
        resolved = [self._resolve_course(c) for c in courses]
        for c in resolved:
            if c is None:
                raise ValueError("set_sampling_courses got an unresolved (None) course")
        self.sampling_courses = resolved
        if block_size is not None:
            self.course_block_size = max(1, int(block_size))
        # Force a fresh roll on the next reset so we don't keep replaying a lane that may
        # no longer be in the pool.
        self._block_course = None
        self._block_resets_left = 0

    def _apply_course(self, course):
        """Point all course-dependent state (start/goal/lane/fall) at `course`."""
        self.active_course = course
        self.start_position = course.start.copy()
        self.goal_position = course.goal.copy()
        self.progress_dir = course.progress_dir.copy()
        self.center_x = course.center_x
        self.target_yaw = course.yaw
        self.target_pitch = course.pitch
        self.fall_y = (
            self.fall_y_override if self.fall_y_override is not None else course.fall_y
        )

    def reset(self, seed=None, course=None):
        # Reset the player back to the start everytime a certain condition(height < fall_y) is activated.
        # Course selection: an explicit `course` wins (used by per-course eval); otherwise
        # random_courses picks one uniformly each episode (domain randomization for training);
        # otherwise the first course is reused.
        super().reset(seed=seed)
        chosen = self._resolve_course(course)
        if chosen is None:
            pool = self.sampling_courses or self.courses
            if self.random_courses and len(pool) > 1:
                # Roll a new lane only when the current block is exhausted (or the held lane
                # dropped out of the pool); otherwise reuse it for course_block_size tries.
                if (self._block_course is None
                        or self._block_resets_left <= 0
                        or self._block_course not in pool):
                    # Prefer a lane different from the one just finished so blocks rotate.
                    options = [c for c in pool if c is not self._block_course] or pool
                    self._block_course = random.choice(options)
                    self._block_resets_left = self.course_block_size
                chosen = self._block_course
                self._block_resets_left -= 1
            else:
                chosen = pool[0]
        self._apply_course(chosen)

        self.elapsed_steps = 0
        self.state_stack = np.zeros((self.stack_size, 14), dtype=np.float32)
        self.frame_stack = np.zeros(self.obs_shape["frame"], dtype=np.float32)

        sx, sy, sz = (float(v) for v in self.start_position)
        reset_action = {
            "command": f"tp @p {sx} {sy} {sz} {self.target_yaw} {self.target_pitch}",
            "forward": False,
            "back": False,
            "left": False,
            "right": False,
            "jump": False,
            "sprint": False,
            "yaw_delta": 0.0,
            "pitch_delta": 0.0,
        }
        self._send_action(reset_action)

        target_position = self.start_position
        packet = None
        current_position = None

        for i in range(RESET_WAIT_PACKETS):
            packet = self._wait_for_telemetry()
            current_position = self._position_from_packet(packet)
            if np.linalg.norm(current_position - target_position) <= RESET_POSITION_TOLERANCE:
                break
            # Player not back yet — re-send tp periodically. Robust to a dropped command
            # and to /tp not resetting fall velocity (player keeps falling otherwise).
            if i > 0 and i % RESET_TP_RESEND_EVERY == 0:
                self._send_action(reset_action)
        else:
            raise RuntimeError(
                f"Reset did not reach target position {target_position.tolist()}; "
                f"latest position was {current_position.tolist()}"
            )

        obs_mlp = self._obs_from_packet(packet)

        self.state_stack = np.repeat(obs_mlp[np.newaxis, :], self.stack_size, axis=0)
        frame = self._frame_from_packet(packet)
        self.frame_stack = np.concatenate([frame] * self.stack_size, axis=0)
        self.last_position = current_position
        self.last_distance_to_goal = np.linalg.norm(current_position - self.goal_position)
        # Track farthest-z platform landed on, for the per-platform bonus.
        self.furthest_landed_z = float(current_position[2])

        return self._build_obs(), {"packet": packet, "course": self.active_course.name}

    def step(self, actionid):
        # step() makes the environment play the game, calculate rewards, and choose actions until reset() is called.
        # Accepts either an int action_id (looked up in action_table) or a pre-built
        # action dict (used directly). Dict path is for the multi-head policy which
        # constructs its own key combinations instead of indexing into ACTION_TABLE.
        if isinstance(actionid, dict):
            action = actionid
        else:
            action = self.action_table[int(actionid)]
        # Remember whether this action moves the camera so _compute_reward can penalize it.
        self.last_action_is_camera = action.get("yaw_delta", 0.0) != 0.0 or action.get("pitch_delta", 0.0) != 0.0
        self._send_action(action)

        reward = 0.0
        packet = None

        for _ in range(self.action_repeat):
            packet = self._wait_for_telemetry()
            reward += self._compute_reward(packet)

        mlp_state = self._obs_from_packet(packet)
        self._update_state_stack(mlp_state)

        frame = self._frame_from_packet(packet)
        self._update_frame_stack(frame)

        self.elapsed_steps += 1
        terminated = self._is_terminated(packet)
        truncated = self.elapsed_steps >= self.max_steps

        info = {
            "packet": packet,
            "action": action,
        }

        return self._build_obs(), float(reward), terminated, truncated, info

    def close(self):
        # stops everything.
        self.writer.close()
        self.reader.close()
        self.socket.close()

    def _obs_from_packet(self, packet):
        # converts the mlp line in console into obs so that env can output numbers for train code
        return np.asarray(packet["mlp_state"], dtype=np.float32)

    def _position_from_packet(self, packet):
        pos = packet["position"]
        return np.array([pos["x"], pos["y"], pos["z"]], dtype=np.float32)

    def _update_state_stack(self, obs):
        # updates stack
        self.state_stack = np.concatenate([self.state_stack[1:], obs[np.newaxis, :]],
            axis=0,
        )
        return self.state_stack.copy()

    def _compute_reward(self, packet):
        current_position = self._position_from_packet(packet)
        current_distance = np.linalg.norm(current_position - self.goal_position)
        if self.last_position is None:
            delta = np.zeros(3, dtype=np.float32)
        else:
            delta = current_position - self.last_position

        reward = float(np.dot(delta, self.progress_dir) * self.progress_weight)

        lane_error = abs(float(current_position[0]) - self.center_x)
        reward -= lane_error * self.lane_penalty

        rotation = packet.get("rotation", {})
        yaw = float(rotation.get("yaw", self.target_yaw))
        pitch = float(rotation.get("pitch", self.target_pitch))
        yaw_error = abs(wrap_degrees(yaw - self.target_yaw)) / 180.0
        pitch_error = abs(pitch - self.target_pitch) / 90.0
        reward -= yaw_error * self.yaw_penalty
        reward -= pitch_error * self.pitch_penalty

        reward -= self.time_penalty

        # Per-platform landing bonus. Fires when the player is on the ground AND has reached
        # a z farther along the course than ever before in this episode. Gives PPO a clean
        # discrete credit signal ("you crossed a gap") on top of the continuous progress reward.
        if self.platform_reward > 0.0:
            mlp_state = packet.get("mlp_state")
            on_ground = bool(mlp_state[6]) if mlp_state and len(mlp_state) > 6 else False
            current_z = float(current_position[2])
            if on_ground and self.furthest_landed_z is not None and (
                current_z >= self.furthest_landed_z + self.platform_z_step
            ):
                reward += self.platform_reward
                self.furthest_landed_z = current_z

        # Optional flat penalty for choosing a camera (yaw/pitch) action at all. Off by default
        # (camera_action_penalty=0.0); applied per telemetry tick like the other penalties above.
        if self.last_action_is_camera:
            reward -= self.camera_action_penalty

        fell = current_position[1] < self.fall_y
        reached_goal = current_distance < self.goal_radius
        if fell:
            reward -= self.fall_penalty
        if reached_goal:
            reward += self.goal_bonus

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

    def _frame_from_packet(self, packet):
        frame = packet.get("frame")
        if frame is None:
            return np.zeros((FRAME_C, FRAME_H, FRAME_W), dtype=np.float32)

        raw = base64.b64decode(frame["data"])
        arr = np.frombuffer(raw, dtype=np.uint8).reshape(FRAME_H, FRAME_W, FRAME_C)
        # HWC -> CHW because pytorch expects channels first
        # divide by 255 to normalize pixel values to 0-1
        return arr.transpose(2, 0, 1).astype(np.float32) / 255.0

    def _update_frame_stack(self, frame):
        # drop oldest frame (3 channels), append new one
        self.frame_stack = np.concatenate([self.frame_stack[FRAME_C:], frame], axis=0)
        return self.frame_stack.copy()

    def _build_obs(self):
        return {
            "frame": self.frame_stack.copy(),
            "mlp": self.state_stack.copy(),
        }
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
        print(f"Environment reset. Frame shape: {obs['frame'].shape}, MLP shape: {obs['mlp'].shape}")
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
