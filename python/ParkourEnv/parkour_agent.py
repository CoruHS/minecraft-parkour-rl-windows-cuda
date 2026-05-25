import json
import sys
import time

MARKER = "RL_TELEMETRY "

STATE_NAMES = [
    "yaw_sin", "yaw_cos", "pitch_norm",
    "velocity_x", "velocity_y", "velocity_z",
    "on_ground", "sprinting",
    "forward_pressed", "back_pressed", "left_pressed", "right_pressed",
    "jump_pressed", "sprint_key_pressed",
]


def parse_line(line):
    # skip lines that aren't telemetry
    if MARKER not in line:
        return None

    # grab everything after "RL_TELEMETRY " and parse as JSON
    json_part = line.split(MARKER, 1)[1]
    return json.loads(json_part)


def handle_packet(packet):
    # pull out the pieces we care about
    movement = packet.get("movement", {})
    yaw = packet.get("rotation", {}).get("yaw", 0.0)
    pitch = packet.get("rotation", {}).get("pitch", 0.0)
    sprinting = packet.get("sprinting", False)
    mlp_state = packet.get("mlp_state")

    print(f"movement={movement}  yaw={yaw:.2f}  pitch={pitch:.2f}  sprinting={sprinting}")

    # print the 14 neural network inputs with labels
    if mlp_state is not None:
        for name, value in zip(STATE_NAMES, mlp_state):
            print(f"  {name} = {value:.4f}")


# --- main script ---

if len(sys.argv) != 2:
    print("Usage: python3 parkour_agent.py <path-to-latest.log>")
    sys.exit(1)

log_path = sys.argv[1]
log_file = open(log_path, "r")
log_file.seek(0, 2)  # skip to end of file, only read new lines

while True:
    line = log_file.readline()

    if not line:
        time.sleep(0.05)  # nothing new, wait and retry
        continue

    packet = parse_line(line)
    if packet is not None:
        handle_packet(packet)