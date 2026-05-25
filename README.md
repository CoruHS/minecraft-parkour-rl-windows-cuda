# Minecraft Parkour RL

Fabric 1.20.1 client mod plus Python PPO environments for training a Minecraft parkour agent.

The Java client opens a local socket on `127.0.0.1:5005`, accepts Python action commands, and sends telemetry every tick. The telemetry includes a 14-value MLP state and an optional `84x84` RGB frame for CNN training.

## What is included

- Fabric Minecraft client mod for 1.20.1.
- Socket bridge between Minecraft and Python.
- MLP-only environment/training/testing path.
- CNN+MLP environment/training/testing path.
- Client/server tickrate control for faster training.
- Auto-open support for a singleplayer world named `ParkourRL`.
- Auto-open-to-LAN support so another Minecraft client can watch training.

## Requirements

- Java 17
- Python 3.10+
- Minecraft/Fabric dependencies downloaded by Gradle
- Python packages from `requirements.txt`

Install Python dependencies:

```bash
python3 -m pip install -r requirements.txt
```

Build the mod:

```bash
./gradlew build
```

## World Setup

Create a singleplayer world once with this exact folder/name:

```text
ParkourRL
```

The save should exist at:

```text
run/saves/ParkourRL
```

The save folder is intentionally ignored by Git, so each computer needs its own world.

## Run The CNN Environment

This starts Minecraft if needed, waits for the socket, resets the player, and runs random actions:

```bash
python3 python/ParkourEnv/env_cnn.py
```

Expected observation shapes with the default `stack_size=4`:

```text
frame: (12, 84, 84)
mlp:   (4, 14)
```

## Train CNN + MLP

```bash
python3 python/ParkourTrain/train_cnn.py
```

CNN checkpoints are written to:

```text
checkpoints_cnn/
```

Training logs are written to:

```text
training_logs/episodes_cnn.csv
```

Both are ignored by Git.

## Test A CNN Checkpoint

```bash
python3 python/ParkourTest/test_cnn.py
```

Or provide a checkpoint explicitly:

```bash
python3 python/ParkourTest/test_cnn.py --checkpoint checkpoints_cnn/latest.pt
```

## MLP-Only Path

Random MLP environment:

```bash
python3 python/ParkourEnv/env.py
```

Train MLP:

```bash
python3 python/ParkourTrain/train.py
```

Test MLP:

```bash
python3 python/ParkourTest/test.py
```

## Telemetry State

The `mlp_state` vector has 14 values:

```text
yaw_sin, yaw_cos, pitch_norm,
velocity_x, velocity_y, velocity_z,
on_ground, sprinting,
forward_pressed, back_pressed, left_pressed, right_pressed,
jump_pressed, sprint_key_pressed
```

The CNN frame telemetry is sent as:

```json
{
  "width": 84,
  "height": 84,
  "channels": 3,
  "format": "rgb_u8_base64",
  "data": "..."
}
```

## Tickrate

The Java tickrate is configured in:

```text
src/client/java/coru/minecraftparkourrl/client/MinecraftParkourRLClient.java
```

Look for:

```java
private static final float TICKRATE = 40.0F;
```

For CNN training, `40 TPS` is a good starting point. If training is stable, try `60 TPS`.
