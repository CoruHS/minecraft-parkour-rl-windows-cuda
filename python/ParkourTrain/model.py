import torch
import torch.nn as nn


class ParkourMLP(nn.Module):
    def __init__(self, input_size, num_actions):
        super().__init__()

        self.backbone = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )
        self.actor = nn.Linear(256, num_actions)
        self.critic = nn.Linear(256, 1)

    def forward(self, x):
        features = self.backbone(x)
        action_logits = self.actor(features)
        value = self.critic(features)
        return action_logits, value

    def get_action(self, state):
        with torch.no_grad():
            device = next(self.parameters()).device
            state_tensor = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            logits, value = self.forward(state_tensor)
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()
            return action.item(), dist.log_prob(action).squeeze(), value.squeeze()

class ParkourCNN(nn.Module):
    def __init__(self, num_actions, mlp_input_size=14, stack_size=4):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(stack_size * 3, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        self.mlp_branch = nn.Sequential(
            nn.Flatten(),
            nn.Linear(stack_size * mlp_input_size, 128),
            nn.ReLU(),
        )

        combined_size = 3136 + 128
        self.shared = nn.Sequential(
            nn.Linear(combined_size, 256),
            nn.ReLU(),
        )

        self.actor = nn.Linear(256, num_actions)
        self.critic = nn.Linear(256, 1)

    def forward(self, frame, mlp_state):
        cnn_features = self.cnn(frame)
        mlp_features = self.mlp_branch(mlp_state)
        combined = torch.cat([cnn_features, mlp_features], dim=-1)
        shared = self.shared(combined)
        return self.actor(shared), self.critic(shared)

    def get_action(self, state):
        with torch.no_grad():
            device = next(self.parameters()).device
            frame_t = torch.as_tensor(state["frame"], dtype=torch.float32, device=device).unsqueeze(0)
            mlp_t = torch.as_tensor(state["mlp"], dtype=torch.float32, device=device).unsqueeze(0)
            logits, value = self.forward(frame_t, mlp_t)
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()
            log_prob = dist.log_prob(action).squeeze()
        return action.item(), log_prob, value.squeeze()


class ParkourCNNMultiHead(nn.Module):
    """Multi-head policy: independent Bernoulli heads for W, Jump, and Sprint.

    Lets the agent learn "hold W", "time the jump", and "hold sprint (ctrl)"
    independently, instead of picking from pre-baked key combos in ACTION_TABLE.
    The sprint head is what lets one policy span courses of different gap sizes:
    walk-jump clears ~1-block gaps, sprint-jump clears ~3-block gaps, so the agent
    can pick jump distance per gap from the pixels. For straight forward parkour;
    for diagonals/camera, extend with extra heads (yaw/pitch) using the same pattern.

    Old 2-head checkpoints (W+Jump only) load via load_state_dict(strict=False):
    cnn/mlp/shared/w_head/jump_head/critic transfer, sprint_head starts fresh.
    """

    def __init__(self, mlp_input_size=14, stack_size=4):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(stack_size * 3, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        self.mlp_branch = nn.Sequential(
            nn.Flatten(),
            nn.Linear(stack_size * mlp_input_size, 128),
            nn.ReLU(),
        )

        combined_size = 3136 + 128
        self.shared = nn.Sequential(
            nn.Linear(combined_size, 256),
            nn.ReLU(),
        )

        self.w_head = nn.Linear(256, 1)
        self.jump_head = nn.Linear(256, 1)
        self.sprint_head = nn.Linear(256, 1)
        self.critic = nn.Linear(256, 1)

    def forward(self, frame, mlp_state):
        cnn_features = self.cnn(frame)
        mlp_features = self.mlp_branch(mlp_state)
        combined = torch.cat([cnn_features, mlp_features], dim=-1)
        shared = self.shared(combined)
        w_logits = self.w_head(shared).squeeze(-1)
        jump_logits = self.jump_head(shared).squeeze(-1)
        sprint_logits = self.sprint_head(shared).squeeze(-1)
        value = self.critic(shared)
        return w_logits, jump_logits, sprint_logits, value

    def get_action(self, state):
        with torch.no_grad():
            device = next(self.parameters()).device
            frame_t = torch.as_tensor(state["frame"], dtype=torch.float32, device=device).unsqueeze(0)
            mlp_t = torch.as_tensor(state["mlp"], dtype=torch.float32, device=device).unsqueeze(0)
            w_logits, jump_logits, sprint_logits, value = self.forward(frame_t, mlp_t)
            w_dist = torch.distributions.Bernoulli(logits=w_logits)
            jump_dist = torch.distributions.Bernoulli(logits=jump_logits)
            sprint_dist = torch.distributions.Bernoulli(logits=sprint_logits)
            w_action = w_dist.sample()
            jump_action = jump_dist.sample()
            sprint_action = sprint_dist.sample()
            log_prob = (
                w_dist.log_prob(w_action)
                + jump_dist.log_prob(jump_action)
                + sprint_dist.log_prob(sprint_action)
            ).squeeze()
            action = (
                int(w_action.item()),
                int(jump_action.item()),
                int(sprint_action.item()),
            )
        return action, log_prob, value.squeeze()
