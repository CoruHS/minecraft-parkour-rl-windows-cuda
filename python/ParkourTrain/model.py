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
    def __init__(self, num_actions, mlp_input_size=17, stack_size=4):
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
