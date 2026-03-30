import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from itertools import count

# =========================
# Environment
# =========================
class CatchEnv:
    def __init__(self, grid_size=5):
        self.grid_size = grid_size
        self.action_space = 5  # up, down, left, right, stay
        self.reset()

    def reset(self):
        while True:
            self.player_pos = np.random.randint(0, self.grid_size, size=2)
            self.agent_pos = np.random.randint(0, self.grid_size, size=2)

            # ensure minimum distance
            if np.linalg.norm(self.player_pos - self.agent_pos) > 2:
                break

        return self._get_state()
    def _get_state(self):
        return np.array([
            self.player_pos[0],
            self.player_pos[1],
            self.agent_pos[0],
            self.agent_pos[1]
        ], dtype=np.float32)

    def step(self, agent_action):
        self.agent_pos = self._move(self.agent_pos, agent_action)

        # heuristic player
        player_action = self._heuristic_player()
        self.player_pos = self._move(self.player_pos, player_action)

        done = np.array_equal(self.agent_pos, self.player_pos)

        reward = 10.0 if done else -0.1

        return self._get_state(), reward, done

    def _move(self, pos, action):
        x, y = pos

        if action == 0: x -= 1
        elif action == 1: x += 1
        elif action == 2: y -= 1
        elif action == 3: y += 1
        elif action == 4: pass

        x = np.clip(x, 0, self.grid_size - 1)
        y = np.clip(y, 0, self.grid_size - 1)

        return np.array([x, y])

    def _heuristic_player(self):
        if np.random.rand() < 0.3:
            return np.random.randint(0, 5)

        dx = self.player_pos[0] - self.agent_pos[0]
        dy = self.player_pos[1] - self.agent_pos[1]

        if abs(dx) > abs(dy):
            return 0 if dx > 0 else 1
        else:
            return 2 if dy > 0 else 3
    def render(self):
        grid = np.full((self.grid_size, self.grid_size), ".")
        px, py = self.player_pos
        ax, ay = self.agent_pos
        grid[px, py] = "P"
        grid[ax, ay] = "A"

        print("\n".join(" ".join(row) for row in grid))
        print()


# =========================
# Policy Network
# =========================
class PGN(nn.Module):
    def __init__(self, input_size, n_actions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions)
        )

    def forward(self, x):
        return self.net(x)


# =========================
# Compute returns + baseline
# =========================
GAMMA = 0.99

def calc_advantages(rewards):
    res = []
    sum_r = 0.0

    for r in reversed(rewards):
        sum_r = r + GAMMA * sum_r
        res.append(sum_r)

    res = list(reversed(res))
    baseline = np.mean(res)

    return [r - baseline for r in res]


# =========================
# Training
# =========================
LEARNING_RATE = 0.01
EPISODES_TO_TRAIN = 4

def train():
    env = CatchEnv()
    net = PGN(4, 5)
    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)

    total_rewards = []
    batch_states, batch_actions, batch_adv = [], [], []
    batch_episodes = 0

    episode = 0

    while True:
        state = env.reset()

        episode_states = []
        episode_actions = []
        episode_rewards = []

        # === PLAY ONE EPISODE ===
        while True:
            state_t = torch.tensor(state, dtype=torch.float32)

            logits = net(state_t)
            probs = F.softmax(logits, dim=0)

            action = torch.multinomial(probs, 1).item()

            next_state, reward, done = env.step(action)

            episode_states.append(state)
            episode_actions.append(action)
            episode_rewards.append(reward)

            state = next_state

            if done:
                break

        # === PROCESS EPISODE ===
        total_reward = sum(episode_rewards)
        advantages = calc_advantages(episode_rewards)

        batch_states.extend(episode_states)
        batch_actions.extend(episode_actions)
        batch_adv.extend(advantages)

        batch_episodes += 1
        total_rewards.append(total_reward)

        # === PRINT STATS ===
        mean_reward = np.mean(total_rewards[-50:])

        print(
            f"Episode {episode:4d} | "
            f"Reward: {total_reward:6.2f} | "
            f"Mean(50): {mean_reward:6.2f}"
        )

        episode += 1

        # === TRAIN ===
        if batch_episodes >= EPISODES_TO_TRAIN:
            states_t = torch.tensor(batch_states, dtype=torch.float32)
            actions_t = torch.tensor(batch_actions, dtype=torch.int64)
            adv_t = torch.tensor(batch_adv, dtype=torch.float32)

            optimizer.zero_grad()

            logits = net(states_t)
            log_probs = F.log_softmax(logits, dim=1)

            selected_log_probs = log_probs[range(len(actions_t)), actions_t]

            loss = -(adv_t * selected_log_probs).mean()

            loss.backward()
            optimizer.step()

            batch_states.clear()
            batch_actions.clear()
            batch_adv.clear()
            batch_episodes = 0

        # === STOP CONDITION ===
        if episode > 100 and mean_reward > 8:  # agent catching quickly
            print("\n🔥 Agent learned good policy!\n")
            break

    return net

# =========================
# Play (Watch agent)
# =========================
def play(net):
    env = CatchEnv()

    state = env.reset()

    for _ in range(50):
        env.render()

        state_t = torch.tensor(state, dtype=torch.float32)
        logits = net(state_t)
        probs = F.softmax(logits, dim=0)

        action = torch.argmax(probs).item()  # greedy for demo

        state, _, done = env.step(action)

        if done:
            print("Agent caught player!")
            env.render()
            break


# =========================
# Main
# =========================
if __name__ == "__main__":
    trained_net = train()

    # 🔥 SAVE MODEL
    torch.save(trained_net.state_dict(), "model.pth")
    print("\n✅ Model saved as model.pth")

    print("\n--- Playing with trained agent ---\n")
    play(trained_net)