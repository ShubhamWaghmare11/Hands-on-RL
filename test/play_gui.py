import pygame
import numpy as np
import torch
import torch.nn.functional as F

# =========================
# CONFIG
# =========================
GRID_SIZE = 5
CELL_SIZE = 100
WIDTH = GRID_SIZE * CELL_SIZE
HEIGHT = GRID_SIZE * CELL_SIZE

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)      # agent
BLUE = (0, 0, 255)     # player

# =========================
# ENV (same logic)
# =========================
class CatchEnv:
    def __init__(self):
        self.grid_size = GRID_SIZE
        self.reset()

    def reset(self):
        while True:
            self.player_pos = np.random.randint(0, self.grid_size, size=2)
            self.agent_pos = np.random.randint(0, self.grid_size, size=2)

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

    def move(self, pos, action):
        x, y = pos

        if action == 0: x -= 1
        elif action == 1: x += 1
        elif action == 2: y -= 1
        elif action == 3: y += 1

        x = np.clip(x, 0, self.grid_size - 1)
        y = np.clip(y, 0, self.grid_size - 1)

        return np.array([x, y])

    def step(self, agent_action, player_action):
        self.agent_pos = self.move(self.agent_pos, agent_action)
        self.player_pos = self.move(self.player_pos, player_action)

        done = np.array_equal(self.agent_pos, self.player_pos)
        return self._get_state(), done


# =========================
# LOAD YOUR TRAINED MODEL
# =========================
class PGN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(4, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 5)
        )

    def forward(self, x):
        return self.net(x)


def load_model(path="model.pth"):
    net = PGN()
    net.load_state_dict(torch.load(path))
    net.eval()
    return net
def play():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("RL Catch Game")

    clock = pygame.time.Clock()

    env = CatchEnv()
    state = env.reset()

    net = load_model()

    running = True

    while running:
        screen.fill(WHITE)

        # --- DRAW GRID ---
        for x in range(GRID_SIZE):
            for y in range(GRID_SIZE):
                rect = pygame.Rect(y * CELL_SIZE, x * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                pygame.draw.rect(screen, BLACK, rect, 1)

        # --- DRAW PLAYER ---
        px, py = env.player_pos
        pygame.draw.rect(screen, BLUE,
                         (py * CELL_SIZE, px * CELL_SIZE, CELL_SIZE, CELL_SIZE))

        # --- DRAW AGENT ---
        ax, ay = env.agent_pos
        pygame.draw.rect(screen, RED,
                         (ay * CELL_SIZE, ax * CELL_SIZE, CELL_SIZE, CELL_SIZE))

        pygame.display.flip()

        # --- WAIT FOR PLAYER INPUT ---
        player_action = None

        while player_action is None:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return

                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_UP:
                        player_action = 0
                    elif event.key == pygame.K_DOWN:
                        player_action = 1
                    elif event.key == pygame.K_LEFT:
                        player_action = 2
                    elif event.key == pygame.K_RIGHT:
                        player_action = 3

        # --- AGENT DECISION ---
        state_t = torch.tensor(state, dtype=torch.float32)
        logits = net(state_t)
        probs = F.softmax(logits, dim=0)
        agent_action = torch.argmax(probs).item()

        # --- APPLY BOTH MOVES ---
        state, done = env.step(agent_action, player_action)

        if done:
            print("💀 Agent caught you!")
            pygame.time.delay(1000)
            state = env.reset()

        clock.tick(30)
if __name__ == "__main__":
    play()