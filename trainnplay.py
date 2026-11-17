import os
import random
import pygame
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from snakeAIenv import SnakeAIEnvironment

# Cấu hình giao diện Pygame
CELL_SIZE = 40
BG_COLOR = (25, 25, 35)
GRID_COLOR = (40, 40, 55)
HEAD_COLOR = (0, 230, 118)
BODY_COLOR = (0, 180, 90)
FOOD_COLOR = (255, 82, 82)
TEXT_COLOR = (255, 255, 255)

class LinearQNet(nn.Module):
    def __init__(self, input_size=28, hidden_size=256, output_size=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, output_size)
        )

    def forward(self, x):
        return self.net(x)

class DQNAgent:
    def __init__(self, lr=0.001, gamma=0.95, model_file="snake_dqn.pth"):
        self.gamma = gamma
        self.epsilon = 1.0
        self.memory = deque(maxlen=50_000)
        self.model = LinearQNet(28, 256, 3)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
        self.model_file = model_file
        self.record = 0
        self.total_episodes = 0
        self.load_checkpoint()

    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, 2)
        state_t = torch.tensor(state, dtype=torch.float32)
        with torch.no_grad():
            q_vals = self.model(state_t)
        return torch.argmax(q_vals).item()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_batch(self, batch_size=256):
        if len(self.memory) < batch_size:
            return
        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states_t = torch.tensor(states, dtype=torch.float32)
        actions_t = torch.tensor(actions, dtype=torch.long).unsqueeze(1)
        rewards_t = torch.tensor(rewards, dtype=torch.float32)
        next_states_t = torch.tensor(next_states, dtype=torch.float32)
        dones_t = torch.tensor(dones, dtype=torch.bool)

        current_q = self.model(states_t).gather(1, actions_t).squeeze(1)
        with torch.no_grad():
            max_next_q = self.model(next_states_t).max(1)[0]
            max_next_q[dones_t] = 0.0
            target_q = rewards_t + self.gamma * max_next_q

        loss = self.criterion(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def save_checkpoint(self, score):
        if score >= self.record:
            self.record = score
            torch.save({
                "model_state": self.model.state_dict(),
                "epsilon": self.epsilon,
                "record": self.record,
                "total_episodes": self.total_episodes
            }, self.model_file)

    def load_checkpoint(self):
        if os.path.exists(self.model_file):
            checkpoint = torch.load(self.model_file, weights_only=True)
            self.model.load_state_dict(checkpoint["model_state"])
            self.epsilon = checkpoint.get("epsilon", 0.05)
            self.record = checkpoint.get("record", 0)
            self.total_episodes = checkpoint.get("total_episodes", 0)
            print(f"Đã nạp ký ức cũ: {self.total_episodes} ván | Kỷ lục: {self.record} điểm | Epsilon: {self.epsilon:.3f}")
        else:
            print("Chưa có file lưu cũ, bắt đầu huấn luyện từ đầu.")

def draw_game(screen, font, env, total_reward, game_num, record):
    screen.fill(BG_COLOR)
    size = env.grid_size
    for x in range(0, size * CELL_SIZE, CELL_SIZE):
        pygame.draw.line(screen, GRID_COLOR, (x, 0), (x, size * CELL_SIZE))
    for y in range(0, size * CELL_SIZE, CELL_SIZE):
        pygame.draw.line(screen, GRID_COLOR, (0, y), (size * CELL_SIZE, y))

    food_center = (env.food[1] * CELL_SIZE + CELL_SIZE // 2, env.food[0] * CELL_SIZE + CELL_SIZE // 2)
    pygame.draw.circle(screen, FOOD_COLOR, food_center, CELL_SIZE // 2 - 4)

    for idx, part in enumerate(env.body):
        rect = pygame.Rect(part[1] * CELL_SIZE + 2, part[0] * CELL_SIZE + 2, CELL_SIZE - 4, CELL_SIZE - 4)
        color = HEAD_COLOR if idx == 0 else BODY_COLOR
        pygame.draw.rect(screen, color, rect, border_radius=8)

    info_text = font.render(f"Van: {game_num} | Diem: {env.score} (Ky luc: {record}) | Reward: {total_reward:.1f}", True, TEXT_COLOR)
    screen.blit(info_text, (10, 10))
    pygame.display.flip()

if __name__ == "__main__":
    env = SnakeAIEnvironment(grid_size=12)
    agent = DQNAgent()

    # BƯỚC 1: Huấn luyện 800 ván (tự động cộng dồn nếu chạy nhiều lần)
    episodes = 800
    print(f"Đang huấn luyện tiếp {episodes} ván...")

    for ep in range(episodes):
        state = env.reset()
        while True:
            action = agent.choose_action(state)
            next_state, reward, done, score = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            if done:
                break
        
        agent.train_batch(batch_size=256)
        agent.epsilon = max(0.01, agent.epsilon * 0.993)
        agent.total_episodes += 1
        agent.save_checkpoint(score)

        if (ep + 1) % 100 == 0:
            print(f"Tiến độ: {ep + 1}/{episodes} (Tổng: {agent.total_episodes} ván) | Kỷ lục: {agent.record} | Epsilon: {agent.epsilon:.2f}")

    # Nạp lại model đạt kỷ lục tốt nhất để trình diễn
    agent.load_checkpoint()
    agent.epsilon = 0

    # BƯỚC 2: Mở cửa sổ Pygame trình diễn
    pygame.init()
    window_size = env.grid_size * CELL_SIZE
    screen = pygame.display.set_mode((window_size, window_size))
    pygame.display.set_caption("Snake AI - Full Map Raycasting DQN")
    font = pygame.font.SysFont("Arial", 16, bold=True)
    clock = pygame.time.Clock()

    state = env.reset()
    total_reward = 0
    game_num = 1
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        action = agent.choose_action(state)
        state, reward, done, score = env.step(action)
        total_reward += reward

        draw_game(screen, font, env, total_reward, game_num, agent.record)
        clock.tick(18)

        if done:
            pygame.time.wait(500)
            state = env.reset()
            total_reward = 0
            game_num += 1

    pygame.quit()
