import random
import pygame
import pickle
import os
from snakeAIenv import SnakeAIEnvironment

CELL_SIZE = 40
BG_COLOR = (25, 25, 35)
GRID_COLOR = (40, 40, 55)
HEAD_COLOR = (0, 230, 118)
BODY_COLOR = (0, 180, 90)
FOOD_COLOR = (255, 82, 82)
TEXT_COLOR = (255, 255, 255)

class QLearningAgent:
    def __init__(self, alpha=0.1, gamma=0.9, epsilon=1.0, brain_file="snake_brain.pkl"):
        self.q_table = {}
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.brain_file = brain_file
        self.total_episodes_trained = 0
        self.best_avg_score = 0.0  # Lưu kỷ lục điểm trung bình cao nhất
        self.load_brain()

    def get_q_values(self, state):
        state_key = tuple(state)
        if state_key not in self.q_table:
            self.q_table[state_key] = [0.0, 0.0, 0.0]
        return self.q_table[state_key]

    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.choice([0, 1, 2])
        q_vals = self.get_q_values(state)
        return q_vals.index(max(q_vals))

    def learn(self, state, action, reward, next_state, done):
        q_vals = self.get_q_values(state)
        next_max = 0.0 if done else max(self.get_q_values(next_state))
        q_vals[action] += self.alpha * (reward + self.gamma * next_max - q_vals[action])

    def save_brain(self, current_avg_score):
        # Chỉ lưu nếu đợt train này đạt điểm trung bình tốt hơn hoặc bằng kỷ lục cũ
        if current_avg_score >= self.best_avg_score:
            self.best_avg_score = current_avg_score
            data = {
                "q_table": self.q_table,
                "epsilon": self.epsilon,
                "alpha": self.alpha,
                "total_episodes": self.total_episodes_trained,
                "best_avg_score": self.best_avg_score
            }
            with open(self.brain_file, "wb") as f:
                pickle.dump(data, f)
            print(f"[KỶ LỤC MỚI] Điểm TB: {current_avg_score:.2f} -> Đã lưu bộ não ({self.total_episodes_trained} ván).")
        else:
            print(f"[BỎ QUA] Điểm TB đợt này ({current_avg_score:.2f}) thấp hơn kỷ lục ({self.best_avg_score:.2f}), giữ nguyên não cũ.")

    def load_brain(self):
        if os.path.exists(self.brain_file):
            with open(self.brain_file, "rb") as f:
                data = pickle.load(f)
            self.q_table = data["q_table"]
            self.epsilon = data["epsilon"]
            self.alpha = data.get("alpha", 0.05)
            self.total_episodes_trained = data.get("total_episodes", 0)
            self.best_avg_score = data.get("best_avg_score", 0.0)
            print(f"Đã nạp ký ức cũ: {self.total_episodes_trained} ván | Kỷ lục TB: {self.best_avg_score:.2f}")
        else:
            print("Chưa có dữ liệu cũ, bắt đầu huấn luyện từ con số 0.")

def draw_game(screen, font, env, total_reward, game_num):
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

    info_text = font.render(f"Van: {game_num} | Diem: {env.score} | Reward: {total_reward:.1f}", True, TEXT_COLOR)
    screen.blit(info_text, (10, 10))
    pygame.display.flip()

if __name__ == "__main__":
    env = SnakeAIEnvironment(grid_size=12)
    agent = QLearningAgent()

    episodes = 3000
    recent_scores = []
    print(f"Đang huấn luyện {episodes} ván...")

    for ep in range(episodes):
        state = env.reset()
        while True:
            action = agent.choose_action(state)
            next_state, reward, done, score = env.step(action)
            agent.learn(state, action, reward, next_state, done)
            state = next_state
            if done:
                break
        
        agent.epsilon = max(0.01, agent.epsilon * 0.997)
        agent.alpha = max(0.01, agent.alpha * 0.999)  # Giảm dần tốc độ học để ổn định kinh nghiệm
        agent.total_episodes_trained += 1

        # Theo dõi điểm 200 ván cuối cùng của đợt train để đánh giá chất lượng
        if ep >= episodes - 200:
            recent_scores.append(score)

    avg_score = sum(recent_scores) / len(recent_scores)
    agent.save_brain(avg_score)
    
    # Nạp lại bản tốt nhất để trình diễn (tránh trường hợp đợt vừa rồi bị tụt điểm)
    agent.load_brain()

    pygame.init()
    window_size = env.grid_size * CELL_SIZE
    screen = pygame.display.set_mode((window_size, window_size))
    pygame.display.set_caption("Snake AI - Q-Learning Demo")
    font = pygame.font.SysFont("Arial", 18, bold=True)
    clock = pygame.time.Clock()

    agent.epsilon = 0
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

        draw_game(screen, font, env, total_reward, game_num)
        clock.tick(15)

        if done:
            pygame.time.wait(600)
            state = env.reset()
            total_reward = 0
            game_num += 1

    pygame.quit()
