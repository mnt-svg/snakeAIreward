import random
import time
import os
from snakeAIenv import SnakeAIEnvironment  # Gọi môi trường từ file đầu tiên

class QLearningAgent:
    def __init__(self, alpha=0.1, gamma=0.9, epsilon=1.0):
        self.q_table = {}
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    def get_q_values(self, state):
        state_key = tuple(state)  # Chuyển list 14 số thành tuple để lưu vào từ điển
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

def render_board(env, episodes, total_reward):
    os.system('cls' if os.name == 'nt' else 'clear')
    print(f"=== AI ĐANG CHƠI THỬ (Sau khi train {episodes} ván) ===")
    print(f"Điểm ăn quả: {env.score} | Tổng Reward: {total_reward:.2f}")
    print("+" + "--" * env.grid_size + "+")
    for r in range(env.grid_size):
        row_str = "|"
        for c in range(env.grid_size):
            if [r, c] == env.head:
                row_str += "Ö "
            elif [r, c] in env.body:
                row_str += "o "
            elif [r, c] == env.food:
                row_str += "🍎"
            else:
                row_str += ". "
        print(row_str + "|")
    print("+" + "--" * env.grid_size + "+")

if __name__ == "__main__":
    env = SnakeAIEnvironment(grid_size=10)
    agent = QLearningAgent()

    # 1. Huấn luyện ngầm 1.500 ván
    episodes = 1500
    print("Đang huấn luyện AI...")
    for ep in range(episodes):
        state = env.reset()
        while True:
            action = agent.choose_action(state)
            next_state, reward, done, score = env.step(action)
            agent.learn(state, action, reward, next_state, done)
            state = next_state
            if done:
                break
        agent.epsilon = max(0.01, agent.epsilon * 0.995)

    print(f"Train xong! Đã học {len(agent.q_table)} trạng thái.")
    time.sleep(1.5)

    # 2. AI tự chơi trình diễn
    agent.epsilon = 0
    state = env.reset()
    total_reward = 0
    while True:
        render_board(env, episodes, total_reward)
        time.sleep(0.15)
        action = agent.choose_action(state)
        state, reward, done, score = env.step(action)
        total_reward += reward
        if done:
            render_board(env, episodes, total_reward)
            print(f"KẾT THÚC! AI đạt được {score} điểm.")
            break
