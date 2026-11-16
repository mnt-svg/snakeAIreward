import random
from collections import deque

class SnakeAIEnvironment:
    def __init__(self, grid_size=10):
        self.grid_size = grid_size
        self.directions = [(-1, 0), (0, 1), (1, 0), (0, -1)] # 0: Lên, 1: Phải, 2: Xuống, 3: Trái
        self.reset()

    def reset(self):
        mid = self.grid_size // 2
        self.head = [mid, mid]
        self.body = [[mid, mid], [mid, mid - 1], [mid, mid - 2]]
        self.dir_idx = 1  # Mặc định đi sang phải
        self.food = self._spawn_food()
        self.score = 0
        self.steps_without_food = 0
        return self.get_state()

    def _spawn_food(self):
        while True:
            pos = [random.randint(0, self.grid_size - 1), random.randint(0, self.grid_size - 1)]
            if pos not in self.body:
                return pos

    def _is_collision(self, pt, body_to_check=None):
        if body_to_check is None:
            body_to_check = self.body
        return (
            pt[0] < 0 or pt[0] >= self.grid_size or
            pt[1] < 0 or pt[1] >= self.grid_size or
            pt in body_to_check
        )

    def _manhattan_dist(self, p1, p2):
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

    # Thuật toán BFS kiểm tra từ đầu rắn có còn đường tới đuôi hay không
    def _can_reach_tail(self, head_pos, body_list):
        if len(body_list) <= 2:
            return True
        tail = tuple(body_list[-1])
        obstacles = set(tuple(p) for p in body_list[:-1]) # Bỏ qua ô đuôi vì đuôi sẽ dịch chuyển
        
        queue = deque([tuple(head_pos)])
        visited = {tuple(head_pos)}
        
        while queue:
            curr = queue.popleft()
            if curr == tail:
                return True
            for dr, dc in self.directions:
                nxt = (curr[0] + dr, curr[1] + dc)
                if (0 <= nxt[0] < self.grid_size and 
                    0 <= nxt[1] < self.grid_size and 
                    nxt not in obstacles and 
                    nxt not in visited):
                    visited.add(nxt)
                    queue.append(nxt)
        return False

    # --- 1. STATE: "ĐÔI MẮT" CỦA AI (14 đặc trưng số) ---
    def get_state(self):
        # 3 hướng tương đối: [Đi thẳng, Rẽ phải, Rẽ trái]
        check_dirs = [
            self.dir_idx,
            (self.dir_idx + 1) % 4,
            (self.dir_idx - 1) % 4
        ]
        
        dangers = []
        reach_tail_flags = []
        
        for d in check_dirs:
            nxt_pt = [self.head[0] + self.directions[d][0], self.head[1] + self.directions[d][1]]
            is_danger = self._is_collision(nxt_pt)
            dangers.append(int(is_danger))
            
            if is_danger:
                reach_tail_flags.append(0)
            else:
                sim_body = [nxt_pt] + self.body[:-1]
                reach_tail_flags.append(int(self._can_reach_tail(nxt_pt, sim_body)))

        state = [
            # Nguy hiểm sát bên (3 bit)
            dangers[0], dangers[1], dangers[2],
            # Hướng di chuyển hiện tại (4 bit)
            int(self.dir_idx == 3), int(self.dir_idx == 1),
            int(self.dir_idx == 0), int(self.dir_idx == 2),
            # Vị trí quả so với đầu (4 bit)
            int(self.food[1] < self.head[1]), int(self.food[1] > self.head[1]),
            int(self.food[0] < self.head[0]), int(self.food[0] > self.head[0]),
            # Duy trì đường đi tới đuôi (3 bit)
            reach_tail_flags[0], reach_tail_flags[1], reach_tail_flags[2]
        ]
        return state

    # --- 2. REWARD FUNCTION: MÔ TẢ MỤC TIÊU CHO AI ---
    def compute_reward(self, is_dead, ate_food, is_looping, has_path_to_tail, old_dist, new_dist):
        if is_dead:
            return -10.0      # Phạt nặng: Đâm tường hoặc cắn thân
        if is_looping:
            return -5.0       # Phạt câu giờ: Đi lòng vòng vô nghĩa
        if ate_food:
            return 10.0       # Thưởng lớn: Đạt mục tiêu ăn quả
        if not has_path_to_tail:
            return -3.0       # Phạt chiến thuật: Tự nhốt mình vào ngõ cụt (mất đường tới đuôi)
        if new_dist < old_dist:
            return 0.1        # Thưởng định hướng: Tiến lại gần quả
        return -0.15          # Phạt nhẹ: Đi ra xa quả

    # --- 3. ACTION & VÒNG LẶP TƯƠNG TÁC ---
    def step(self, action):
        # action: 0 = Đi thẳng, 1 = Rẽ phải, 2 = Rẽ trái
        if action == 1:
            self.dir_idx = (self.dir_idx + 1) % 4
        elif action == 2:
            self.dir_idx = (self.dir_idx - 1) % 4

        old_dist = self._manhattan_dist(self.head, self.food)
        new_head = [self.head[0] + self.directions[self.dir_idx][0],
                    self.head[1] + self.directions[self.dir_idx][1]]
        self.steps_without_food += 1

        is_dead = self._is_collision(new_head)
        ate_food = (new_head == self.food)
        is_looping = (self.steps_without_food > self.grid_size * self.grid_size)
        new_dist = self._manhattan_dist(new_head, self.food)

        done = is_dead or is_looping
        has_path_to_tail = True

        if not done:
            self.head = new_head
            self.body.insert(0, list(new_head))
            if ate_food:
                self.score += 1
                self.steps_without_food = 0
                self.food = self._spawn_food()
            else:
                self.body.pop()
            has_path_to_tail = self._can_reach_tail(self.head, self.body)

        reward = self.compute_reward(is_dead, ate_food, is_looping, has_path_to_tail, old_dist, new_dist)
        return self.get_state(), reward, done, self.score

if __name__ == "__main__":
    env = SnakeAIEnvironment(grid_size=10)
    state = env.reset()
    total_reward = 0
    print(f"State khởi tạo (14 features): {state}")
    
    for step in range(100):
        action = random.choice([0, 1, 2]) # Thử nghiệm hành động ngẫu nhiên
        state, reward, done, score = env.step(action)
        total_reward += reward
        if done:
            print(f"Kết thúc ở bước {step + 1} | Điểm ăn quả: {score} | Tổng Reward: {total_reward:.2f}")
            break
