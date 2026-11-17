import random
from collections import deque

class SnakeAIEnvironment:
    def __init__(self, grid_size=12):
        self.grid_size = grid_size
        # 8 hướng quét theo chiều kim đồng hồ bắt đầu từ Hướng Lên (0)
        self.all_8_dirs = [
            (-1, 0),  # 0: Lên
            (-1, 1),  # 1: Chéo Lên-Phải
            (0, 1),   # 2: Phải
            (1, 1),   # 3: Chéo Xuống-Phải
            (1, 0),   # 4: Xuống
            (1, -1),  # 5: Chéo Xuống-Trái
            (0, -1),  # 6: Trái
            (-1, -1)  # 7: Chéo Lên-Trái
        ]
        # 4 hướng di chuyển chính: Lên (0), Phải (1), Xuống (2), Trái (3)
        self.directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        self.reset()

    def reset(self):
        mid = self.grid_size // 2
        self.head = [mid, mid]
        self.body = [[mid, mid], [mid, mid - 1], [mid, mid - 2]]
        self.dir_idx = 1
        self.food = self._spawn_food()
        self.score = 0
        self.steps_without_food = 0
        return self.get_state()

    def _spawn_food(self):
        while True:
            pos = [random.randint(0, self.grid_size - 1), random.randint(0, self.grid_size - 1)]
            if pos not in self.body:
                return pos

    def _is_collision(self, pt):
        return (
            pt[0] < 0 or pt[0] >= self.grid_size or
            pt[1] < 0 or pt[1] >= self.grid_size or
            pt in self.body
        )

    def _manhattan_dist(self, p1, p2):
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

    def _can_reach_tail(self, head_pos, body_list):
        if len(body_list) <= 2:
            return True
        tail = tuple(body_list[-1])
        obstacles = set(tuple(p) for p in body_list[:-1])
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

    # TẦM NHÌN TOÀN BẢN ĐỒ: Quét 8 tia xuyên suốt bản đồ (28 đặc trưng)
    def get_state(self):
        state = []
        base_8_idx = self.dir_idx * 2

        for i in range(8):
            d_idx = (base_8_idx + i) % 8
            dr, dc = self.all_8_dirs[d_idx]
            
            curr_r, curr_c = self.head[0] + dr, self.head[1] + dc
            distance = 1.0
            found_food = 0.0
            found_body = 0.0

            while 0 <= curr_r < self.grid_size and 0 <= curr_c < self.grid_size:
                if found_food == 0.0 and [curr_r, curr_c] == self.food:
                    found_food = 1.0
                if found_body == 0.0 and [curr_r, curr_c] in self.body[1:]:
                    found_body = 1.0 / distance
                
                curr_r += dr
                curr_c += dc
                distance += 1.0

            wall_dist = 1.0 / distance
            state.extend([wall_dist, found_body, found_food])

        state.extend([
            float(self.dir_idx == 0),
            float(self.dir_idx == 1),
            float(self.dir_idx == 2),
            float(self.dir_idx == 3)
        ])
        return state

    def compute_reward(self, is_dead, ate_food, is_looping, has_path_to_tail, old_dist, new_dist):
        if is_dead:
            return -15.0
        if is_looping:
            return -10.0
        if ate_food:
            return 25.0
        if not has_path_to_tail:
            return -2.0
        if new_dist < old_dist:
            return 0.2
        return -0.25

    def step(self, action):
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
        is_looping = (self.steps_without_food > self.grid_size * self.grid_size * 1.5)
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
        
