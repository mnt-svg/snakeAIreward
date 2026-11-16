# Snake AI - State & Reward Design (Reinforcement Learning)

## 1. Tổng quan kiến trúc AI
* **Phương pháp huấn luyện:** Reinforcement Learning (Học tăng cường) — AI tự học thông qua cơ chế thử và sai (Trial & Error) khi tương tác với môi trường.
* **Loại tác tử (AI Agent):** **Utility-based Learning Agent** — Tác tử tự cải thiện hành vi qua kinh nghiệm dựa trên hàm tiện ích (Reward Function) để tối đa hóa tổng điểm thưởng dài hạn.

## 2. Giao tiếp với AI qua bộ 3: State – Action – Reward

### A. State (Không gian trạng thái - 14 đặc trưng nhị phân)
Cung cấp thông tin toàn diện để AI nhận biết môi trường xung quanh:
* **Nguy hiểm tức thời (3 bit):** Phát hiện tường hoặc thân rắn ở ngay phía `[Thẳng, Phải, Trái]`.
* **Hướng di chuyển hiện tại (4 bit):** `[Trái, Phải, Lên, Xuống]`.
* **Hướng mục tiêu/quả (4 bit):** Vị trí tương đối của quả so với đầu rắn `[Trái, Phải, Trên, Dưới]`.
* **An toàn không gian (3 bit):** Sử dụng thuật toán **BFS** kiểm tra xem nếu đi theo hướng `[Thẳng, Phải, Trái]` thì từ đầu rắn có còn đường thông tới đuôi hay không (`1` = An toàn, `0` = Ngõ cụt).

### B. Action (Không gian hành động)
* Gồm 3 hành động tương đối theo góc nhìn của rắn: `0` (Đi thẳng), `1` (Rẽ phải), `2` (Rẽ trái).

### C. Reward Function (`compute_reward` - Hàm mục tiêu)
Chuyển hóa mục tiêu *"ăn quả, không chết, không tự nhốt mình"* thành hệ thống điểm số:

| Sự kiện | Điểm (`Reward`) | Ý nghĩa thiết kế |
| :--- | :---: | :--- |
| **Ăn được quả** | `+10.0` | Thưởng lớn cho mục tiêu chính (tăng điểm). |
| **Đâm tường / Cắn thân** | `-10.0` | Phạt nặng để AI học cách sinh tồn. |
| **Đi vòng tròn quá lâu** | `-5.0` | Ngăn chặn *Reward Hacking* (câu giờ né chết). |
| **Mất đường tới đuôi** | `-3.0` | Phạt chiến thuật khi AI tự nhốt mình vào ngõ cụt lúc cơ thể dài. |
| **Tiến lại gần quả** | `+0.1` | *Reward Shaping*: Dẫn đường cho AI ở giai đoạn đầu chưa biết gì. |
| **Đi ra xa quả** | `-0.15` | Khuyến khích tìm đường đi ngắn nhất tới mồi. |

## 3. Chạy thử nghiệm môi trường
```bash
python snakeAIenv.py
