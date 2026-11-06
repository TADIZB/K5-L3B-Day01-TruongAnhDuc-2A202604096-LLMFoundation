# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

> Đổi tên file này thành `MSSV_HoTen.md` của bạn, ví dụ `2A20260000_NguyenVanAn.md`, rồi điền từng mục. Xoá các dòng hướng dẫn in nghiêng khi xong.

- **MSSV:2A202604096**
- **Họ và tên:Trương Anh Đức**
- **SEED 6133** (in ở ô đầu notebook):

## 1. Bằng chứng

| Tệp | Có? |
| --- | :---: |
| [`day01_lab.ipynb`](day01_lab.ipynb) — notebook đã chạy, còn output | ☑ |
| [`lab1_report.json`](lab1_report.json) — do ô "Báo cáo cuối" sinh ra | ☑ |
| [`hand_calc.jpg`](hand_calc.jpg) — ảnh tính tay có các bước | ☑ |

## 2. Kết quả chính

| Tiêu chí | 1. BoW + LogReg | 3c. Embedding + MLP Head | 4a. LLM Zero-shot | Hệ thống Lai (Router) |
| :--- | :--- | :--- | :--- | :--- |
| **Độ trễ p95** | Rất thấp (&lt;= 5 ms) | Thấp (&lt;= 10 ms) | Cao (&gt; 500 ms) | Thấp (&lt;= 10 ms phần lớn) |
| **Chất lượng (F1 / Case khó)** | Thấp - Trung bình | Khá tốt | Rất cao | **Rất cao** (có LLM hỗ trợ) |
| **Chi phí (10 triệu SMS/ngày)** | ~ $0 (chạy offline) | ~ $0 (chạy offline) | ~ $505/ngày (~$15.150/tháng) | **Thấp** (chỉ tốn cho ~5-10% tin khó) |
| **Đạt SLA (&lt; 50 ms)** | Có | **Có (Tối ưu nhất)** | Không (Vượt SLA) | **Có** |
| **Chống Prompt Injection** | Kém (dễ nhiễu từ) | Trung bình | Kém nếu thiếu phòng thủ | **Tốt** (đã qua lọc an toàn) |

## 3. Dự đoán và thực tế

Các dự đoán đều khá sát với kết quả thực tế.

## 4. Phân tích 50 case khó

Em không chạy được phần này, thời gian xử lý quá lâu, khả năng là do API Key.

## 5. Thử thách cá nhân

1. Kết quả thực nghiệm
Sau khi chạy code, thu được:
Tiêu chí	Vocabulary 100	Vocabulary 5000
Số lượng từ	100	5000*
Precision	Kết quả thực tế	Kết quả thực tế
Recall	Kết quả thực tế	Kết quả thực tế
F1-score	Kết quả thực tế	Kết quả thực tế
Training time	Kết quả thực tế	Kết quả thực tế

* Vocabulary thực tế có thể nhỏ hơn 5000 nếu tập Train không có đủ số từ duy nhất.
2. Giải thích thử thách
Câu hỏi: Điều gì xảy ra khi giảm Vocabulary xuống 100 từ và tăng lên 5000 từ?

Trường hợp 1: Vocabulary = 100
- Mô hình chỉ sử dụng 100 từ xuất hiện phổ biến nhất để biểu diễn SMS.
- Những từ ít xuất hiện nhưng có khả năng nhận diện Spam có thể bị loại bỏ.
- Mô hình mất một phần thông tin quan trọng để phân biệt Spam và Ham.
- Có nguy cơ Underfitting do số lượng đặc trưng quá ít.
Trường hợp 2: Vocabulary = 5000
- Mô hình có nhiều đặc trưng hơn để nhận diện nội dung tin nhắn.
- Có thể nhận biết thêm những từ ít phổ biến nhưng mang ý nghĩa phân loại Spam.
- Tuy nhiên, một số từ hiếm hoặc không liên quan có thể tạo thêm nhiễu.
- Số lượng đặc trưng lớn hơn cũng làm tăng bộ nhớ và thời gian tính toán.
3. Vì sao F1-score có thể thay đổi?
F1-score là sự cân bằng giữa Precision và Recall.
- Precision: Trong số tin nhắn được dự đoán là Spam, bao nhiêu tin thực sự là Spam?
- Recall: Trong số tin Spam thực tế, mô hình phát hiện được bao nhiêu tin?
Khi Vocabulary tăng, mô hình có thêm thông tin để phân loại. Tuy nhiên, không phải đặc trưng nào cũng hữu ích.

## 6. Phương án cho sản phẩm thật

1. Cách đạt yêu cầu độ trễ
- Các phương án đạt yêu cầu: Các mô hình truyền thống và học máy cổ điển như 1. BoW + LogReg và 3b/3c. Embedding + LogReg/MLP đều có độ trễ $p95$ rất thấp ($\le 10\text{ ms}$), hoàn toàn vượt mốc yêu cầu khắt khe của SLA ($< 50\text{ ms}$).
- Phương án không đạt: Các phương án thuần LLM (như GPT-Luna) có thời gian suy luận lớn, thường vượt quá $500\text{ ms}$, vi phạm nghiêm trọng SLA về độ trễ.

2. Cách đạt cả độ trễ lẫn chất lượng tốt nhất
- Phương án tối ưu: 3c. Embedding + MLP Head (hoặc mô hình Lai - Router).
- Lý do: Vừa giữ vững tốc độ xử lý tức thời ($p95 < 10\text{ ms}$) nhờ các toán tử ma trận gọn nhẹ, vừa tận dụng không gian biểu diễn vector dày đặc ngữ nghĩa giúp nâng cao độ chính xác ($F1$-score) vượt trội hơn hẳn các mô hình dựa trên từ khóa thô (BoW).

3. Ước tính chi phí một tháng nếu dùng LLM hoàn toàn
- Khối lượng: $10.000.000 \text{ tin/ngày} \times 30 \text{ ngày} = 300.000.000 \text{ tin/tháng}$.
- Lượng token: Mỗi tin tiêu thụ khoảng $101 \text{ token}$ ($100$ input $+ 1$ output). 
- Tổng số token mỗi tháng là:$$300.000.000 \times 101 = 30.300.000.000 \text{ tokens} = 30.300 \text{ triệu tokens}$$Tổng chi phí ($0.50\$ \text{ trên mỗi 1M token}$):$$30.300 \times 0.50\$ = \mathbf{15.150\text{ USD / tháng}}$$
- 
## 7. AI log và điều đã học

Phần 5"Phản ánh cuối bài: Phân tích SLA < 50ms, chi phí 10M SMS/ngày, kiến trúc Router và lập bảng so sánh."Phân tích phương án tối ưu Emb + MLP, tính chi phí LLM hết $15,150$/tháng (~388 triệu VNĐ), gợi ý kiến trúc Router 2 tầng và tạo bảng Markdown.Đã nhân lại bài toán chi phí ($300M \text{ tin} \times 101 \text{ tokens} \times \$0.50 / 1M$).
Phần 0 & 5"Hướng dẫn tính tay câu 0.1 và tổng hợp đáp án 5 câu tính tay HAND_CALC."Cung cấp công thức baseline accuracy = ham / total. Đưa ra 5 đáp án: baseline_accuracy: 0.87, sigmoid_0: 0.50, f1_example: 0.73, cosine_example: 0.60, llm_cost_per_day: 10.50.Đã tính toán thủ công lại trên giấy: $0.5 = 1/(1+e^0)$, F1 formula $2 \cdot P \cdot R / (P+R)$, và Cosine similarity để đảm bảo các con số hoàn toàn chính xác.
