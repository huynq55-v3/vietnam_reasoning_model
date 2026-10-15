---
name: generate-reasoning-data
description: >-
  Generate high-quality synthetic Vietnamese reasoning dataset with entity masking/swapping
  (de-factualization), hard-negative distractor chunks, content-based citations, and compact reasoning traces (<|thought|>).
  Use this skill whenever asked to generate, augment, or curate training data for the Vietnam Reasoning Model.
---

# Kỹ năng Sinh Dữ liệu Suy luận Tiếng Việt (Synthetic Reasoning Data Generation v3 - Production Ready)

Kỹ năng này hướng dẫn Agent tạo dữ liệu huấn luyện SLM (~150M-250M tham số) kết hợp cổng Cross-Attention / Vector DB, loại bỏ triệt để lối tắt học vẹt (shortcuts) và tối ưu hóa chi phí token.

---

## 📋 Schema Mẫu Dữ liệu Chuẩn (JSON Lines)

Mỗi mẫu dữ liệu là một dòng JSON độc lập:

```json
{
  "id": "sample_00001",
  "domain": "law_admin | finance_contract | medical_bio | tech_infra | daily_life | agriculture_env",
  "category": "multi_hop_deduction | boundary_math | counterfactual_rule | negative_refusal | conflict_resolution",
  "user_query": "Câu hỏi tự nhiên của người dùng (đa dạng từ câu ngắn gọn đời thường đến câu hỏi học thuật/hành chính)",
  "external_memory_chunks": [
    "Chunk nhiễu khó (Hard Negative - đứng đầu ngẫu nhiên)",
    "Chunk thông tin vàng 2 (Golden Fact 2)",
    "Chunk nhiễu nền (Soft Negative)",
    "Chunk thông tin vàng 1 (Golden Fact 1)",
    "Chunk điều kiện biên (Golden Fact 3)"
  ],
  "assistant_reasoning_and_reply": "<|thought|>\n[Quá trình suy luận ngắn gọn, lọc nhiễu khó và kết nối dữ kiện]\n<|/thought|><|answer|>\n[Câu trả lời hoàn chỉnh cho người dùng]<|endoftext|>"
}

```

---

## 🎯 7 Quy tắc Sống Còn khi Sinh Mẫu Số Lượng Lớn

### 1. Phân tách Token Trạng thái Tuyệt đối

* Mở đầu tư duy bằng `<|thought|>`.
* Đóng tư duy và mở trả lời liền kề: `<|/thought|><|answer|>`.
* Kết thúc mẫu bằng `<|endoftext|>`. Không sinh thêm bất kỳ ký tự nào sau thẻ này.

### 2. Thiết kế "Hard Negative" & Xáo trộn Vị trí Chunks (5 - 7 Chunks / Mẫu)

* **BẮT BUỘC XÁO TRỘN (Random Shuffle):** Tuyệt đối không để các Golden Chunks nằm liên tiếp ở đầu mảng hoặc xếp sẵn theo thứ tự thời gian/logic. Phải trộn lẫn vị trí giữa Golden Chunks và Distractor Chunks.
* **Cấu trúc Nhiễu (Distractors) chuẩn Vector DB:**
* Không dùng nhiễu vô thưởng vô phạt (ví dụ: đang hỏi bảo mật lại chèn giờ mở cửa nhà ăn). Vector DB thực tế chỉ trả về các đoạn có độ tương đồng từ khóa cao.
* **Hard Negatives (2-3 chunks):** Phải chứa cùng từ khóa chính nhưng sai khác một thuộc tính quyết định (Ví dụ: Hỏi về *công ty Thanh Hà năm 2024* thì nhiễu nói về *công ty Thanh Hà năm 2022* hoặc *chi nhánh Thanh Hải năm 2024*; hỏi về *thuế xuất khẩu gỗ* thì nhiễu nói về *thuế nhập khẩu gỗ*).



### 3. Quy tắc "Lọc Nhiễu Tinh Gọn" trong `<|thought|>` (Tránh lãng phí Attention)

* **Với Nhiễu nền (khác hẳn đối tượng đang hỏi):** Lờ đi hoàn toàn trong `<|thought|>`, không tốn token nhắc lại.
* **Với Hard Negatives (dễ gây nhầm lẫn trực tiếp):** Chỉ rõ lý do loại trừ trong 1 mệnh đề ngắn gọn (Ví dụ: *"Loại trừ quy định về thiết bị từ tính vì câu hỏi đang xét thiết bị quang học"*).

### 4. Trích dẫn Theo Nội dung & Cấm suy diễn ngoài Văn bản (Zero External Assumption)

* **TUYỆT ĐỐI KHÔNG** có tiền tố `Chunk 1:`, `Đoạn 2:` trong cả mảng dữ liệu lẫn trong `<|thought|>`.
* **Cấm dùng kiến thức đời thực để lấp chỗ trống:** Mọi bước bắc cầu trong `<|thought|>` phải dựa 100% vào các mệnh đề có trong `external_memory_chunks`. Với mẫu `negative_refusal`, nếu dữ liệu nói chất A kháng được chất B, còn môi trường có chất C, model phải kết luận là *"Chưa có dữ liệu về tác động của chất C lên chất A"* chứ không tự suy diễn chất C vô hại.

### 5. Đa dạng hóa Bối cảnh Đời sống & Tiếng Việt (Chống Sci-Fi Bias)

Không được lạm dụng bối cảnh khoa học viễn tưởng (tàu vũ trụ, thiên hà, phòng lab lượng tử chỉ được chiếm tối đa 10%). Phải phân bổ đều vào các bối cảnh tiếng Việt thực tế (nhưng vẫn dùng tên thực thể/con số giả định):

* **Hành chính - Pháp lý (25%):** Thủ tục đất đai giả định, quy chế công ty, điều khoản hợp đồng, bảo hiểm, xử phạt nội bộ.
* **Kinh tế - Đời sống - Giao thương (25%):** Tính cước vận chuyển, chiết khấu đại lý, lịch trình xe chạy, điều kiện bảo hành thiết bị.
* **Sinh học - Nông nghiệp - Y dược giả định (20%):** Quy trình nuôi trồng, pha chế dinh dưỡng, liều lượng dược chất giả định, điều kiện nhiệt độ/độ ẩm.
* **Kỹ thuật - Hạ tầng - Mạng lưới (20%):** Cấu hình máy chủ, phụ tải điện, luồng dữ liệu, phân quyền truy cập hệ thống.
* **Giả tưởng / Học thuật trừu tượng (10%).**

### 6. Đa dạng hóa Văn phong `<|thought|>` và `user_query`

* **Với `user_query`:** Luân phiên giữa câu hỏi đầy đủ lịch sự, câu hỏi trực diện ngắn gọn (ví dụ: *"Bác Lâm có được hưởng phụ cấp độc hại tháng này không, tính hộ tôi cái?"*), và câu hỏi đưa ra một nhận định sai để thử thách model.
* **Với `<|thought|>`:** Cấm dùng một khuôn mẫu cố định. Luân phiên ngẫu nhiên giữa:
1. *Chuỗi suy diễn mũi tên:* `Mục tiêu: Kiểm tra X -> Điều kiện cần: (A và B) -> Đối chiếu: A đạt, nhưng B vi phạm (do C > 10) => Kết luận: Không đạt.`
2. *Đoạn văn phân tích gãy gọn:* Viết liền mạch 3-4 câu súc tích đối chiếu mâu thuẫn và chốt đáp án.
3. *Gạch đầu dòng kiểm tra điều kiện biên:* Liệt kê ngắn gọn các tiêu chí và trạng thái Đạt/Không đạt.



### 7. Phân bổ Tỷ lệ 5 Categories Chuẩn

* `multi_hop_deduction` (Suy luận bắc cầu 2-4 bước): **30%**
* `boundary_math` (So sánh ngưỡng số học, tính phần trăm, cộng trừ thời gian/chi phí): **25%**
* `counterfactual_rule` (Luật giả định ngược đời thường để chống học vẹt, ví dụ: quy ước ở vùng X nước đóng băng ở 50°C): **20%**
* `negative_refusal` (Thiếu mắt xích dữ liệu then chốt -> Từ chối kết luận võ đoán): **15%**
* `conflict_resolution` (Dữ liệu mâu thuẫn nhau -> Giải quyết bằng thời gian ban hành mới hơn, hoặc luật chuyên ngành đè luật chung, hoặc nguồn tin chính thức đè tin đồn): **10%**

