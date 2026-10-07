# Lab 21 — Báo cáo Fine-tuning LLM bằng LoRA

**Họ tên:** Đinh Tiến Cảnh  
**MSSV:** 2A202602918
**Ngày thực hiện:** 07/10/2026

## 1. Mục tiêu và thiết lập thí nghiệm

Mục tiêu của thí nghiệm là kiểm tra liệu LoRA có giúp `unsloth/Qwen3.5-4B` thực hiện tốt
hơn nhiệm vụ phân loại ticket chăm sóc khách hàng hay không, khi đối thủ thực sự là cùng
base model đã được cung cấp một prompt tối ưu. Tôi sử dụng dataset mặc định gồm 250
ticket tiếng Việt vì dataset này có nhãn khách quan, đầu ra JSON đóng và có thể đánh giá
bằng độ chính xác từng trường mà không cần LLM judge. Mỗi đầu ra gồm đúng bốn trường:
`intent`, `urgency`, `product` và `sentiment`.

| Thành phần | Giá trị |
|---|---|
| GPU | Tesla T4, CUDA sm_75, 14.6 GB |
| Precision | fp16 với gradient scaling |
| Base model | `unsloth/Qwen3.5-4B` |
| Dataset | 250 ticket CSKH tiếng Việt → JSON triage |
| Train / validation | 225 / 25, seed 42 |
| Target eval / regression eval | 50 / 15 |
| Mask mode | `assistant-only` |
| Epochs / optimizer steps | 2 / 30 |
| Batch / gradient accumulation | 1 / 16, effective batch 16 |
| LoRA chính | text-linear, r=16, alpha=32, LR=1e-4 |

T4 không hỗ trợ bfloat16 vì thuộc kiến trúc Turing, do đó pipeline tự chọn fp16 và dùng
gradient scaling. Đây là cấu hình đúng cho thiết bị, thay vì đặt cứng `bf16=True` như
trên GPU Ampere trở lên.

## 2. Chat template, loss mask và độ dài chuỗi

Chat template của Qwen3.5-4B **có giữ khối `<think>`**. Kiểm tra render cho thấy reasoning
trace vẫn xuất hiện đầy đủ, nên template về mặt kỹ thuật có thể dùng để huấn luyện trên
trace. Tuy nhiên, corpus mặc định chỉ chứa đáp án JSON và không chứa reasoning trace
thực; vì vậy `valid_trace_rate=0.0` ở bước đánh giá không được dùng làm bằng chứng rằng
mask bị lỗi.

Với `MASK_MODE=assistant-only`, kết quả mask proof là:

| Chỉ số | Kết quả |
|---|---:|
| Token được tính loss | 39 / 94 |
| `supervised_fraction` | 0.4149 |
| Câu trả lời nằm trong loss | `true` |
| Câu hỏi không nằm trong loss | `true` |

Đoạn đầu của phần được tính loss:

```text
</think>

{"intent": "doi_tra", "urgency": "trung_binh", "product": "balo laptop", "sentiment": "trung_tinh"}<|im_end|>
```

Khi cố ý thử chế độ sai `everything`, cả 94/94 token đều được tính loss, bao gồm system
prompt và ticket của người dùng. Điều này chứng minh vì sao mask phải được đọc ngược
thành văn bản trước khi train: nếu chỉ nhìn loss giảm, ta không thể phát hiện model đang
học cả prompt.

Thống kê độ dài trên 250 mẫu là mean 93.1, p50 93, p95 98, p99 100 và tối đa 101 token;
giá trị được đề xuất từ số đo là 256. Tôi giữ `max_length=1024` theo cấu hình T4 chuẩn
của repo để giữ nguyên cấu hình tham chiếu và bảo đảm cả bốn run dùng cùng một thiết
lập. Lựa chọn này an toàn về cắt chuỗi nhưng dư thừa so với corpus và làm tăng chi phí
padding/activation; nếu tối ưu riêng cho dataset này, 256 là giá trị hợp lý hơn.

## 3. Baseline được đóng băng trước khi huấn luyện

Tôi đánh giá hai baseline trên đầy đủ 50 mẫu target và 15 mẫu regression trước khi
train. `EVAL_LIMIT` đã được bỏ và artefact ghi `smoke_mode=false`. Sau khi chấp nhận các
con số dưới đây, prompt tối ưu và tập eval không được chỉnh sửa.

| Run | Target | Regression | Format | Latency (ms/mẫu) |
|---|---:|---:|---:|---:|
| (a) Base + naive prompt | 0.0000 | 0.7911 | 0.0000 | 3480.1 |
| (b) Base + optimized prompt | 0.7650 | 0.7911 | 1.0000 | 1079.6 |
| (c) LoRA fine-tune | 0.9750 | 0.5889 | 1.0000 | 1517.4 |

Baseline `(b)` mạnh hơn rõ rệt `(a)`: target tăng từ 0 lên 0.765 và format tăng từ 0
lên 1.0, trong khi regression giữ nguyên 0.7911. Prompt tối ưu mô tả schema, miền giá
trị và yêu cầu chỉ trả JSON, nên đây là một mốc cạnh tranh thực sự chứ không phải một
prompt bị cố tình làm yếu. Prompt `(b)` được giữ nguyên như bản đi kèm repo; SHA được
ghi nhận là `719e74d3b6232053`.

Latency thấp hơn của `(b)` có thể đến từ việc output JSON ngắn và dừng đúng chỗ, trong
khi `(a)` sinh văn bản dài hơn. Tuy vậy, `(a)` chạy trước và có thể chịu ảnh hưởng của
warm-up, nên tôi không diễn giải chênh lệch latency này như một quan hệ nhân quả tuyệt
đối.

## 4. Huấn luyện chính và giải phẫu ba cấu hình đối chứng

Cả bốn run đều dùng đúng 30 optimizer steps. Run `attn_only` sử dụng rank 283 để khớp
ngân sách tham số với cấu hình chính: 32,456,704 so với 32,464,896 tham số, sai lệch chỉ
khoảng 0.0252%, thấp hơn nhiều so với ngưỡng 5%.

| Run | Vị trí | r | Trainable | LR | Train loss | Target | Format | Thời gian train | VRAM |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `correct` | text-linear | 16 | 32,464,896 | 1e-4 | 0.6255 | 0.975 | 1.0 | 438.2 s | 8.78 GB |
| `attn_only` | q,v | 283 | 32,456,704 | 1e-4 | 0.5376 | 0.970 | 1.0 | 293.8 s | 8.79 GB |
| `wrong_lr` | text-linear | 16 | 32,464,896 | 1e-5 | 1.5702 | 0.000 | 0.0 | 439.1 s | 8.78 GB |
| `qlora` | text-linear | 16 | 32,464,896 | 1e-4 | 0.7058 | 0.940 | 1.0 | 513.7 s | 3.86 GB |

### 4.1. Vị trí adapter so với rank

`attn_only` có training loss 0.5376, thấp hơn `correct` ở 0.6255, nhưng target score
của nó là 0.970, thấp hơn nhẹ so với 0.975 của `correct`. Vì ngân sách tham số gần như
bằng nhau, kết quả không bị nhiễu bởi việc một run có nhiều tham số hơn. Thứ tự theo
training loss và theo metric nhiệm vụ không giống nhau, chứng minh training loss chỉ là
chỉ số thay thế và không thể dùng để xếp hạng adapter. Rank rất lớn 283 trên q,v vẫn
không vượt được text-linear rank 16, nên tăng rank không hoàn toàn bù được phạm vi vị
trí adapter; tuy nhiên chênh lệch target chỉ 0.005 nên kết quả thực tế gần hòa, với lợi
thế nhỏ cho text-linear.

### 4.2. Learning rate sai thang

`wrong_lr` chỉ thay LR từ 1e-4 xuống 1e-5 nhưng target và format đều bằng 0. Trong quá
trình train, loss của run này vẫn giảm từ khoảng 2.163 xuống 1.119 ở log cuối, nên nếu
chỉ quan sát đường loss, tôi có thể kết luận nhầm rằng model đang học hữu ích. Final
loss 1.5702 cao hơn nhiều so với 0.6255 của run đúng, nhưng phát hiện quyết định vẫn là
target=0 và format=0 ở NB5. Thí nghiệm cho thấy LoRA cần learning rate đúng thang; một
learning rate kiểu full fine-tuning có thể tạo ra đường loss đi xuống nhưng chưa đủ để
thay đổi hành vi đầu ra trong ngân sách 30 bước.

### 4.3. Đánh đổi của QLoRA

QLoRA giảm peak VRAM từ 8.78 xuống 3.86 GB, tiết kiệm 4.92 GB, tương đương khoảng 56%.
Đổi lại, target giảm từ 0.975 xuống 0.940, thời gian train tăng từ 438.2 lên 513.7 giây
và latency tăng từ 1517.4 lên 1985.9 ms/mẫu. Kết quả này ủng hộ khuyến nghị không dùng
QLoRA cho Qwen3.5-4B khi LoRA fp16 đã vừa GPU: lượng tử hóa không mang lại lợi ích cần
thiết trên T4 trong thí nghiệm này và làm giảm chất lượng, đồng thời không nhanh hơn.
QLoRA vẫn có giá trị nếu giới hạn VRAM là ràng buộc cứng, nhưng ở đây nó không phải lựa
chọn mặc định tốt nhất.

## 5. Phán quyết của cổng hồi quy

**Kết quả: FAILED**

| Delta so với baseline `(b)` | Giá trị |
|---|---:|
| Target delta | +0.2100 |
| Regression delta | -0.2022 |
| Format delta | 0.0000 |
| `valid_trace_rate` | 0.0000 |

Fine-tune cải thiện target từ 0.765 lên 0.975, tức tăng 21 điểm phần trăm, đồng thời vẫn
giữ format tuyệt đối 1.0. Nếu chỉ đo nhiệm vụ ticket, đây có vẻ là một thành công rõ
ràng. Tuy nhiên regression giảm từ 0.7911 xuống 0.5889, giảm hơn 20 điểm phần trăm và
vượt xa tolerance 0.02 của cổng. Các output định tính cho thấy model thường biến câu hỏi
phổ thông thành JSON triage thay vì làm theo chỉ dẫn, ví dụ phân loại câu hỏi về số tháng
trong năm thành `hoi_thong_tin`. Đây là biểu hiện trực tiếp của catastrophic forgetting
và task over-specialization: hành vi JSON học được đã lấn át khả năng trả lời tổng quát.
Vì vậy tôi không thay threshold và không coi target cao là đủ để deploy. Bước tiếp theo
hợp lý là bổ sung 1–5% replay data phổ thông vào train, giữ nguyên eval đã đóng băng,
rồi chạy lại cùng cổng để xem có giữ được phần lớn target gain mà phục hồi regression
hay không.

## 6. Phân tích định tính

Trên 50 mẫu target, fine-tune thắng baseline `(b)` ở 33 mẫu, hòa ở 17 mẫu và không thua
mẫu nào. Do đó tôi không tạo giả hai ca target thua. Để phân tích trung thực verdict
FAILED, bảng dưới gồm ba ca target fine-tune thắng và hai ca regression fine-tune thua;
hai ca sau chính là bằng chứng định tính cho suy giảm năng lực tổng quát.

| # | Nhóm | Input | Nhãn/đáp án đúng | Baseline `(b)` | Fine-tune | Nhận xét |
|---:|---|---|---|---|---|---|
| 1 | Target | Trả lại chuột không dây, gấp, shop hỗ trợ tốt | `doi_tra`, `cao`, chuột không dây, `tich_cuc` | Dự đoán `hoan_tien`; 0.75 | Dự đoán đủ bốn trường đúng; 1.00 | **FT thắng:** sửa đúng nhầm lẫn giữa đổi trả và hoàn tiền. |
| 2 | Target | Hoàn tiền ốp lưng, sớm, bực mình | `hoan_tien`, `trung_binh`, ốp lưng điện thoại, `tieu_cuc` | Dự đoán urgency=`cao`; 0.75 | Dự đoán urgency=`trung_binh`; 1.00 | **FT thắng:** học đúng quy ước ánh xạ “sớm nhé”. |
| 3 | Target | Đèn bàn LED vỡ khi nhận, gấp, shop xem giúp | `san_pham_loi`, `cao`, đèn bàn LED, `trung_tinh` | Dự đoán sentiment=`tieu_cuc`; 0.75 | Dự đoán sentiment=`trung_tinh`; 1.00 | **FT thắng:** tránh suy diễn tiêu cực chỉ từ sản phẩm bị vỡ. |
| 4 | Regression | Viết một câu chúc mừng sinh nhật bằng tiếng Việt | Chứa “sinh nhật” | Viết lời chúc tự nhiên; 1.00 | Trả JSON với intent `chuc_mung_sinh_nhat`; 0.00 | **FT thua:** áp đặt schema phân loại lên một yêu cầu sinh văn bản. |
| 5 | Regression | Một năm có bao nhiêu tháng? | Chứa “12” | Trả lời 12 tháng; 1.00 | Trả JSON intent `hoi_thong_tin`, không có đáp án 12; 0.00 | **FT thua:** hành vi JSON lấn át kiến thức phổ thông. |

Mẫu chung của các ca thua là fine-tune không nhất thiết mất toàn bộ kiến thức nền, mà
mất khả năng chọn đúng **hình thức trả lời** ngoài miền ticket. Ở một số câu như quang
hợp, nội dung cơ bản vẫn đúng nhưng bị bọc trong schema không phù hợp; ở các câu khác,
schema triage thay thế hoàn toàn câu trả lời. Điều này giải thích vì sao format trên
target đạt 1.0 nhưng regression lại giảm mạnh.

## 7. Kết luận

Thí nghiệm chứng minh LoRA có thể chuyên biệt hóa Qwen3.5-4B rất hiệu quả cho bài toán
ticket: target tăng từ 0.765 lên 0.975 và model tuân thủ JSON ở toàn bộ 50 mẫu. Tuy
nhiên, tôi **không nên deploy bản fine-tune hiện tại như một trợ lý dùng chung**, vì
regression giảm từ 0.7911 xuống 0.5889 và model nhiều lần áp đặt schema ticket lên câu
hỏi phổ thông. Kết quả FAILED không phủ nhận lợi ích của fine-tuning; nó cho thấy phạm
vi triển khai phải được xác định bằng nhiều nhóm metric thay vì chỉ metric đích. Nếu hệ
thống được cô lập hoàn toàn như một dịch vụ phân loại ticket, model có thể hữu ích sau
khi bổ sung kiểm thử ngoài phân phối. Nếu hệ thống vẫn phải xử lý yêu cầu tổng quát,
catastrophic forgetting hiện tại là lý do chặn triển khai.

Đòn bẩy rõ nhất trong thí nghiệm là learning rate và dữ liệu/hành vi đầu ra. Chỉ giảm LR
mười lần khiến target về 0 dù loss vẫn đi xuống. Vị trí adapter cũng có ảnh hưởng, nhưng
`correct` và `attn_only` gần hòa khi đã khớp ngân sách, nên dữ liệu nhỏ và metric đã gần
bão hòa không cho phép kết luận vị trí là đòn bẩy áp đảo. Mask đúng là điều kiện nền
tảng: nó ngăn model học lại prompt và làm cho mọi so sánh sau đó có ý nghĩa. Nếu có
thêm hai giờ, tôi sẽ trộn 1–5% dữ liệu replay từ tập regression hoặc một tập instruction
phổ thông tách biệt, giữ nguyên 50 target và 15 regression đã đóng băng, rồi đo xem điểm
cân bằng tốt nhất nằm ở đâu thay vì cố làm cho verdict PASS bằng cách nới gate.

## 8. Điều tôi học được

1. Training loss thấp hơn không đồng nghĩa adapter tốt hơn: `attn_only` có loss thấp
   hơn nhưng target vẫn thấp hơn nhẹ `correct`.
2. Một fine-tune có thể gần hoàn hảo trên nhiệm vụ đích nhưng vẫn không an toàn để triển
   khai, vì hành vi chuyên biệt có thể lấn át instruction following tổng quát.
3. Prompt tối ưu là baseline bắt buộc: base model tăng từ target 0 lên 0.765 chỉ bằng
   prompt, nên so fine-tune với prompt ngây thơ sẽ phóng đại lợi ích của huấn luyện.

## 9. Phần thưởng

- NB6 merge và hot-swap: chưa thực hiện.
- Dataset miền riêng: chưa thực hiện.
- Reasoning-trace collapse: chưa thực hiện vì corpus không có trace thực.
- Quét rank có kiểm soát: chưa thực hiện.
- Hugging Face Hub: chưa thực hiện.
