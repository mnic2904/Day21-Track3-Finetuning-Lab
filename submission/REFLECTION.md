# Reflection — Lab 21

## 1. Điều gì làm tôi ngạc nhiên nhất?

Điều làm tôi ngạc nhiên nhất là fine-tune có thể cải thiện rất mạnh nhiệm vụ đích nhưng
đồng thời làm model kém đi rõ rệt ở các yêu cầu phổ thông. Target score tăng từ 0.765
lên 0.975 và không có mẫu target nào fine-tune thua baseline prompt: fine-tune thắng 33
mẫu và hòa 17 mẫu. Tuy nhiên, regression lại giảm từ 0.7911 xuống 0.5889. Khi xem output
từng mẫu, tôi thấy model thường ép những câu hỏi như “Một năm có bao nhiêu tháng?” vào
JSON triage thay vì trả lời “12”. Điều này khiến tôi hiểu cụ thể hơn rằng catastrophic
forgetting không nhất thiết là model mất hết kiến thức; đôi khi model vẫn có kiến thức
nhưng hành vi chuyên biệt đã lấn át cách làm theo chỉ dẫn phù hợp.

## 2. Tôi mất nhiều thời gian nhất ở đâu? Nó có phải chỗ tôi dự đoán không?

Trong pipeline chính, NB4 tốn nhiều thời gian nhất: khoảng 23.7 phút để train ba cấu
hình đối chứng, so với tổng thời gian khoảng 56.7 phút cho NB1–NB5. Đây là điều có thể
dự đoán vì NB4 phải huấn luyện ba adapter riêng. Phần tôi không dự đoán trước là sau khi
pipeline hoàn tất, tôi vẫn cần chạy thêm inference để lấy prediction từng mẫu của
baseline và fine-tune. Artefact ban đầu chỉ có điểm tổng và các output fine-tune rút
gọn, nên chưa đủ để chứng minh một ví dụ cụ thể là fine-tune thắng hay thua baseline.
Việc kiểm tra định tính cẩn thận mất thêm thời gian nhưng giúp tránh gắn nhãn sai cho các
“ca tệ nhất”.

## 3. Trước lab này tôi tin điều gì về fine-tuning mà giờ tôi không còn tin?

Trước lab, tôi dễ cho rằng nếu training loss giảm mạnh và điểm nhiệm vụ đích tăng thì
fine-tune đã thành công. Sau thí nghiệm, tôi không còn xem hai dấu hiệu đó là đủ. Run
`attn_only` có training loss thấp hơn `correct` nhưng target score lại thấp hơn nhẹ;
run `wrong_lr` vẫn có đường loss đi xuống nhưng target và format đều bằng 0. Quan trọng
hơn, adapter chính đạt target 0.975 nhưng vẫn bị cổng hồi quy đánh FAILED. Tôi hiện cho
rằng thành công của fine-tuning phải được quyết định bằng metric tác vụ thật, baseline
prompt mạnh và các kiểm tra hồi quy, không phải chỉ bằng loss hoặc một con số accuracy.

## 4. Tôi dùng AI assistant vào việc gì trong lab? Chỗ nào nó sai?

Tôi dùng AI assistant để đọc rubric, diễn giải log NB1–NB5, kiểm tra các phép so sánh,
tính mức chênh tham số/VRAM và hỗ trợ tổ chức báo cáo. AI cũng giúp viết cell bổ sung để
sinh prediction từng mẫu cho baseline và fine-tune. Điểm chưa chính xác ban đầu là việc
có thể xem ba mẫu fine-tune có điểm thấp nhất như các “ca fine-tune thua”. Sau khi chấm
hai model trên cùng từng mẫu, kết quả cho thấy fine-tune không thua baseline ở bất kỳ
mẫu target nào; các ca thua thực sự nằm trong tập regression. Tôi đã sửa phân tích dựa
trên số đo này thay vì giữ kết luận ban đầu. Bài học là AI có thể giúp tổng hợp nhanh,
nhưng mọi nhận định định lượng vẫn phải được kiểm tra bằng artefact và phép so sánh đúng.

## 5. Nếu ngày mai phải fine-tune cho một khách hàng thật, bước đầu tiên tôi làm là gì?

Bước đầu tiên của tôi sẽ là định nghĩa và đóng băng bộ đánh giá trước khi huấn luyện.
Tôi sẽ cùng khách hàng xác định metric nhiệm vụ, các trường hợp lỗi quan trọng, một
baseline prompt mạnh và một tập regression đại diện cho những năng lực không được phép
mất. Sau đó tôi mới kiểm tra dữ liệu train, leakage và loss mask. Cách làm này ngăn việc
thay đổi tiêu chuẩn sau khi đã thấy kết quả và giúp trả lời câu hỏi kinh doanh quan trọng
hơn “loss có giảm không”: model mới có thực sự tốt hơn giải pháp hiện tại mà không tạo
ra một loại rủi ro khác hay không?
