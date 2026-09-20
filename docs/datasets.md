# Các dataset dùng được cho nhận dạng chữ cảnh tự nhiên tiếng Việt

Khảo sát 2026-08. Kết luận nhanh: **VinText là chuẩn duy nhất được cộng
đồng công nhận** — bắt buộc là dataset chính. Các bộ khác chỉ nên là phụ.

## 1. VinText (chính — bắt buộc)

- Nguồn: VinAI, CVPR 2021 "Dictionary-guided Scene Text Recognition".
- 2.000 ảnh / 56.084 instance; crops của ta: 24.838 / 6.922 / 9.789.
- Là benchmark duy nhất mọi bài Việt ngữ đều so (ABCNet, SwinTextSpotter,
  DeepSolo, LRANet++ 75.6% 2025...). Công khai, giấy phép nghiên cứu.

## 2. BKAI-NAVER SoICT Hackathon 2022 — Track OCR (phụ, đáng thử xin)

- Cuộc thi detection + recognition chữ Việt cảnh đường phố (aihub.ml,
  competition 214). Ảnh biển hiệu Hà Nội, có polygon + transcript.
- Truy cập: cần đăng ký cuộc thi (đã đóng) → **email BTC BKAI/SoICT xin
  bản research**. Nếu xin được: dùng làm tập đánh giá chéo (train VinText
  → test BKAI) chứng minh khả năng khái quát — reviewer rất thích.
- Rủi ro: có thể không xin được / nhãn kém chuẩn → không đưa vào kế
  hoạch chính.

## 3. VnSceneText (không khuyến nghị)

- Xuất hiện trong vài bài (Ensemble Learning 2024, empirical study 2022)
  nhưng không có trang phát hành công khai rõ ràng → khó tái lập, bỏ qua.

## 4. VNOnDB / HANDS-VNOnDB + UIT-HWDB (khác miền — thí nghiệm mở rộng)

- Chữ **viết tay** (online/offline), không phải scene text. Chỉ dùng nếu
  muốn thêm 1 bảng "phương pháp phân rã khái quát sang miền viết tay"
  — điểm cộng, không bắt buộc. VNOnDB xin qua TUAT/ICFHR 2018.

## 5. Dữ liệu tổng hợp (chủ động — không giới hạn)

- SynthTIGER + corpus/font Việt (xem synthetic_data.md): 500k–1M ảnh,
  kiểm soát được phân bố thanh điệu (oversample hỏi/ngã).
- Đây là "dataset thứ hai" thực tế nhất: vừa là dữ liệu pretrain, vừa là
  một đóng góp mô tả được trong bài (tone-rebalanced synthesis).

## Khuyến nghị giao thức cuối cùng

1. Chính: pretrain synth → fine-tune VinText train → báo cáo VinText test.
2. Khái quát hóa (nếu xin được BKAI): zero-shot test trên BKAI.
3. Không trộn BKAI/VnSceneText vào train — giữ giao thức sạch để so sánh
   công bằng với các baseline retrain trên cùng VinText.
