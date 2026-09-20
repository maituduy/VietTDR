# DASR — Siêu phân giải nhận thức dấu cho chữ Việt cảnh tự nhiên

Hướng mới của công trình 2, chốt ngày 2026-08-06 sau khi số liệu vòng 2
cho thấy các cải tiến kiến trúc recognizer không còn tác dụng ở ngân sách
huấn luyện đầy đủ.

## Vì sao đổi hướng — chuỗi bằng chứng đo được

1. Recognizer tốt nhất đạt 82.1% trên VinText; phân tích ma trận nhầm lẫn:
   lỗi thanh điệu chỉ còn 1.38% ký tự, giải thích **tối đa 25% số lỗi từ**.
   → **≥75% lỗi không phải do dấu thanh.**
2. Phân bố kích thước ảnh thật: 25% crop có chiều cao ≤16px (1.746/6.922
   trên tập val), p10 = 13px. Ảnh nhỏ/mờ chính là nhóm lỗi chủ đạo.
3. Chương 5 luận văn (công trình 1) cũng đã nêu siêu phân giải là hướng
   tương lai — tính liên tục học thuật có sẵn.

## Phương pháp: DASR (Diacritic-Aware Super-Resolution)

Model SR nhẹ ×2 (16×64 → 32×128, ~1.2M tham số, SRResNet-style + global
bicubic skip) đặt trước recognizer. Ba điểm mới:

### ① Mặt nạ dấu tự sinh bằng render kép (không cần annotation)

Mỗi từ được render HAI lần y hệt nhau (font, vị trí, màu, nền) — một lần
đủ dấu, một lần chỉ chữ gốc (ệ→e, ắ→a, đ→d). Hiệu pixel của hai bản render
= mặt nạ chính xác của vùng dấu, hoàn toàn miễn phí. Mặt nạ chiếm 0.3–3.4%
diện tích ảnh → loss thường gần như bỏ qua vùng này; ta thêm **loss trọng
số vùng dấu** (chuẩn hóa theo diện tích mặt nạ).

Đã kiểm chứng thị giác: `samples/sr_pairs_check.jpg` — mặt nạ khoanh đúng
dấu hỏi/móc/sắc/mũ/nặng/ngã trên 10/10 mẫu.

### ② Recognizer phân rã đông cứng làm giám sát thành phần

Dùng VietTDR (công trình phân rã) đóng băng làm loss: ảnh SR phải giữ cho
recognizer đọc đúng (chữ gốc, dấu nguyên âm, thanh điệu). Đầu thanh điệu
cấp gradient nhắm thẳng vào pixel dấu — **đây là chỗ phân rã ba thành phần
phát huy tác dụng thật**: làm prior cho SR thay vì làm đầu phân loại.
(Câu chuyện nối tiếp trung thực: vòng 2 chứng minh phân rã không giúp
recognizer ở ngân sách cao, nhưng cấu trúc thành phần của nó là tín hiệu
giám sát đúng cho SR.)

### ③ Giao thức đánh giá không cần cặp LR/HR thật

- **D-PSNR** (Diacritic-PSNR): PSNR tính riêng trong mặt nạ dấu — đo đúng
  vài chục pixel quyết định ỏ/õ/o thay vì trung bình toàn ảnh.
- **Đánh giá hạ nguồn trên ảnh thật**: lấy crop VinText thật cao ≤16px,
  so word accuracy của recognizer trên ảnh gốc vs ảnh qua SR. Không cần
  ground-truth HR — vượt qua hạn chế "không có TextZoom tiếng Việt".

## Bảng thực nghiệm dự kiến

| Run | Cấu hình | Đo được gì |
|---|---|---|
| R | Recognizer full (500k synth, 15ep + FT 30ep) | prior + bộ đánh giá; checkpoint PHẢI lưu được lần này |
| S0 | SR chỉ Charbonnier | baseline |
| S1 | + loss mặt nạ dấu (①) | đóng góp của ① qua D-PSNR + hạ nguồn |
| S2 | + loss thành phần từ recognizer (②) | đóng góp của ② |
| B | Bicubic (không SR) | hạ đáy |

So sánh chính: word acc trên tập VinText-LR thật (≤16px, ~2.5k mẫu test)
của B vs S0 vs S1 vs S2, kèm PSNR/SSIM/D-PSNR trên cặp tổng hợp.

## Ngân sách Kaggle (30h GPU, reset 15-08)

| Việc | Ước tính |
|---|---|
| Phiên 1: retrain recognizer full + lưu checkpoint (đã sửa lỗi mất output) | ~6h |
| Phiên 2: sinh 120k cặp (~10 phút) + S0/S1/S2 (mỗi cái ~30–45 phút) + eval | ~3h |
| Dự phòng + baseline bổ sung | còn ~20h |

SR net rất nhỏ nên chu kỳ thí nghiệm nhanh hơn hẳn vòng recognizer.

## Sản phẩm viết

- **Bài báo** (学报): 《面向低分辨率越南语场景文字的变音符号感知超分辨率方法》
  — động cơ bằng phân tích lỗi định lượng (75%/25%), phương pháp ①②③,
  ablation S0/S1/S2, và phần "bài học từ recognizer" (vòng 1 vs vòng 2)
  làm phân tích phụ trung thực.
- **Chuyên lợi**: có thể nộp bản phân rã hiện có (hiệu quả ghi theo điều
  kiện ít dữ liệu), hoặc soạn bản mới cho render kép + mặt nạ dấu + loss
  thành phần (mới hơn, khớp bài báo hơn). Quyết sau khi có số liệu S1/S2.

## Trạng thái code (đã smoke test toàn bộ trên CPU)

- `tools/gen_sr_pairs.py` — sinh cặp HR/LR + mặt nạ (render kép), tái dùng
  font/corpus/nền/tone-balance của gen_synth
- `viettdr/sr_model.py` — VietSR + SRLoss (mask-weighted) + PSNR/D-PSNR
- `train_sr.py` — 3 trục loss (--lam-mask, --rec-ckpt/--lam-rec), AMP,
  resume, chọn best theo D-PSNR
- `eval_sr.py` — chế độ `synth` (PSNR/SSIM/D-PSNR + acc HR/bicubic/SR) và
  `real` (delta word acc trên crop thật ≤ max-h)
