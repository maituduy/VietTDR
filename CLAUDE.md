# VietTDR / DASR — bối cảnh dự án

> File này được Claude Code tự nạp khi mở thư mục. Nó là nguồn context duy
> nhất khi làm việc trên máy/server mới. Cập nhật nó mỗi khi trạng thái dự
> án đổi.

## 1. Người dùng & mục tiêu

MAI TU DUY (23032903201), học viên cao học 桂林电子科技大学 (GUET), hướng
dẫn 邓珍荣. Đề tài: phát hiện và nhận dạng chữ Việt trong cảnh tự nhiên.
Giao tiếp bằng **tiếng Việt** (có trộn thuật ngữ học thuật tiếng Trung).

Điều kiện tốt nghiệp: luận văn LaTeX + **2 công trình** + 1 bài báo (学报
trường được chấp nhận) **hoặc** 1 bằng sáng chế (专利).

- **Công trình 1** — DeepSolo cải tiến (MFSA/SEI/CDN) trên VinText. Đã
  xong, luận văn đã viết, đã qua phản biện kín. Code ở
  `d:\yanjiusheng\DeepSolo\DeepSolo-main\DeepSolo` (chưa đưa lên GitHub).
- **Công trình 2** — chính là repo này. Xem mục 2.

## 2. Trạng thái hiện tại (chốt sau vòng 4, 2026-08-10)

Repo này đã đi qua hai giai đoạn:

**Giai đoạn A — recognizer phân rã ba thành phần** (base × dấu nguyên âm ×
thanh điệu, 229/2850 tổ hợp hợp lệ):
- Vòng 1 (200k synth, 6 epoch): 72.67%, mọi ablation đều dương
  (tone-prior +5.74).
- Vòng 2 (500k synth cân bằng thanh + khớp chiều cao, 15 epoch): **82.0%**,
  nhưng **cả ba điểm mới kiến trúc trở nên không phân biệt được về mặt
  thống kê** với ablation (~0.82 ± 0.4 điểm mỗi cái). Dữ liệu đã thay thế
  các thiên kiến quy nạp.

**Giai đoạn B — DASR (siêu phân giải nhận thức dấu)**, chốt hướng
2026-08-06 sau phân tích lỗi: lỗi thanh điệu chỉ 1.38% ký tự (≤25% lỗi
từ), trong khi **25% crop thật cao ≤16px**. Ba điểm mới: ① mặt nạ dấu tự
sinh bằng render kép, ② recognizer phân rã đông cứng làm loss thành phần,
③ D-PSNR + đánh giá hạ nguồn trên crop thật (không cần cặp LR/HR thật).

**Kết luận đã chốt** (bảng đầy đủ: `docs/ket_qua_dasr.md`):
- Mặt nạ dấu (①) có tác dụng đo được ở mức pixel: **+1.05 dB D-PSNR**.
- Loss thành phần (②) là khác biệt duy nhất giữa M1 và M2: **+5.4 điểm**
  word acc; M2 là biến thể **duy nhất vượt bicubic** trên miền khớp
  huấn luyện (+1.0).
- Nhưng trên **crop thật ≤16px, mọi biến thể SR đều thua bicubic**
  (tốt nhất −5.4 điểm), vì recognizer vốn đã được train trên chính phân bố
  crop thật độ phân giải thấp.
- Kết luận lớn, nhất quán qua 4 vòng: **pipeline dữ liệu (+9.3 điểm) thắng
  mọi bổ sung kiến trúc/tiền xử lý.**

## 3. Việc tiếp theo

1. **Viết bài báo empirical study** (学报):《面向低分辨率越南语场景文字的
   变音符号感知超分辨率方法》— động cơ từ phân tích lỗi định lượng, phương
   pháp ①②③, ablation S0/S1/S2, và phần "bài học từ recognizer" (vòng 1 vs
   vòng 2) làm phân tích phụ trung thực. Khung có sẵn ở
   `docs/bai_bao_khung.md`, dàn ý ở `docs/dan_y_bai_bao.md`.
2. **Hoàn thiện đơn sáng chế** cho phương pháp mặt nạ render kép + loss
   thành phần, phạm vi hiệu quả ghi **trung thực** (có tác dụng ở miền
   khớp huấn luyện / mức pixel, không thổi phồng kết quả trên ảnh thật).
   ⚠ Bản thảo **KHÔNG nằm trong repo public này** — đã chuyển sang repo
   luận văn (private) tại `毕业论文/专利/`. Đừng đưa nội dung sáng chế vào
   repo public trước khi nộp đơn: công bố sớm làm mất tính mới.

## 4. Bố cục code

```
viettdr/            lõi: vietchar.py (phân rã ký tự), tokenizer.py,
                    model.py (VietTDR + TonePriorAttention), dataset.py,
                    metrics.py, sr_model.py (VietSR + SRLoss + D-PSNR)
train.py            huấn luyện recognizer
train_sr.py         huấn luyện SR — 3 trục loss (--lam-mask, --rec-ckpt/--lam-rec)
eval.py / eval_sr.py  eval_sr có 2 chế độ: `synth` và `real`
tools/              gen_synth.py, gen_sr_pairs.py (render kép), crop_vintext_words.py,
                    fetch_fonts.py, parse_kaggle_log.py (sinh bảng kết quả)
docs/               kế hoạch, kết quả từng vòng, quy trình Kaggle
ckpt/               checkpoint qua Git LFS (xem mục 5)
viettdr_kaggle_v4.ipynb   notebook Kaggle mới nhất
```

## 5. Dữ liệu KHÔNG nằm trong git

`.gitignore` loại các thứ sau — phải chuẩn bị lại trên máy mới:

| Thứ | Cách lấy lại |
|---|---|
| `data/vintext_words/{train,val,test}/` (41k ảnh crop) | chạy `tools/crop_vintext_words.py` trên bộ VinText gốc |
| `data/synth/` | `tools/gen_synth.py` |
| `data/sr_pairs/` | `tools/gen_sr_pairs.py` |
| `vintext_words.zip` (154MB) | đóng gói lại từ `data/vintext_words/` |

Nhãn (`data/vintext_words/*_gt.jsonl`), font, và **checkpoint** thì **có**
trong repo. `ckpt/*.pth` đi qua **Git LFS** → trên máy mới phải
`git lfs install` **trước** khi clone, nếu không chỉ nhận được file con trỏ.

Checkpoint sẵn có: `rec_full_slim.pth` (recognizer 82.0%, dùng làm prior
đông cứng cho SR), `sr_s0_best.pth`, `sr_s1_best.pth`, `sr_s2_best.pth`.

## 6. Quy trình Kaggle — các bẫy đã trả giá

Chi tiết: `docs/quy_trinh_kaggle.md`. Tóm tắt những chỗ từng mất dữ liệu:

- **Người dùng tự thao tác trên web Kaggle.** Không chạy lệnh ghi qua
  Kaggle CLI (`datasets create/version`, `kernels push`) — chỉ đọc trạng
  thái/log khi được yêu cầu, và chỉ khi có token.
- Notebook `maituduy/viettdr-round-2`; dataset `viettdr-data`,
  `viettdr-code-v4`, `vintext-train-images`.
- `machine_shape: NvidiaTeslaT4` — **viết hoa chữ N**; viết thường sẽ âm
  thầm cấp P100 không tương thích.
- **Thư mục làm việc phải ở `/tmp`, KHÔNG phải `/kaggle/working`.** Hơn
  500k file trong `/kaggle/working` khiến Kaggle **bỏ toàn bộ output** —
  hai vòng đầu mất sạch checkpoint theo đúng cách này.
- Quota GPU 30h/tuần.

## 7. Môi trường

Máy local Windows không có GPU (Python 3.12, torch CPU chỉ để smoke test).
Huấn luyện chạy trên server Linux từ xa hoặc Kaggle T4. Hướng dẫn kết nối
server lab: `docs/ket_noi_server_lab.md`.

Cài đặt: `pip install -r requirements.txt`.
Smoke test nhanh: `pytest tests/` (chạy được trên CPU).
