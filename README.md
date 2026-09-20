# VietTDR — Vietnamese Triple-Decomposition Recognizer

Nhận dạng văn bản cảnh tự nhiên tiếng Việt (ảnh từ đã cắt) bằng **phân rã
ký tự ba thành phần**: chữ gốc × dấu nguyên âm × thanh điệu.

## Động cơ

Bộ từ vựng VinText có 229 lớp ký tự vì mỗi tổ hợp "nguyên âm + dấu" là một
lớp riêng (ế ề ể ễ ệ …), trong khi tập train chỉ có 1.200 ảnh (~25k từ) →
dữ liệu mỗi lớp cực thưa, các cặp chỉ khác dấu thanh (ỏ/õ) rất dễ nhầm.

Phân rã mỗi ký tự thành ba thành phần trực giao:

| Thành phần | Số lớp | Ví dụ |
|---|---|---|
| Chữ gốc (base) | ~95 | a A b B … đ→d 0-9 dấu câu |
| Dấu nguyên âm (modifier) | 5 | none, ^ (â), ˘ (ă), móc (ơ ư), gạch (đ) |
| Thanh điệu (tone) | 6 | ngang, sắc, huyền, hỏi, ngã, nặng |

`ệ = (e, ^, nặng)`, `ắ = (a, ˘, sắc)`, `đ = (d, gạch, ngang)`.

**Chỉ 229/2850 (8%) tổ hợp là hợp lệ về ngôn ngữ học** — ràng buộc này
được dùng làm mask lúc giải mã và làm loss phạt lúc huấn luyện.

## Ba điểm mới

1. **Giải mã phân rã ba nhánh** (`viettdr/model.py: VietTDR`): decoder tự
   hồi quy dùng chung, ba đầu phân loại song song; embedding đầu vào cũng
   là tổng ba embedding thành phần (`TripleEmbedding`).
2. **Chú ý tiên nghiệm vị trí thanh điệu** (`TonePriorAttention`): nhánh
   cross-attention riêng cho đầu thanh điệu với bias học được theo hàng
   dọc của ảnh, khởi tạo thiên về dải trên/dưới (nơi dấu thanh xuất hiện),
   gắn qua cổng tiến trình `σ(τ), τ₀=−2` (kế thừa cơ chế gate của công
   trình DeepSolo cải tiến).
3. **Ràng buộc tổ hợp hợp lệ** (`TripleLoss` + constrained greedy
   decoding): phạt khối lượng xác suất rơi vào tổ hợp không tồn tại
   (ví dụ "q + ngã"), và chỉ giải mã trong 229 tổ hợp hợp lệ.

## Cấu trúc

```
viettdr/vietchar.py    # bảng phân rã/tái tổ hợp + Charset (đã có unit test)
viettdr/tokenizer.py   # text <-> tensor ba thành phần
viettdr/model.py       # ViT encoder + AR decoder + 3 heads + tone prior
viettdr/dataset.py     # dataset ảnh từ + augmentation
viettdr/metrics.py     # word acc / 1-NED / subset nhiều dấu / ma trận nhầm thanh
tools/crop_vintext_words.py   # cắt ảnh từ từ VinText
tools/prepare_synth.py        # chuyển output SynthTIGER -> jsonl
train.py, eval.py
docs/synthetic_data.md        # công thức sinh 500k-1M ảnh tổng hợp
```

## Chạy

```bash
pip install -r requirements.txt

# 1. Cắt dữ liệu từ VinText (đường dẫn tới repo DeepSolo)
python tools/crop_vintext_words.py \
    --vintext ../DeepSolo/DeepSolo-main/DeepSolo/datasets/vintext \
    --out data/vintext_words

# 2. Train (baseline nhanh, chỉ dữ liệu thật)
python train.py --data data/vintext_words \
    --charset charset_vintext.txt --out runs/tdr_v1

# 3. Đánh giá
python eval.py --ckpt runs/tdr_v1/best.pth \
    --data data/vintext_words --split test --charset charset_vintext.txt
```

Pretrain dữ liệu tổng hợp: xem `docs/synthetic_data.md`.

## Ablation cho bài báo / chuyên lợi

| Cấu hình | Cờ |
|---|---|
| Full model | (mặc định) |
| − tone prior | `--no-tone-prior` |
| − composition loss | `--lam-c 0.0` |
| Baseline không phân rã | train một đầu 229 lớp (so sánh với PARSeq/SVTR gốc) |

Chỉ số báo cáo: word accuracy, 1-NED, accuracy trên **tập con nhiều dấu
(>2 ký tự mang dấu)** — cùng giao thức với luận văn (DeepSolo cải tiến đạt
+2.3 điểm trên tập con này), và **ma trận nhầm lẫn thanh điệu** (hỏi↔ngã).

## Phần cứng

Model ~21M tham số, ảnh 32×128 — train thoải mái trên 1× Tesla P40 24GB
(batch 192, vài giờ cho 60 epoch trên VinText; ~1 ngày nếu kèm 800k ảnh
tổng hợp).
