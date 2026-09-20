# Sinh dữ liệu tổng hợp tiếng Việt

VinText chỉ có ~25k ảnh từ để train — **không đủ** để train recognizer từ
đầu. Mọi phương pháp hiện đại (PARSeq, ABINet, SVTR, VietOCR) đều pretrain
trên hàng triệu ảnh tổng hợp rồi mới tinh chỉnh trên dữ liệu thật. Bỏ bước
này thì số liệu không so sánh được với các công bố khác.

## Bộ sinh dùng trong dự án: `tools/gen_synth.py`

Không dùng SynthTIGER vì nó chỉ render vài ảnh/giây/core — một triệu ảnh
tốn nhiều CPU-giờ hơn một phiên Kaggle có. Bộ sinh này render bằng PIL ở
mức ~1–3 ms/ảnh, tức **300k ảnh trong ~5 phút với 4 core**.

Các phép biến đổi: font ngẫu nhiên, cỡ chữ 22–46px, nền đơn sắc / gradient
/ nhiễu / cắt từ ảnh thật, màu chữ đảm bảo tương phản tối thiểu, viền và
bóng đổ, xoay ±4°, làm mờ Gauss, hạ độ phân giải rồi phóng lại (giả lập
ảnh nhỏ), nhiễu cộng, nén JPEG.

### ⚠️ Cạm bẫy: chọn đúng từ điển

VinText có hai từ điển và **chỉ một dùng được**:

| File | Có dấu? | Dùng được? |
|---|---|---|
| `vn_dictionary.txt` | **Không** — thuần ASCII | ❌ dùng sẽ tạo dữ liệu không có thanh điệu nào |
| `general_dict.txt` | Có (14.531/22.224 từ) | ✅ |
| `train_gt.jsonl` (nhãn thật) | Có | ✅ nên trộn thêm |

`gen_synth.py` đọc được cả file văn bản thường lẫn `*_gt.jsonl`.

### Cân bằng thanh điệu — một luận cứ của bài báo

Phân bố thanh điệu trong VinText rất lệch (ký tự): ngang 79.550, sắc 3.726,
huyền 3.163, nặng 3.153, hỏi 1.618, **ngã chỉ 760 (0.8%)**. Cờ
`--tone-balance` gom từ theo thanh điệu rồi lấy mẫu đều các nhóm. Đây là
điều **chỉ biểu diễn phân rã mới cho phép** — model một đầu 229 lớp không
có khái niệm "thanh điệu" để mà cân bằng — nên nó vừa là kỹ thuật vừa là
lập luận ủng hộ điểm mới số ①.

**Nhưng cân bằng 100% thì lệch phân bố khác.** Mọi tên thương hiệu, chữ
viết tắt và số điện thoại đều nằm trong nhóm thanh ngang, nên lấy mẫu đều
các nhóm sẽ bóp nghẹt chữ Latin không dấu. Số đo thực tế:

| | ngã | hỏi | thuần ASCII (không dấu) | chứa chữ số |
|---|---|---|---|---|
| VinText thật | 0.8% | 1.8% | **38.7%** | 10.0% |
| Cân bằng 100% (`--balance-ratio 1.0`) | 4.0% | 4.2% | **12.0%** ✗ | 6.5% |
| **Mặc định `--balance-ratio 0.5`** | **2.3%** | **2.7%** | **23.3%** | 13.6% |

Vì vậy mặc định chỉ **một nửa** số mẫu lấy theo nhóm thanh điệu, nửa còn
lại theo phân bố tự nhiên của corpus: thanh ngã vẫn được tăng gần 3 lần
trong khi chữ không dấu không bị bỏ rơi. Giai đoạn 2 (tinh chỉnh trên
VinText thật) khôi phục nốt phân bố thật.

### Ghi chú: từ đơn âm tiết có "vô nghĩa" không?

Không phải vấn đề, vì hai lý do đo được:

1. **99.0% nhãn thật của VinText là một âm tiết, không có dấu cách.** Tiếng
   Việt viết rời từng âm tiết, nên mỗi vùng chữ cắt ra chính là một âm
   tiết. Sinh "lẵng", "ngỏn", "rỡ" riêng lẻ là khớp đúng dữ liệu thật —
   tất cả đều là âm tiết có thật trong từ điển (ngỏn ngoẻn, lẵng hoa,
   rực rỡ).
2. **25.9% nhãn thật thậm chí không phải từ trong từ điển**: số điện
   thoại, viết tắt (ĐC:, ĐT:, GPS:), tên thương hiệu (PHARMACY, Cafe,
   Sieuthi), và mảnh chữ bị cắt đôi giữa hai vùng (Vie / iệt, LƯƠ / ng).
   Một bộ nhận dạng bám vào ngữ nghĩa sẽ sai ở một phần tư dữ liệu thật.
   Vì vậy corpus còn được trộn thêm 20% chuỗi số/mã do máy sinh.

## Quy trình

```bash
# 1. Tải font mở có hỗ trợ đầy đủ dấu tiếng Việt (tự kiểm tra glyph ễ ộ ữ ẳ)
pip install fonttools
python tools/fetch_fonts.py --out fonts

# 2. Sinh 300k ảnh (~5 phút, 4 core)
python tools/gen_synth.py --out data/synth --count 300000 \
    --fonts fonts \
    --corpus data/vintext_words/train_gt.jsonl <VinText>/general_dict.txt \
    --bg-dir <VinText>/train_images \
    --workers 4 --tone-balance

# 3. Giai đoạn 1: pretrain trên synth + VinText
python train.py --data data/vintext_words --charset charset_vintext.txt \
    --synth-jsonl data/synth/synth_gt.jsonl --synth-dir data/synth \
    --epochs 8 --out runs/pre_full

# 4. Giai đoạn 2: tinh chỉnh chỉ trên VinText (nạp trọng số, reset optimizer)
python train.py --data data/vintext_words --charset charset_vintext.txt \
    --epochs 60 --lr 2e-4 --out runs/full \
    --init-from runs/pre_full/best.pth
```

`--init-from` khác `--resume`: nó chỉ nạp trọng số model và bắt đầu lịch
learning rate mới, đúng nghĩa fine-tune. `--resume` khôi phục cả optimizer
và số epoch, dùng khi phiên bị ngắt giữa chừng.

## Nếu muốn tăng chất lượng thêm

- Nâng `--count` lên 800k–1M (thời gian sinh tăng tuyến tính, ~15 phút).
- Thêm `--bg-dir` trỏ tới ảnh VinText gốc để nền sát thực tế hơn.
- Bổ sung font trang trí/viết tay — biển hiệu Việt Nam dùng rất nhiều.
