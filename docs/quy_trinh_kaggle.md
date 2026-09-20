# Quy trình huấn luyện VietTDR trên Kaggle

Bản hướng dẫn đầy đủ, thay thế mọi hướng dẫn trước đó.

## 0. Ba file cần có (đã có sẵn trong `d:\yanjiusheng\VietTDR`)

| File | Kích thước | Nội dung |
|---|---|---|
| `VietTDR.zip` | 4.9 MB | mã nguồn + **21 font tiếng Việt** |
| `vintext_words.zip` | 155 MB | 41.549 ảnh từ đã cắt (24.838/6.922/9.789) |
| `viettdr_kaggle.ipynb` | 20 KB | notebook 9 cell |

## 1. Tạo tài khoản và xác minh

kaggle.com → Settings → **Phone Verification** (bắt buộc để bật GPU).

## 2. Upload hai dataset

**Datasets → New Dataset**, làm hai lần:

| Tên đặt | File kéo vào |
|---|---|
| `viettdr-data` | `vintext_words.zip` |
| `viettdr-code` | `VietTDR.zip` |

Để **Private** → **Create**. Kaggle tự giải nén, chờ nó xử lý xong.

## 3. Tạo notebook

**Code → New Notebook → File → Import Notebook** → chọn
`viettdr_kaggle.ipynb`.

Panel bên phải:
- **Input → Add Input** → thêm cả `viettdr-data` và `viettdr-code`
- **Session options → Accelerator → GPU T4 x2**
- Internet: **không cần bật** (font đã nằm trong zip)

## 4. Chạy

| Cell | Việc | Thời gian |
|---|---|---|
| 1 | kiểm tra GPU | 5 giây |
| 2 | nạp code, copy dữ liệu về ổ local | 2 phút |
| 3 | kiểm thử module phân rã | 10 giây |
| 4 | sinh 200k ảnh tổng hợp | ~20 phút |
| 5 | model đầy đủ, 2 giai đoạn | ~45 phút |
| 6 | ba thí nghiệm đối chứng | ~2.2 giờ |
| 7 | đánh giá trên 9.789 ảnh test | ~10 phút |
| 8 | đóng gói kết quả | 1 phút |

Tổng **~3.5 giờ** (giới hạn một phiên là 12 giờ, quota 30 giờ/tuần).

**Khuyến nghị: Save Version → Save & Run All** — Kaggle chạy nền trên máy
chủ, không phụ thuộc trình duyệt, tắt máy đi ngủ vẫn chạy. Nhưng **hãy
chạy tương tác cell 1–3 trước** để chắc chắn đường dẫn đúng, rồi mới commit.

Chạy tương tác thì phải giữ tab mở **và** tắt chế độ ngủ của Windows
(Settings → System → Power → Screen and sleep → Never khi cắm điện).

## 5. Kiểm tra ở từng chặng

**Cell 2** phải in đủ 5 dòng `[OK]`:
```
  [OK]    --no-decompose
  [OK]    --init-from
  [OK]    GradScaler
  [OK]    --val-subset
  [OK]    1-NED
```
Nếu có `[THIEU]` → đang nạp nhầm code cũ, kiểm tra lại dataset code.

**Cell 3** phải in `ALL TESTS PASSED`.

**Cell 4** phải in bảng thanh điệu 6 nhóm, ví dụ:
```
tone buckets: level=17205, acute=5824, grave=3422, hook=2210, tilde=1237, dot=3679
```
Nếu chỉ có `level=...` → đang dùng nhầm từ điển không dấu.

**Cell 5–6**, mỗi dòng log:
```
ep  12 | loss 2.41 (b 1.62 m 0.42 t 0.44 c 0.11) | word 0.31 1-NED 0.78 heavy 0.19 | lr 6.9e-04 | 42s
```
- theo dõi **1-NED** (chính xác mức ký tự) — tăng mượt ngay từ đầu
- **word** là đường cong chữ S, nằm sát 0 lâu rồi mới bật lên; thấp ở
  epoch đầu là bình thường, đừng hoảng
- **heavy** = tập từ nhiều dấu — chỉ số quan trọng nhất cho bài báo
- epoch không đánh giá hiện `nan` (do `--eval-every 3`), đúng như thiết kế

## 6. Lấy kết quả

Sau cell 8, tải `results_viettdr.zip` (~330 MB) ở panel **Output**. Bên
trong mỗi run có `log.csv`, `eval_test.json` (kèm ma trận nhầm thanh điệu
và danh sách lỗi) và `best_slim.pth`.

## 7. Nếu phiên bị ngắt

Chạy lại cell 1, 2, rồi thêm `--resume runs/<tên>/last.pth` vào lệnh train
đang dở. Checkpoint nằm trong `/kaggle/working` nên còn nguyên.

## 8. Khi cần sửa mã nguồn

Tạo dataset mới tên **có chữ `code`** (ví dụ `viettdr-code-v4`) với
`VietTDR.zip` mới, Add Input, chạy lại cell 2. Cell 2 luôn ưu tiên dataset
có chữ `code`. **Không cần đụng tới 155 MB dữ liệu.**

---

## Phụ lục: bốn thí nghiệm và ý nghĩa

| Run | Cờ | Cô lập điểm mới |
|---|---|---|
| `full` | (mặc định) | — |
| `flat` | `--no-decompose` | ① phân rã ký tự ba thành phần |
| `no_toneprior` | `--no-tone-prior` | ② tiên nghiệm vị trí thanh điệu |
| `no_comploss` | `--lam-c 0.0` | ③ ràng buộc tổ hợp hợp lệ |

`flat` dùng **một đầu phân loại 231 lớp** (kiểu truyền thống), giữ nguyên
backbone/optimizer/dữ liệu — chênh lệch tham số chỉ 489K trên 19.5M (2.5%).

Quy trình hai giai đoạn cho **cả bốn**: pretrain 6 epoch trên synth+VinText
→ tinh chỉnh 60 epoch chỉ VinText, lr 2e-4, nối bằng `--init-from` (chỉ nạp
trọng số, reset optimizer — khác `--resume` khôi phục cả optimizer lẫn epoch).

## Phụ lục: các con số dùng cho bài báo

- Bộ sinh: **8.000 ảnh / 24 giây / 8 core** (~24 ms/ảnh), nhanh hơn
  SynthTIGER hàng chục lần.
- Cân bằng thanh điệu: ngã từ **0.8% → 4.0%** số ký tự (gấp 5 lần).
- Từ vựng: 228 ký tự tổ hợp → 95 base × 5 dấu × 6 thanh, chỉ **229/2850
  (8%)** tổ hợp hợp lệ.
- Độ thưa lớp trên VinText train: **82/228 lớp có dưới 50 mẫu**, 2 lớp
  (ẽ, ỵ) không có mẫu nào.
