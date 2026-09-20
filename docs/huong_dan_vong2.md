# Hướng dẫn chạy vòng 2 — MỘT phiên duy nhất

Đã tổng kiểm tra 2026-08-04: giải nén chính bản `VietTDR.zip` sẽ upload ra
thư mục sạch rồi chạy thật toàn bộ đường đi (sinh dữ liệu có `--height-ref`
+ nền thật → pretrain `--synth-aug light` → fine-tune `--init-from` có trộn
synth → eval 4 tập con mới → nhánh `flat`). Mọi cell của notebook cũng được
kiểm tra cú pháp Python. Tất cả thông.

## Bước 1 — Upload một file

**Datasets → New Dataset** → upload `VietTDR.zip` (4.9 MB) → tên
**`viettdr-code-v4`** (bắt buộc chứa chữ `code`) → Private → Create.

Không đụng `viettdr-data` và `vintext-train-images`.

## Bước 2 — Tạo notebook

1. **Code → New Notebook → File → Import Notebook** → `viettdr_kaggle_v2.ipynb`
2. **Add Input** đủ **4 nguồn**:
   - `viettdr-data` — ảnh từ đã cắt
   - `viettdr-code-v4` — mã nguồn mới
   - `vintext-train-images` — nền thật cho ảnh tổng hợp
   - **Notebook Output vòng 1**: Add Input → tab *Your Work / Notebooks* →
     chọn notebook vòng 1 → cấp checkpoint cho cell 3
3. **Session options**: GPU **T4 x2**. Internet không cần.

## Bước 3 — Kiểm tra nhanh rồi chạy

Chạy tương tác cell 1 → 2, nhìn bảng kiểm tra phải đủ 5 dòng:

```
  [OK]    --synth-aug
  [OK]    --init-from
  [OK]    --no-decompose
  [OK]    confusable
  [OK]    stacked
```

Có `[THIEU]` = đang nạp nhầm code cũ, kiểm tra lại tên dataset.

Cell 4 phải in `nen that: /kaggle/input/vintext-train-images/...`. Nếu in
`KHONG TIM THAY` là quên Add Input ảnh gốc — vẫn chạy được nhưng mất 35%
nền thật.

Ổn rồi thì **Save Version → Save & Run All** và đi làm việc khác. Toàn bộ
4 cấu hình chạy trong một phiên, ~6.9 giờ.

## Bước 4 — Lấy kết quả

Tab **Output** → tải `results_v2.zip`. Gửi tôi hoặc để tôi tự kéo về bằng
CLI — tôi sẽ dựng bảng ablation cuối, ma trận nhầm thanh điệu, biểu đồ
huấn luyện và điền số vào bài báo + chuyên lợi.

---

## Ngân sách và chốt chặn an toàn

| Việc | Thời gian |
|---|---|
| Cell 3: đánh giá lại 4 checkpoint vòng 1 | ~0.3 h |
| Cell 4: sinh 500k ảnh tổng hợp | ~0.25 h |
| Cell 5: 4 cấu hình × (pretrain 15 ep + fine-tune 30 ep) | ~6.3 h |
| **Tổng** | **~6.9 h** |

Ước tính dựa trên tốc độ **đo được** ở vòng 1 (508 ảnh/s với augmentation
đầy đủ), nhân hệ số 5.2× của `--synth-aug light`.

Giới hạn cứng Kaggle là 12 giờ. **Phiên quá giờ bị đánh dấu thất bại và kết
quả trong `/kaggle/working` có thể mất trắng**, nên notebook có ba lớp bảo
vệ:

1. Cell 1 đặt đồng hồ `BUDGET_H = 10.0`; trước mỗi cấu hình cell 5 kiểm tra
   thời gian đã chạy, vượt mốc thì bỏ qua phần còn lại và đi thẳng xuống
   đóng gói.
2. Bốn cấu hình xếp theo độ quan trọng giảm dần (`full` → `flat` →
   `no_toneprior` → `no_comploss`), nên thiếu giờ thì mất cái ít quan trọng
   nhất.
3. Mỗi cấu hình được đánh giá ngay sau khi train xong, không dồn đến cuối.

Nếu vì lý do nào đó có cấu hình bị bỏ qua, chạy lại notebook lần nữa: cell 5
tự bỏ qua pretrain đã hoàn thành (kiểm tra `runs/pre_<tên>/last.pth`).

## Khác vòng 1 ở đâu — tóm tắt cho bài báo

1. Pretrain 15 epoch × 500k ảnh (vòng 1: 6 epoch × 200k) — sửa đúng nút
   thắt đã chẩn đoán.
2. Ảnh tổng hợp khớp phân bố độ phân giải thật (`--height-ref`):
   p10/p50/p90 = 12/25/88 px so với ảnh thật 13/29/110. Vòng 1 là 25/37/50,
   thiếu hẳn nhóm ảnh nhỏ mờ — đúng nhóm khó nhất.
3. Nền cắt từ ảnh cảnh gốc VinText (~35% số mẫu, đo được 35.3%).
4. Fine-tune trộn 50k ảnh tổng hợp chống overfit. Vòng 1 loss huấn luyện đã
   chạm sàn (0.838 so với sàn lý thuyết 0.776) nhưng val khựng ở 71.8%.
5. `--synth-aug light`: bỏ augmentation trùng lặp trên ảnh tổng hợp (chúng
   đã có sẵn mờ/nhiễu/xoay/JPEG lúc sinh) → nạp dữ liệu nhanh 5.2×
   (4.50 ms/ảnh → 0.87 ms/ảnh).
6. Bốn tập con đủ cỡ mẫu thay cho `heavy` n=62: `any_diac` 61.5%,
   `multi_diac` 7.9%, `confusable` (hỏi/ngã) 9.2%, `stacked` 23.7%.
