# Kết quả DASR — vòng 3 + vòng 4 (tổng hợp)

## Recognizer nền (đóng băng cho mọi thí nghiệm SR)
82.0% word acc trên VinText test; trên tập con crop ≤16px (2.177 mẫu):
**bicubic đạt 0.7184** — vạch mọi phương pháp SR phải vượt.

## Chất lượng ảnh trên cặp tổng hợp (3.000 cặp)

| | PSNR | SSIM | D-PSNR |
|---|---|---|---|
| bicubic | 21.6–22.2 | 0.70–0.72 | 17.9 |
| S0 (chỉ pixel) | **24.73** | **0.842** | 19.87 |
| S1 (+mặt nạ dấu) | 24.64 | 0.832 | **20.92** |
| M2 (v4: +cặp thật +λ_rec 0.1) | 23.82 | 0.796 | 19.80 |

→ Mặt nạ dấu đánh đổi 0.09dB toàn ảnh lấy **+1.05dB vùng dấu**: điểm mới ①
có tác dụng đo được ở mức pixel.

## Nhận dạng trên cặp tổng hợp (cùng miền huấn luyện)

| Ảnh vào recognizer | word acc |
|---|---|
| HR gốc (trần) | 0.932–0.941 |
| bicubic | 0.797–0.803 |
| S0 | 0.7537 |
| S1 | 0.7627 |
| M1 (v4, không loss thành phần) | 0.7533 |
| **M2 (v4, +loss thành phần)** | **0.8070** ← duy nhất VƯỢT bicubic (+1.0; hỏi/ngã +1.9) |

→ M1 vs M2 chỉ khác đúng một thứ: loss thành phần từ recognizer phân rã.
Chênh **+5.4 điểm**. Điểm mới ② có tác dụng lớn, đo sạch.

## Nhận dạng trên crop THẬT ≤16px (2.177 mẫu — chỉ số triển khai)

| Phương pháp | word acc | Δ vs bicubic |
|---|---|---|
| **bicubic (không SR)** | **0.7184** | — |
| S2 v3 (chỉ synth) | 0.6302 | −8.8 |
| M1 v4 (cặp thật, không rec loss) | 0.5912 | −12.7 |
| M2 v4, giữ tỷ lệ | 0.6514 | −6.7 |
| M2 v4, bóp 16×64 | 0.6647 | −5.4 |

→ Kể cả với cặp thật + loss thành phần, SR gắn ngoài vẫn **thua bicubic**
trên dữ liệu thật.

## Diễn giải (nhất quán với mọi số liệu)

Recognizer đã được huấn luyện trên chính phân bố crop thật (25% dữ liệu
train của nó là crop ≤16px) — nó **đã tự thích nghi miền độ phân giải
thấp**. Module SR gắn ngoài làm dịch phân bố đầu vào ra khỏi vùng
recognizer quen thuộc; loss thành phần kéo lại được (bằng chứng: M2 vượt
bicubic +1.0 trên miền khớp huấn luyện) nhưng không đủ bù lệch suy giảm
thật (blur quang học/chuyển động/nén cấp cảnh mà họ suy giảm nhân tạo
không mô phỏng hết).

Chuỗi 4 vòng thí nghiệm cho một kết luận lớn nhất quán: **với một
recognizer được huấn luyện đủ tốt trên đúng phân bố mục tiêu, cả thiên
kiến kiến trúc lẫn tiền xử lý SR đều không thêm được gì; thứ quyết định là
pipeline dữ liệu** (+9.3 điểm). Các đóng góp dương còn đứng vững: mặt nạ
render kép (+1.05dB D-PSNR), loss thành phần (+5.4 điểm trên miền khớp),
công thức sinh dữ liệu cân bằng thanh + khớp độ phân giải.

## Chi phí
Vòng 4 chỉ 0.82h (nhờ tái dùng checkpoint). Tổng 4 vòng ≈ 26h GPU.
Quota còn ~22.4h, hạn 15-08.
