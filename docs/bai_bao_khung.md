# Khung bài báo — 桂林电子科技大学学报

Toàn bộ số liệu dưới đây là **kết quả thật** đã chạy, không có chỗ trống.
Bạn chỉ cần viết văn xuôi nối các phần lại.

---

## 题目

**《数据还是结构？低分辨率越南语场景文字识别的实证研究》**

Tiếng Anh: *Data or Architecture? An Empirical Study on Low-Resolution
Vietnamese Scene Text Recognition*

> Vì sao chọn đề tài dạng thực nghiệm (实证研究) thay vì "đề xuất phương
> pháp": số liệu của ta cho thấy các cải tiến kiến trúc **không** trụ được
> ở ngân sách huấn luyện đầy đủ. Viết theo dạng "đề xuất phương pháp" sẽ
> phải giấu điều đó, và phản biện chỉ cần hỏi "pretrain bao lâu" là lộ.
> Dạng thực nghiệm biến chính điều đó thành đóng góp.

## 摘要 (200–250 chữ)

Bối cảnh: chữ Việt mang nhiều dấu, nhận dạng cảnh tự nhiên khó. Vấn đề:
các nghiên cứu thường đề xuất thiên kiến kiến trúc cho dấu thanh nhưng
đánh giá ở ngân sách huấn luyện hạn chế. Công việc: xây dựng recognizer
phân rã ba thành phần và module siêu phân giải nhận thức dấu, rồi khảo sát
có kiểm soát ở **hai mức ngân sách**. Kết quả: (1) công thức sinh dữ liệu
cân bằng thanh điệu + khớp phân bố độ phân giải nâng độ chính xác từ
72.67% lên **82.0%** (+9.3 điểm); (2) ở ngân sách thấp, ba cơ chế kiến
trúc đóng góp +1.33 ~ +5.74 điểm, nhưng ở ngân sách đầy đủ **cả ba trở nên
không phân biệt được với nhiễu**; (3) đề xuất **mặt nạ dấu sinh bằng render
kép** và chỉ số **D-PSNR**, cải thiện +1.05 dB vùng dấu; (4) **loss thành
phần** từ recognizer phân rã cải thiện +5.4 điểm so với SR thường và là
biến thể duy nhất vượt bicubic trên miền khớp; (5) chỉ ra rằng khi
recognizer đã thích nghi miền, SR gắn ngoài vẫn thua bicubic trên ảnh thật.

关键词: 越南语;场景文字识别;变音符号;超分辨率;合成数据;消融实验

---

## 1 引言

- 1.1 Bối cảnh, ứng dụng.
- 1.2 Đặc thù chữ Việt: 229 lớp ký tự tổ hợp so với 36 của tiếng Anh; dấu
  nằm trên/dưới thân chữ; cặp hỏi/ngã khác nhau vài pixel.
- 1.3 Khoảng trống: khảo sát arXiv:2506.05061 chưa ghi nhận phương pháp
  phân rã nào cho nhận dạng tiếng Việt; và **hầu như không nghiên cứu nào
  kiểm chứng cải tiến ở nhiều mức ngân sách huấn luyện**.
- 1.4 Đóng góp (5 gạch như trong 摘要).

## 2 相关工作

- 2.1 STR: CRNN → ASTER → ABINet → PARSeq → SVTR (đều coi ký tự tổ hợp là
  lớp nguyên tử).
- 2.2 Phân rã ký tự: jamo tiếng Hàn (arXiv:2208.06079) — cấu trúc ghép
  tuần tự, khác thuộc tính trực giao của tiếng Việt; patent US6920247B1
  tách dấu mức pixel trong OCR cổ điển; tách dấu bằng connected component
  trong viết tay tiếng Việt (JST-UD; VNOnDB/ICFHR 2018).
- 2.3 Siêu phân giải văn bản: TSRN/TextZoom, TPGSR, TATT — **đều dựa trên
  TextZoom tiếng Anh; tiếng Việt không có bộ cặp LR/HR thật**.
- 2.4 Chữ Việt cảnh tự nhiên: VinText (CVPR 2021), SwinTextSpotter,
  DeepSolo, LRANet++, ViTextCaps (arXiv:2604.27712).

## 3 方法

### 3.1 三元分解识别器 (recognizer nền)

Mỗi ký tự → (chữ gốc, dấu nguyên âm, thanh điệu). Bảng thống kê:

| | Số lớp |
|---|---|
| Ký tự tổ hợp (cách truyền thống) | 228 |
| Chữ gốc × dấu × thanh | 95 × 5 × 6 = 2850 |
| **Tổ hợp hợp lệ về ngôn ngữ học** | **229 (8.0%)** |

Độ thưa lớp trên VinText train (số liệu thật, `docs/class_stats.json`):
**82/228 lớp có dưới 50 mẫu**, 2 lớp (ẽ, ỵ) không có mẫu nào.

Kiến trúc: ViT encoder (32×128 → 8×16 token, dim 384, 8 tầng) + decoder tự
hồi quy 2 tầng + 3 đầu phân loại + TripleEmbedding. Tiên nghiệm vị trí
thanh điệu qua cổng σ(τ), τ₀=−2. Ràng buộc tổ hợp:
L_comp = −log Σ_hợp lệ p_b·p_m·p_t.

### 3.2 数据合成流水线 (đóng góp lớn nhất về số liệu)

Ba thành phần, mỗi cái đều có số đo:

**(a) Cân bằng thanh điệu.** Phân bố thật rất lệch: ngang 79.550, sắc
3.726, huyền 3.163, nặng 3.153, hỏi 1.618, **ngã chỉ 760 (0.8%)**. Lấy mẫu
theo nhóm thanh với tỷ lệ trộn 0.5.

| | ngã | hỏi | thuần ASCII | chứa chữ số |
|---|---|---|---|---|
| VinText thật | 0.8% | 1.8% | 38.7% | 10.0% |
| Cân bằng 100% | 4.0% | 4.2% | **12.0%** ✗ | 6.5% |
| **Trộn 0.5 (dùng)** | **2.3%** | **2.7%** | **23.3%** | 13.6% |

→ Nêu rõ vì sao không cân bằng 100%: nó bóp nghẹt chữ Latin không dấu.

**(b) Khớp phân bố độ phân giải.** Lấy mẫu chiều cao từ chính crop thật:

| | p10 | p50 | p90 |
|---|---|---|---|
| VinText thật | 13 px | 29 px | 110 px |
| Sinh với cỡ chữ cố định | 25 | 37 | 50 |
| **Sinh khớp phân bố (dùng)** | **12** | **25** | **88** |

**(c) Nền cắt từ ảnh cảnh gốc** (35.3% số mẫu, đo thực nghiệm).

### 3.3 变音符号感知超分辨率 (DASR)

**Mặt nạ dấu sinh bằng render kép** (điểm mới cốt lõi): render mỗi từ hai
lần y hệt nhau — một bản đủ dấu, một bản chỉ chữ gốc (ệ→e, ắ→a, đ→d);
hiệu pixel = mặt nạ chính xác vùng dấu, **không cần chú thích thủ công**.
Mặt nạ chiếm 0.3–3.4% diện tích, nên loss thường bỏ qua chúng.

L_SR = Charbonnier(SR, HR) + λ_m · Charbonnier_masked + λ_rec · L_component

trong đó L_component là loss ba đầu của recognizer **đóng băng** chạy trên
ảnh SR — đầu thanh điệu cấp gradient nhắm thẳng vào pixel dấu.

**Chỉ số D-PSNR**: PSNR tính riêng trong mặt nạ (Hình: minh họa HR / LR /
mặt nạ chồng lên HR — đã có `samples/sr_pairs_check.jpg`).

**Cặp thật tự tạo**: crop thật ≥28px làm HR (12.548 mẫu), tự suy giảm
thành LR — thay cho TextZoom tiếng Việt không tồn tại.

---

## 4 实验

### 4.1 数据集与协议

VinText, cắt thành ảnh từ: **24.838 / 6.922 / 9.789** (train/val/test),
script cắt tất định công bố kèm bài — **benchmark crop-recognition đầu tiên
cho VinText**. Chỉ số: word accuracy, 1−NED, và bốn tập con:

| Tập con | Định nghĩa | Cỡ (test) |
|---|---|---|
| any_diac | có bất kỳ dấu nào | 6.023 (61.5%) |
| multi_diac | ≥2 ký tự mang dấu | 771 (7.9%) |
| confusable | chứa hỏi hoặc ngã | 903 (9.2%) |
| stacked | ký tự chồng dấu (ế ệ ữ) | 2.320 (23.7%) |

> Ghi chú phương pháp luận đáng viết: định nghĩa quen dùng ">2 ký tự mang
> dấu" chỉ chọn được **62/9789 = 0.6%** mẫu, quá nhỏ để kết luận.

### 4.2 表1 — 消融：两个训练预算下的对比 (bảng cốt lõi)

| Cấu hình | Ngân sách thấp (200k, 6 ep) | Ngân sách đủ (500k, 15 ep) |
|---|---|---|
| full | 0.7267 | 0.8196 |
| − phân rã (flat) | 0.7134 (**−1.33**) | 0.8194 (−0.02) |
| − tone prior | 0.6693 (**−5.74**) | 0.8213 (+0.17) |
| − composition loss | 0.6886 (**−3.81**) | (chưa chạy) |

Phân tích ý nghĩa thống kê: với n=9789 và p≈0.82, sai số chuẩn ≈ 0.39
điểm, nên **chênh lệch dưới ~1.1 điểm không có ý nghĩa**. Ở ngân sách thấp
các chênh lệch vượt xa ngưỡng này; ở ngân sách đủ thì không.

**Kết luận 1**: thiên kiến kiến trúc cho dấu thanh thay thế được bằng dữ
liệu, và nghiên cứu chỉ đánh giá ở một mức ngân sách dễ đưa ra kết luận sai.

### 4.3 表2 — 错误来源分析 (biện minh cho việc chuyển sang SR)

Từ ma trận nhầm thanh điệu của model 82%:

- Lỗi thanh điệu: **443 / 32.208 ký tự = 1.38%**
- Từ sai: 1.749 / 9.789 = 17.9%
- → thanh điệu giải thích **tối đa 25%** số lỗi; **≥75% không phải do dấu**
- Trong khi **25% crop thật cao ≤16px** (2.177/9.789)

### 4.4 表3 — 超分辨率：图像质量 (3.000 cặp)

| | PSNR | SSIM | D-PSNR |
|---|---|---|---|
| bicubic | 21.59 | 0.7038 | 17.95 |
| S0 (chỉ pixel) | **24.73** | **0.8423** | 19.87 |
| S1 (+mặt nạ dấu) | 24.64 | 0.8318 | **20.92** |

**Kết luận 2**: mặt nạ dấu đổi 0.09 dB toàn ảnh lấy **+1.05 dB vùng dấu**
— đúng hành vi thiết kế.

### 4.5 表4 — 超分辨率：下游识别 (bảng thuyết phục nhất)

Trên cặp tổng hợp (miền khớp huấn luyện):

| Ảnh vào recognizer | word acc | hỏi/ngã |
|---|---|---|
| HR gốc (trần trên) | 0.9317 | 0.9525 |
| bicubic | 0.7970 | 0.7961 |
| M1 (mặt nạ, không loss thành phần) | 0.7533 | 0.7575 |
| **M2 (+loss thành phần)** | **0.8070** | **0.8155** |

**Kết luận 3**: M1 vs M2 chỉ khác một biến → loss thành phần đáng **+5.4
điểm**, và M2 là biến thể duy nhất vượt bicubic.

### 4.6 表5 — 真实低分辨率裁剪 (kết quả âm tính, phải báo cáo)

Crop thật ≤16px, 2.177 mẫu:

| | word acc | Δ vs bicubic |
|---|---|---|
| **bicubic** | **0.7184** | — |
| S2 (chỉ synth) | 0.6302 | −8.8 |
| M1 (cặp thật, không rec loss) | 0.5912 | −12.7 |
| M2 (cặp thật + rec loss) | 0.6647 | −5.4 |

**Kết luận 4**: khi recognizer đã được huấn luyện trên chính phân bố crop
thật độ phân giải thấp, module SR gắn ngoài đẩy đầu vào ra khỏi phân bố
quen thuộc và **không bù lại được**. Đây là cảnh báo có giá trị: nhiều bài
TSR chỉ so trên TextZoom mà không so với recognizer đã thích nghi miền.

## 5 讨论

- Vì sao dữ liệu thắng kiến trúc: cả hai đều nhắm vào cùng một thiếu hụt
  (độ thưa lớp / thiếu mẫu khó), nhưng dữ liệu giải quyết trực tiếp hơn.
- Giới hạn: một bộ dữ liệu, một họ kiến trúc, phép suy giảm nhân tạo chưa
  mô phỏng hết suy giảm cảnh thật.
- Hướng tiếp: huấn luyện đồng thời SR + recognizer (co-adaptation).

## 6 结论

Nhắc lại 4 kết luận, nhấn mạnh khuyến nghị phương pháp luận: **mọi đề xuất
kiến trúc cho ngôn ngữ ít tài nguyên nên được kiểm chứng ở ít nhất hai mức
ngân sách huấn luyện.**

---

## Hình cần vẽ

1. Kiến trúc tổng thể (recognizer + DASR).
2. Minh họa phân rã ký tự (ệ → e + ^ + nặng).
3. **HR / LR / mặt nạ chồng HR** — đã có `samples/sr_pairs_check.jpg`.
4. So sánh ảnh thật vs tổng hợp — đã có
   `samples/so_sanh_that_vs_tong_hop.jpg`.
5. Đường cong huấn luyện hai ngân sách — từ `logs/*.log` bằng
   `tools/parse_kaggle_log.py`.
6. Ma trận nhầm thanh điệu (heatmap) — số liệu trong `docs/ket_qua*.md`.

## Việc còn thiếu (không bắt buộc)

- `no_comploss` ở ngân sách đủ để hoàn thiện ô trống bảng 1 (~6h GPU).
- Baseline ngoài (CRNN/VietOCR) nếu phản biện đòi so sánh với phương pháp
  khác (~3h).
