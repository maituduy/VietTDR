# Dàn ý bài báo — 桂林电子科技大学学报

**Đề xuất tiêu đề (tiếng Trung):**
《基于字符三元分解与声调位置先验的越南语场景文字识别方法》

**Tiêu đề tiếng Anh:** Vietnamese Scene Text Recognition via Character
Triple Decomposition and Tone Position Prior

Độ dài mục tiêu: 6–8 trang, ~6000–8000 chữ Hán. Song song có thể nộp bản
tiếng Anh cho hội nghị/tạp chí khác nếu cần sau này.

---

## 摘要 (150–250 chữ)

Bối cảnh (chữ Việt mang nhiều dấu, từ vựng tổ hợp 229 lớp thưa trên dữ
liệu nhỏ) → phương pháp (phân rã ba thành phần + tiên nghiệm vị trí thanh
điệu + ràng buộc tổ hợp) → kết quả (word acc trên VinText, cải thiện X
điểm so với baseline một đầu, X điểm trên tập nhiều dấu).

关键词: 场景文本识别; 越南语; 变音符号; 字符分解; 注意力机制

## 1 引言

- Ứng dụng nhận dạng chữ cảnh tự nhiên; khoảng trống nghiên cứu cho tiếng
  Việt (chỉ VinText 2020, ít công trình).
- Vấn đề cốt lõi: (1) từ vựng tổ hợp phình to 229 lớp vs 36 của tiếng Anh,
  (2) dữ liệu nhỏ → lớp thưa/mất cân bằng, (3) cặp thanh dễ nhầm ỏ/õ chỉ
  khác vài pixel ở dải trên của ký tự.
- Đóng góp 3 gạch đầu dòng = 3 điểm mới.

## 2 相关工作

- 2.1 STR tổng quát: CRNN → attention seq2seq (ASTER) → ABINet, PARSeq,
  SVTR. Tất cả coi mỗi ký tự tổ hợp là một lớp nguyên tử.
- 2.2 Nhận dạng chữ có dấu/tổ hợp — trích dẫn cụ thể để tự phòng thủ
  tính mới:
  - Chữ Hàn: "Character decomposition to resolve class imbalance problem
    in Hangul OCR" (arXiv:2208.06079, 2022) — phân rã jamo tuần tự
    (đầu-giữa-cuối); khác bản chất với phân rã *thuộc tính trực giao*
    của ta; không có validity mask, không có position prior.
  - OCR cổ điển: patent US6920247B1 (2005) tách dấu ở mức pixel bằng
    phân tích ảnh — không phải học end-to-end trong không gian nhãn.
  - Viết tay tiếng Việt: tách dấu bằng connected component (JST-UD;
    VNOnDB/ICFHR 2018 ghi nhận delayed strokes) — tiền xử lý ảnh,
    không phải decoding phân rã.
  - Linguistics-informed gần đây: ViTextCaps + phonological attention
    (arXiv:2604.27712, 2026) — task captioning, không phải recognition.
  - Khảo sát arXiv:2506.05061 (2025): chưa ghi nhận phương pháp phân rã
    nào cho nhận dạng tiếng Việt → khoảng trống nghiên cứu của ta.
- 2.3 Tiếng Việt: VinText/dict-guided, các cải tiến DeepSolo (trích dẫn
  luận văn/công trình 1 nếu đã công bố).

## 3 方法 (kèm hình kiến trúc tổng thể)

- 3.1 Phân rã ba thành phần: bảng ánh xạ, thống kê **229/2850 = 8% tổ hợp
  hợp lệ**; phân tích tần suất lớp trước/sau phân rã trên VinText (vẽ
  biểu đồ long-tail: lớp hiếm nhất trong 229 lớp có < N mẫu, sau phân rã
  mỗi thành phần có ≥ M mẫu) ← **số liệu này lấy từ script thống kê, rất
  thuyết phục**.
- 3.2 Kiến trúc: ViT encoder + decoder AR, TripleEmbedding, 3 đầu song
  song (công thức).
- 3.3 Tone Position Prior Attention: công thức bias theo hàng + gate
  σ(τ), τ₀=−2 (kế thừa ý tưởng gate từ công trình 1 — tính liên tục học
  thuật tốt).
- 3.4 Huấn luyện với ràng buộc tổ hợp: L = CE_b + CE_m + CE_t + λ_c·L_comp,
  L_comp = −log Σ_valid p_b·p_m·p_t; giải mã greedy có mask hợp lệ.

## 4 实验

- 4.1 Dữ liệu & giao thức: VinText crops (24.838/6.922/9.789 — script
  cắt tất định, công bố kèm bài = benchmark crop-recognition đầu tiên
  cho VinText), tổng hợp SynthTIGER 800k; phụ (nếu xin được):
  BKAI-NAVER SoICT 2022; metrics: word acc, 1−NED, acc tập nhiều dấu
  (>2 ký tự mang dấu), ma trận nhầm thanh điệu.
- 4.2 Chi tiết cài đặt: 32×128, dim 384, enc 8 / dec 2, AdamW 7e-4,
  batch 192, P40.
- 4.3 So sánh 3 tầng (trả lời trước câu hỏi phản biện "so sánh thế nào
  cho công bằng"):
  - **Tầng 1** — retrain baseline công khai trên cùng crops, cùng
    charset 229, cùng synthetic pretrain, recipe chính thức từng repo:
    CRNN, ABINet, PARSeq, SVTR, VietOCR. Báo cáo kèm params/FLOPs.
    (Cùng phương pháp luận "所有结果均为本文复现" như luận văn.)
  - **Tầng 2** — ablation cô lập: đúng backbone VietTDR nhưng 1 đầu
    229 lớp vs 3 đầu phân rã, mọi siêu tham số giống hệt → chênh lệch
    = đóng góp thuần của phân rã.
  - **Tầng 3** — cầu nối end-to-end: cắm recognizer vào sau detector
    DeepSolo (công trình 1) → so trực tiếp với số công bố trên VinText
    (SwinTextSpotter 70.82 / DeepSolo 73.59 / luận văn 74.45 /
    LRANet++ 75.6).
- 4.4 Ablation: (a) bỏ phân rã (1 đầu 229 lớp cùng backbone — so sánh công
  bằng nhất), (b) −tone prior, (c) −composition loss, (d) −synthetic
  rebalancing. Bảng phụ + ma trận nhầm hỏi/ngã trước/sau.
- 4.5 Phân tích định tính: ảnh minh họa các từ ỏ/õ sửa đúng; visualize
  attention của nhánh tone prior (heatmap tập trung dải trên/dưới).

## 5 结论

Tóm tắt + hạn chế (chưa xử lý chữ dọc/cong mạnh) + hướng mở (tích hợp
ngược vào pipeline end-to-end của công trình 1 → khép vòng hai công trình).

---

## Việc cần làm để có đủ số liệu

1. Chạy baseline 1 đầu 229 lớp (thêm cờ `--no-decompose`? → đơn giản nhất:
   train PARSeq/SVTR công khai trên cùng crops, hoặc thêm nhánh baseline
   vào repo — đã có kế hoạch trong code).
2. Chạy full + 3 ablation trên P40 (~vài giờ mỗi run, không kèm synth;
   1 ngày nếu kèm synth).
3. Script thống kê phân bố lớp trước/sau phân rã (dùng vietchar, chạy trên
   train_gt.jsonl) → hình long-tail cho mục 3.1.
4. Heatmap attention nhánh tone (hook lấy attention weights trong
   TonePriorAttention).
