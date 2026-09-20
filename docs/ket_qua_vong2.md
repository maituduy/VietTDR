# Kết quả VietTDR

## Vòng 2 phiên A (full, flat)

Thời lượng phiên: **11.71 giờ**. Số cấu hình đánh giá: **2**.

| Cấu hình | word acc | word acc (ci) | 1−NED | có dấu | ≥2 dấu | hỏi/ngã | chồng dấu | lỗi |
|---|---|---|---|---|---|---|---|---|
| `full` | 0.8196 | 0.8301 | 0.9104 | 0.8567 | 0.8508 | 0.8184 | 0.8526 | 9789 |
| `flat` | 0.8194 | 0.8300 | 0.9093 | 0.8639 | 0.8573 | 0.8394 | 0.8539 | 9789 |

Chênh lệch so với `full` (điểm phần trăm):

- `flat`: word +0.02, 1−NED +0.11

**full** — ma trận nhầm thanh điệu (hàng = đúng, cột = dự đoán)

| |ngang|sắc|huyền|hỏi|ngã|nặng| đúng |
|---|---|---|---|---|---|---|---|
| **ngang** |27433|41|38|22|7|28| 99.5% |
| **sắc** |35|1297|19|25|3|2| 93.9% |
| **huyền** |47|16|1139|30|3|6| 91.8% |
| **hỏi** |25|18|18|517|4|2| 88.5% |
| **ngã** |7|7|12|3|250|1| 89.3% |
| **nặng** |30|6|6|1|3|1128| 96.1% |

Nhầm hỏi↔ngã: hỏi→ngã 4, ngã→hỏi 3, **tổng 7**

**flat** — ma trận nhầm thanh điệu (hàng = đúng, cột = dự đoán)

| |ngang|sắc|huyền|hỏi|ngã|nặng| đúng |
|---|---|---|---|---|---|---|---|
| **ngang** |27393|36|31|27|6|23| 99.6% |
| **sắc** |40|1283|15|25|7|4| 93.4% |
| **huyền** |40|12|1150|28|10|3| 92.5% |
| **hỏi** |20|11|21|528|4|4| 89.8% |
| **ngã** |8|6|7|5|254|1| 90.4% |
| **nặng** |33|3|5|0|2|1130| 96.3% |

Nhầm hỏi↔ngã: hỏi→ngã 4, ngã→hỏi 5, **tổng 9**

Thời gian huấn luyện từng chặng:

- chặng 1: 15 epoch, 1076s/epoch, tổng 4.48h, val cuối 0.7861
- chặng 2: 30 epoch, 154s/epoch, tổng 1.29h, val cuối 0.8162
- chặng 3: 15 epoch, 1061s/epoch, tổng 4.42h, val cuối 0.7925
- chặng 4: 30 epoch, 152s/epoch, tổng 1.27h, val cuối 0.8156

## Vòng 2 phiên B (no_toneprior)

Thời lượng phiên: **6.20 giờ**. Số cấu hình đánh giá: **1**.

| Cấu hình | word acc | word acc (ci) | 1−NED | có dấu | ≥2 dấu | hỏi/ngã | chồng dấu | lỗi |
|---|---|---|---|---|---|---|---|---|
| `no_toneprior` | 0.8213 | 0.8328 | 0.9121 | 0.8629 | 0.8599 | 0.8328 | 0.8578 | 9789 |

**no_toneprior** — ma trận nhầm thanh điệu (hàng = đúng, cột = dự đoán)

| |ngang|sắc|huyền|hỏi|ngã|nặng| đúng |
|---|---|---|---|---|---|---|---|
| **ngang** |27420|40|40|25|5|22| 99.5% |
| **sắc** |31|1304|12|27|7|2| 94.3% |
| **huyền** |48|12|1148|24|10|2| 92.3% |
| **hỏi** |24|14|25|515|7|1| 87.9% |
| **ngã** |10|6|10|4|251|0| 89.3% |
| **nặng** |26|3|3|1|2|1127| 97.0% |

Nhầm hỏi↔ngã: hỏi→ngã 7, ngã→hỏi 4, **tổng 11**

Thời gian huấn luyện từng chặng:

- chặng 1: 15 epoch, 1109s/epoch, tổng 4.62h, val cuối 0.7896
- chặng 2: 30 epoch, 159s/epoch, tổng 1.33h, val cuối 0.8075
