# Kết quả VietTDR

## Vòng 1 (200k synth, pretrain 6 epoch)

Thời lượng phiên: **6.76 giờ**. Số cấu hình đánh giá: **4**.

| Cấu hình | word acc | word acc (ci) | 1−NED | lỗi |
|---|---|---|---|---|
| `full` | 0.7267 | 0.7383 | 0.8571 | 2675 |
| `flat` | 0.7134 | 0.7247 | 0.8502 | 2806 |
| `no_toneprior` | 0.6693 | 0.6811 | 0.8314 | 3237 |
| `no_comploss` | 0.6886 | 0.6978 | 0.8372 | 3048 |

Chênh lệch so với `full` (điểm phần trăm):

- `flat`: word +1.33, 1−NED +0.69
- `no_toneprior`: word +5.74, 1−NED +2.57
- `no_comploss`: word +3.81, 1−NED +1.99

**full** — ma trận nhầm thanh điệu (hàng = đúng, cột = dự đoán)

| |ngang|sắc|huyền|hỏi|ngã|nặng| đúng |
|---|---|---|---|---|---|---|---|
| **ngang** |24815|51|51|17|7|34| 99.4% |
| **sắc** |51|1226|15|37|7|4| 91.5% |
| **huyền** |71|25|1072|30|11|4| 88.4% |
| **hỏi** |25|51|29|459|7|0| 80.4% |
| **ngã** |13|15|15|9|219|2| 80.2% |
| **nặng** |36|8|7|0|1|1088| 95.4% |

Nhầm hỏi↔ngã: hỏi→ngã 7, ngã→hỏi 9, **tổng 16**

**flat** — ma trận nhầm thanh điệu (hàng = đúng, cột = dự đoán)

| |ngang|sắc|huyền|hỏi|ngã|nặng| đúng |
|---|---|---|---|---|---|---|---|
| **ngang** |24274|48|55|21|6|44| 99.3% |
| **sắc** |50|1201|18|42|6|7| 90.7% |
| **huyền** |61|37|1058|29|10|8| 87.9% |
| **hỏi** |29|40|25|469|4|0| 82.7% |
| **ngã** |19|10|20|9|214|3| 77.8% |
| **nặng** |37|9|8|1|1|1090| 95.1% |

Nhầm hỏi↔ngã: hỏi→ngã 4, ngã→hỏi 9, **tổng 13**

**no_toneprior** — ma trận nhầm thanh điệu (hàng = đúng, cột = dự đoán)

| |ngang|sắc|huyền|hỏi|ngã|nặng| đúng |
|---|---|---|---|---|---|---|---|
| **ngang** |23898|84|72|21|13|56| 99.0% |
| **sắc** |84|1119|42|57|6|17| 84.5% |
| **huyền** |78|55|999|31|19|12| 83.7% |
| **hỏi** |36|69|31|428|8|2| 74.6% |
| **ngã** |16|10|25|18|197|6| 72.4% |
| **nặng** |52|15|13|3|2|1043| 92.5% |

Nhầm hỏi↔ngã: hỏi→ngã 8, ngã→hỏi 18, **tổng 26**

**no_comploss** — ma trận nhầm thanh điệu (hàng = đúng, cột = dự đoán)

| |ngang|sắc|huyền|hỏi|ngã|nặng| đúng |
|---|---|---|---|---|---|---|---|
| **ngang** |23952|64|69|16|8|31| 99.2% |
| **sắc** |61|1170|25|58|4|4| 88.5% |
| **huyền** |76|30|1044|25|12|11| 87.1% |
| **hỏi** |33|52|29|434|4|2| 78.3% |
| **ngã** |19|18|23|11|201|2| 73.4% |
| **nặng** |46|16|4|0|1|1069| 94.1% |

Nhầm hỏi↔ngã: hỏi→ngã 4, ngã→hỏi 11, **tổng 15**

Thời gian huấn luyện từng chặng:

- chặng 1: 6 epoch, 458s/epoch, tổng 0.76h, val cuối 0.4816
- chặng 2: 60 epoch, 53s/epoch, tổng 0.88h, val cuối 0.7161
- chặng 3: 6 epoch, 455s/epoch, tổng 0.76h, val cuối 0.5093
- chặng 4: 60 epoch, 52s/epoch, tổng 0.87h, val cuối 0.6996
- chặng 5: 6 epoch, 452s/epoch, tổng 0.75h, val cuối 0.4296
- chặng 6: 60 epoch, 52s/epoch, tổng 0.87h, val cuối 0.6528
- chặng 7: 6 epoch, 461s/epoch, tổng 0.77h, val cuối 0.4417
- chặng 8: 60 epoch, 53s/epoch, tổng 0.89h, val cuối 0.6849

## Vòng 2 (500k synth, pretrain 15 epoch)

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
