# Kết nối vào server lab bằng ZeroTier (miễn phí, không cần mở cổng)

Mạng trường không mở cổng vào — đó chính là bài toán ZeroTier giải quyết. Cả
hai máy chỉ tạo kết nối **đi ra**, gặp nhau qua máy chủ điều phối rồi tự đục
lỗ NAT để nối **trực tiếp** với nhau. Không cần IP công cộng, không cần xin
mở cổng, không cần VPS trả tiền.

> Nên hỏi quản trị lab một câu trước khi cài. Nhiều trường bắt dùng VPN
> chính thức; nếu có VPN thì dùng VPN đơn giản hơn hẳn. Và luôn dùng **tài
> khoản của chính bạn** trên server, không mượn tài khoản người khác.

---

## Có hai kịch bản, chọn cái phù hợp

**Kịch bản A — cài được ZeroTier ngay trên server lab** (cần quyền sudo và
sự đồng ý của quản trị). Đây là cách gọn nhất: server vào thẳng mạng riêng,
không cần máy trung gian, không phụ thuộc máy bạn bè có bật hay không.

**Kịch bản B — không cài được trên server**, dùng máy của bạn trong trường
làm cầu nối (jump host). Máy đó phải bật khi bạn cần khởi động job.

Cả hai đều dùng chung bước 1–3 dưới đây.

---

## Bước 1 — Tạo mạng ZeroTier

1. Vào **my.zerotier.com**, đăng ký bằng **email** (đừng chọn đăng nhập
   Google, ở Trung Quốc không vào được).
2. Bấm **Create A Network**.
3. Ghi lại **Network ID** — chuỗi 16 ký tự, ví dụ `8056c2e21c000001`.

## Bước 2 — Cài ZeroTier trên các máy

**Máy Windows của bạn** và **máy Windows trong trường**: tải bộ cài ở
`zerotier.com/download`, cài xong bấm chuột phải biểu tượng ZeroTier ở khay
hệ thống → **Join Network** → dán Network ID.

**Máy Linux (server lab hoặc máy Linux trong trường)**:

```bash
curl -s https://install.zerotier.com | sudo bash
sudo zerotier-cli join <NETWORK_ID>
sudo zerotier-cli info          # phải thấy "ONLINE"
```

## Bước 3 — Duyệt thiết bị

Quay lại my.zerotier.com → mở mạng vừa tạo → kéo xuống mục **Members**. Mỗi
máy vừa join sẽ hiện một dòng. **Tích vào ô `Auth?`** cho từng máy, rồi đặt
tên cho dễ nhớ. Sau vài giây mỗi máy được cấp một IP dạng `10.147.x.x` hoặc
`172.2x.x.x` — ghi lại.

Kiểm tra: từ máy bạn, `ping 10.147.x.x` (IP máy kia). Thông là xong phần mạng.

---

## Kịch bản A — server đã ở trong mạng ZeroTier

Chỉ cần SSH thẳng. Thêm vào `C:\Users\duymt\.ssh\config`:

```
Host labgpu
    HostName 10.147.x.x            # IP ZeroTier của server
    User <tai-khoan-cua-ban>
```

Rồi `ssh labgpu`.

## Kịch bản B — qua máy bạn bè làm cầu nối

Máy bạn bè phải bật **SSH server**:

- **Nếu là Windows**: Settings → System → Optional features → Add → cài
  **OpenSSH Server**, sau đó mở PowerShell quyền admin chạy
  `Start-Service sshd; Set-Service -Name sshd -StartupType Automatic`.
- **Nếu là Linux**: `sudo apt install openssh-server` (thường có sẵn).

Rồi cấu hình nhảy hai chặng trong `C:\Users\duymt\.ssh\config`:

```
Host caunoi
    HostName 10.147.x.x            # IP ZeroTier của máy bạn bè
    User <tai-khoan-may-ban-be>

Host labgpu
    HostName <ip-noi-bo-server-lab>   # ví dụ 192.168.x.x, IP trong mạng trường
    User <tai-khoan-cua-BAN-tren-server>
    ProxyJump caunoi
```

Gõ `ssh labgpu` là vào thẳng server. `scp` và `rsync` cũng dùng được ngay:

```bash
scp labgpu:/duong/dan/results.zip .
```

---

## Kiểm tra tốc độ — quan trọng ở Trung Quốc

Máy chủ điều phối của ZeroTier đặt ở nước ngoài. Nếu đục lỗ NAT thành công,
dữ liệu đi **trực tiếp giữa hai máy trong nước** nên nhanh. Nếu thất bại, nó
chuyển tiếp qua máy chủ nước ngoài và sẽ **rất chậm**.

Kiểm tra:

```bash
zerotier-cli peers
```

Nhìn cột cuối: **`DIRECT`** là tốt. Nếu thấy **`RELAY`** thì đang đi vòng.

Cách xử lý khi bị `RELAY`:

1. Bật **UPnP** trên router ở đầu nào có thể (thường là nhà bạn).
2. Thử đổi mạng ở đầu của bạn (4G/5G phát wifi thay vì mạng nhà) — một số
   nhà mạng dùng CGNAT làm hỏng việc đục lỗ.
3. Nếu vẫn `RELAY`: dựng **moon** (máy chủ điều phối phụ đặt trong nước).
   Cách này cần một máy có IP công cộng, tức là quay lại chuyện VPS — lúc đó
   cân nhắc dùng **frp** hoặc **花生壳** thì đơn giản hơn.

Ping thử để ước lượng: dưới 50 ms là đi trực tiếp trong nước; trên 200 ms
gần như chắc chắn đang chuyển tiếp qua nước ngoài.

---

## Sau khi vào được server: chạy job không sợ rớt mạng

Điểm mấu chốt khiến cách này khả thi: **chỉ cần kết nối lúc khởi động job**.
Dùng `tmux` để job tiếp tục chạy trên server kể cả khi SSH đứt.

```bash
ssh labgpu
tmux new -s viettdr          # tạo phiên làm việc
# ... chạy lệnh train ...
# Bấm Ctrl+B rồi D để thoát ra, job vẫn chạy
exit                          # ngắt SSH thoải mái
```

Lần sau vào xem tiến độ:

```bash
ssh labgpu
tmux attach -t viettdr
```

## Lưu ý cho GPU P40 của lab

P40 là kiến trúc **sm_61**, PyTorch bản mới đã bỏ hỗ trợ (đúng lỗi làm hỏng
phiên P100 trên Kaggle). Phải ghim phiên bản cũ:

```bash
python -m venv venv && source venv/bin/activate
pip install torch==2.4.1 torchvision==0.19.1 \
    --index-url https://download.pytorch.org/whl/cu121
pip install pillow numpy fonttools
python -c "import torch; print(torch.cuda.get_device_name(0), torch.rand(3).cuda())"
```

Dòng cuối phải in ra tên GPU và một tensor — nếu báo
`CUDA error: no kernel image is available` là torch vẫn quá mới.
