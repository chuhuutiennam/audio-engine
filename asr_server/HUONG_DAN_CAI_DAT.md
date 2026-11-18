# HƯỚNG DẪN CÀI ĐẶT VÀ SỬ DỤNG `asr_server` TRÊN SERVER GPU

> **Áp dụng cho:** cấu hình mặc định (`asr.backend: native_vllm`, `vad.backend: silero`), giao thức `speech.proto`, server GPU **không có Internet**.
> **Trạng thái:** các script của project (`scripts/build.sh`, `scripts/update.sh`, `bin/*.sh`, `--check-config`) đã chạy thử đầu cuối với vLLM giả. **Chưa chạy trên máy GPU với vLLM và checkpoint Qwen thật**; các điểm cần xác nhận khi cài lần đầu nằm ở mục 13.

Hai thư mục trên server:

| Ký hiệu | Đường dẫn | Vai trò |
|---|---|---|
| **SRC** | `/u01/ims/namcht/audio-engine/asr_server` | Mã nguồn up lên. Chứa script build và update. Server không chạy từ đây. |
| **APP** | `/u01/ims/audio-engine` | Thư mục chạy app: `bin`, `config`, `lib`, `logs`, `model`, `data`. |

Tài liệu giả định up cả thư mục `audio-engine`, nên project nằm ở `/u01/ims/namcht/audio-engine/asr_server`. Nếu up nội dung `asr_server` thẳng vào `/u01/ims/namcht/audio-engine` thì SRC là thư mục đó; các script tự xác định vị trí của mình nên không cần sửa gì.

Trình tự cài lần đầu: chuẩn bị model (mục 3) → up source (mục 4) → kiểm tra môi trường (mục 5) → cài vào APP (mục 6) → cấu hình (mục 7) → chạy (mục 8) → kiểm tra bằng audio (mục 9) → đo WER và tải (mục 10).

---

## 1. Yêu cầu

| Hạng mục | Yêu cầu | Ghi chú |
|---|---|---|
| Hệ điều hành | Linux x86_64 | vLLM chỉ hỗ trợ Linux. |
| GPU | NVIDIA, driver khớp với bản CUDA của wheel `torch`/`vllm` | Kiểm tra bằng `nvidia-smi`. |
| Python | 3.10 trở lên | Project yêu cầu `>=3.10`. |
| Gói Python | Các gói trong `requirements.txt`, cùng `torch` (bản CUDA) và `vllm[audio]` từ 0.16.0 | Tài liệu này coi như môi trường đã có sẵn; mục 5 kiểm tra lại. Thiếu gói thì xem Phụ lục A. |
| Model | `Qwen3-ASR-1.7B`, `punctuation-restoration-small`, `silero_vad.jit` | Đặt trong `APP/model`. Server tự đặt chế độ offline và không tải gì từ mạng. |
| CPU | Silero VAD và punctuation chạy trên CPU | Số đo trên máy thử 2 nhân: VAD khoảng 2% một nhân và 4,4 MiB RAM cho mỗi stream. |
| Cổng | TCP 50051 (mặc định) | Không mã hóa, không xác thực: chỉ mở cho Gateway. |

Không cần: `qwen-asr`, `webrtcvad`, `onnxruntime`, gói Python `silero-vad`, `protoc`, `grpcio-tools`.

**Lưu ý về phiên bản vLLM.** Mức đã chốt là 0.16.0 trở lên. Tài liệu vLLM-Ascend lại ghi Qwen3-ASR-1.7B có từ vLLM gốc 0.19.0. Hai nguồn lệch nhau và chưa kiểm chứng trên máy GPU. Nếu khi nạp model vLLM báo không nhận ra kiến trúc, hãy dùng bản vLLM mới hơn.

**Chưa có số đo VRAM.** `asr.gpu_memory_utilization: 0.75` nghĩa là vLLM giữ 75% bộ nhớ của GPU. Số VRAM thực dùng đo ở mục 10.

---

## 2. Bố cục thư mục chạy app

`scripts/update.sh` tạo và cập nhật `/u01/ims/audio-engine`:

| Thư mục / file | Nội dung | Khi cập nhật |
|---|---|---|
| `bin/start.sh`, `stop.sh`, `status.sh` | Script chạy, dừng, xem trạng thái | Thay bằng bản mới |
| `bin/env.sh` | Thiết lập riêng của máy: đường dẫn Python, GPU | Tạo một lần, **không ghi đè** |
| `config/server.yaml`, `logging.yaml` | Cấu hình đang dùng | Tạo một lần, **không ghi đè** |
| `config/*.dist` | Cấu hình mặc định của phiên bản vừa cài, để so sánh | Thay bằng bản mới |
| `lib/asr_server/` | Mã của ứng dụng | Thay bằng bản mới |
| `lib/asr_server.prev/` | Mã của phiên bản ngay trước, để quay lại | Thay bằng bản vừa bị thay |
| `logs/` | Log, file pid | Không đụng tới |
| `model/` | Ba model | Không đụng tới |
| `data/` | File audio thử, kết quả đo | Không đụng tới |
| `VERSION` | Phiên bản, thời điểm build và cài, đường dẫn source | Ghi lại |

`lib/` chỉ chứa mã của `asr_server`. Các gói Python bên thứ ba nằm trong môi trường Python của server.

---

## 3. Chuẩn bị model (trên máy có Internet)

```bash
python -m pip install -U huggingface_hub
mkdir -p model

hf download Qwen/Qwen3-ASR-1.7B --local-dir ./model/Qwen3-ASR-1.7B
hf download Horizon-Labs/punctuation-restoration-small \
  --local-dir ./model/punctuation-restoration-small
```

Silero VAD là một file TorchScript. Lấy từ gói chính thức trên PyPI (lệnh đã chạy thử):

```bash
mkdir -p /tmp/silero && python -m pip download silero-vad --no-deps -d /tmp/silero
python - <<'PY'
import glob, zipfile
whl = glob.glob('/tmp/silero/silero_vad-*.whl')[0]
with zipfile.ZipFile(whl) as z, open('model/silero_vad.jit', 'wb') as out:
    out.write(z.read('silero_vad/data/silero_vad.jit'))
print('ok', whl)
PY
```

File TorchScript chứa mã thực thi: chỉ lấy từ nguồn chính thức của Silero.

Chép thư mục `model/` lên server ở bước 6.2.

---

## 4. Up source lên server

Up thư mục `audio-engine` thành `/u01/ims/namcht/audio-engine`. Khi up từ máy Windows, **không up** `asr_server/.venv` và `asr_server/.idea`: môi trường ảo tạo trên Windows không dùng được trên Linux.

```bash
ls /u01/ims/namcht/audio-engine/asr_server
# HUONG_DAN_CAI_DAT.md README.md bin config data logs pyproject.toml requirements.txt scripts src tests tools
```

Không cần cấp quyền chạy cho script trong SRC: các lệnh bên dưới gọi qua `bash`.

---

## 5. Kiểm tra môi trường

Chạy bằng user sẽ chạy app. Lệnh nào lỗi thì dừng ở đó.

```bash
# 1. GPU và driver
nvidia-smi

# 2. PyTorch thấy GPU (phải in True)
python3 -c "import torch; print(torch.__version__, torch.cuda.is_available())"

# 3. Phiên bản vLLM (phải từ 0.16.0)
python3 -c "import vllm; print(vllm.__version__)"

# 4. Các gói trong requirements.txt
python3 -c "import grpc, numpy, soxr, transformers, yaml, google.protobuf; print('ok')"
```

Nếu các gói nằm trong một môi trường riêng (venv, conda) thì thay `python3` bằng đường dẫn Python của môi trường đó, và ghi đường dẫn ấy vào `bin/env.sh` ở bước 6.3. Thiếu gói thì xem Phụ lục A.

---

## 6. Cài lần đầu vào thư mục chạy

### 6.1. Build và cài

```bash
cd /u01/ims/namcht/audio-engine/asr_server
bash scripts/update.sh
```

`update.sh` gọi `build.sh` (kiểm tra cú pháp, dựng cây thư mục trong `build/app`), rồi cài vào `/u01/ims/audio-engine`. Kết quả lần đầu:

```text
built asr_server 0.1.0 -> /u01/ims/namcht/audio-engine/asr_server/build/app
created  /u01/ims/audio-engine/config/server.yaml
created  /u01/ims/audio-engine/config/logging.yaml
created  /u01/ims/audio-engine/bin/env.sh
asr_server 0.1.0 installed in /u01/ims/audio-engine
MISSING  ASR model directory: /u01/ims/audio-engine/model/Qwen3-ASR-1.7B
MISSING  punctuation model directory: /u01/ims/audio-engine/model/punctuation-restoration-small
MISSING  Silero VAD file: /u01/ims/audio-engine/model/silero_vad.jit
copy the missing models, then start with: /u01/ims/audio-engine/bin/start.sh
```

- User chạy lệnh phải có quyền ghi vào `/u01/ims/audio-engine`. Nên chạy `update.sh` bằng chính user chạy app.
- Trong `config/server.yaml` vừa tạo, ba đường dẫn model đã trỏ sẵn vào `/u01/ims/audio-engine/model`.
- Cài vào thư mục khác: `bash scripts/update.sh --app-home /duong/dan/khac`, hoặc đặt biến `ASR_APP_HOME`.
- Muốn chạy bộ test trước khi build (cần `pytest`): `bash scripts/build.sh --test`.

### 6.2. Chép model

```text
/u01/ims/audio-engine/model/
├── Qwen3-ASR-1.7B/                    (thư mục)
├── punctuation-restoration-small/     (thư mục)
└── silero_vad.jit                     (file)
```

### 6.3. Thiết lập riêng của máy

Chỉ cần sửa `/u01/ims/audio-engine/bin/env.sh` khi Python không phải `python3` mặc định, hoặc khi cần chọn GPU:

```bash
PYTHON=/duong/dan/toi/venv/bin/python
export CUDA_VISIBLE_DEVICES=0
```

### 6.4. Kiểm tra lại

```bash
bash /u01/ims/namcht/audio-engine/asr_server/scripts/update.sh --no-build
```

Chạy lại như trên không làm đổi cấu hình; khi đã đủ model, kết quả không còn dòng `MISSING`.

---

## 7. Cấu hình

Cấu hình đang dùng là `/u01/ims/audio-engine/config/server.yaml`. Khóa lạ hoặc giá trị sai kiểu bị từ chối ngay khi khởi động. Sau mỗi lần sửa, kiểm tra:

```bash
cd /u01/ims/audio-engine
PYTHONPATH=lib python3 -m asr_server --check-config     # in ra giá trị, không nạp model, không tạo file log
```

Các khóa cần xem khi cài:

| Khóa | Mặc định sau khi cài | Khi nào sửa |
|---|---|---|
| `server.host`, `server.port` | `0.0.0.0`, `50051` | Giới hạn giao diện mạng hoặc đổi cổng. |
| `asr.model` | `/u01/ims/audio-engine/model/Qwen3-ASR-1.7B` | Model đặt ở chỗ khác. Luôn dùng đường dẫn tuyệt đối. |
| `punctuation.model` | `/u01/ims/audio-engine/model/punctuation-restoration-small` | Như trên. |
| `vad.silero_model` | `/u01/ims/audio-engine/model/silero_vad.jit` | Như trên. |
| `asr.gpu_memory_utilization` | `0.75` | GPU dùng chung với tiến trình khác thì giảm. |
| `asr.enforce_eager` | `false` | Đặt `true` để giảm VRAM (tắt CUDA graph). |
| `punctuation.device` | `cpu` | Giữ `cpu` trừ khi đã đo và cần chuyển. |
| `server.ingress_max_ms` | `3000` | Chỉ đặt `0` trên server thử khi đo tải bằng `--fast` (mục 10.2). |
| `asr.partial_max_backlog_ms` | `500` | Audio của một stream chờ quá mức này thì bỏ lần nhận dạng PARTIAL. Chỉ đặt `0` khi đo tải bằng `--fast` (mục 10.2). |
| `server.max_streams` | `64` | Là giới hạn của gRPC, **không phải** năng lực đã đo. |

Các khóa còn lại (VAD, punctuation, semantic) có chú thích ngay trong file; giữ mặc định cho lần chạy đầu.

Log cấu hình ở `config/logging.yaml`: file `logs/asr-server.log`, xoay vòng lúc nửa đêm, giữ 14 ngày. Muốn thấy các dòng `discard` và `empty final text`, đổi `root.level` thành `DEBUG`.

---

## 8. Chạy, dừng, xem trạng thái

### 8.1. Lần đầu: chạy ở tiền cảnh

Chạy ở tiền cảnh để nhìn thấy quá trình nạp model và lỗi nếu có:

```bash
cd /u01/ims/audio-engine
PYTHONPATH=lib python3 -m asr_server --config config/server.yaml
```

Các dòng log lần lượt xuất hiện khi khởi động thành công:

```text
configuration loaded from config/server.yaml
Silero VAD (CPU) from /u01/ims/audio-engine/model/silero_vad.jit, threshold=0.50
loading Qwen3-ASR via native vLLM: /u01/ims/audio-engine/model/Qwen3-ASR-1.7B
loading punctuation model: /u01/ims/audio-engine/model/punctuation-restoration-small
listening on 0.0.0.0:50051; plaintext for lab only
```

Server chỉ nhận cuộc gọi sau dòng `listening on`. Dừng bằng `Ctrl+C`.

### 8.2. Chạy nền

```bash
/u01/ims/audio-engine/bin/start.sh          # chạy nền; pid ghi ở logs/asr-server.pid
/u01/ims/audio-engine/bin/status.sh         # mã thoát 0 = đang chạy, 3 = không chạy
/u01/ims/audio-engine/bin/stop.sh           # SIGTERM, chờ tối đa 15 giây; thêm --force để SIGKILL
```

- Các script gọi được từ thư mục bất kỳ; chúng tự chuyển về `/u01/ims/audio-engine`.
- `start.sh` kiểm tra cấu hình trước, rồi trả về sau khoảng 1 giây trong khi model vẫn đang nạp. `status.sh` báo `not listening on port ... yet (models may still be loading)` cho tới khi server sẵn sàng.
- Theo dõi quá trình nạp: `tail -f /u01/ims/audio-engine/logs/asr-server.log`.
- `start.sh` đọc `bin/env.sh` trước (Python, GPU). Biến môi trường khác mà script đọc: `ASR_SERVER_CONFIG`, `ASR_SERVER_LOGGING_CONFIG`.

| File trong `logs/` | Nội dung |
|---|---|
| `asr-server.log` | Log của server, có xoay vòng. |
| `asr-server.out` | Những gì in ngoài hệ thống log (traceback lúc khởi động, output của vLLM). Không xoay vòng. |
| `asr-server.pid` | PID của tiến trình chạy nền. |

Project chưa kèm unit file `systemd`; server sẽ không tự chạy lại sau khi máy khởi động lại.

---

## 9. Kiểm tra bằng audio

Client thử nằm trong SRC. Dùng một file WAV tiếng nói thật (PCM 16 bit, mono, 8 hoặc 16 kHz), ví dụ đặt ở `/u01/ims/audio-engine/data/mau.wav`. Client gửi như Gateway: 16 kHz, chunk 80 ms, `chunk_seq` từ 0, `end_of_stream` ở chunk cuối.

```bash
cd /u01/ims/namcht/audio-engine/asr_server
PYTHONPATH=src python3 scripts/client.py --wav /u01/ims/audio-engine/data/mau.wav --realtime
```

Server ở máy khác hoặc cổng khác: thêm `--target <ip>:<cổng>`. Payload G.711 thô (không có RTP header): `--g711 <file> --codec pcma`.

Kết quả có dạng (nội dung chỉ để minh họa):

```text
[PARTIAL seq=1 utt=1:1] xin chao anh
[PARTIAL seq=2 utt=1:2] xin chao anh, em goi ...
[FINAL   seq=3 utt=1 COMMIT_REASON_SILENCE lang=vi rtp 0-21504] xin chao anh, em goi ... .
FINAL CALL: ...
```

Cần xem ở lần chạy đầu:

| Kiểm tra | Mong đợi | Nếu sai |
|---|---|---|
| Có PARTIAL rồi FINAL | Mỗi câu nói kết thúc bằng một FINAL | Không có gì: xem mục 12. |
| `lang=` | `vi` hoặc `en` | Rỗng: model báo ngôn ngữ khác, xem dòng `model reported language` trong log. |
| Nội dung `text` | Không có `language ...` hay `<asr_text>` | Còn marker: dạng output của model khác với giả định, cần sửa bộ tách marker (mục 13). |
| `seq` | Tăng 1, không nhảy số | |

Bỏ `--realtime` thì client gửi nhanh hơn thời gian thực và sẽ bị đóng bằng `RESOURCE_EXHAUSTED` khi vượt `server.ingress_max_ms`; đó là hành vi đúng.

---

## 10. Đo WER, VRAM và năng lực tải

Client đo nằm ở `SRC/tools/eval`. Nó cần thêm `soundfile`, và `nvidia-ml-py` nếu muốn đo VRAM (`tools/eval/requirements_gpu.txt`).

```bash
cd /u01/ims/namcht/audio-engine/asr_server/tools/eval
```

### 10.1. WER và VRAM

```bash
python3 eval_wav_grpc.py \
  --wav /u01/ims/audio-engine/data/ghi_am.wav \
  --truth /u01/ims/audio-engine/data/ground_truth.txt \
  --target 127.0.0.1:50051 --output-dir /u01/ims/audio-engine/data/eval_out
```

In ra WER trước và sau dấu câu, số FINAL, RTF và VRAM. Kết quả ghi vào `--output-dir`:

| File | Nội dung |
|---|---|
| `metrics.json` | WER trước và sau dấu câu, `processing_rtf`, số đo VRAM. |
| `stream_events.jsonl` | Từng kết quả: `seq`, `utterance_id`, `reason`, `ts_start`/`ts_end`, `raw_text`, `text`. |
| `asr_before_punctuation.txt`, `asr_after_punctuation.txt` | Văn bản ghép từ các FINAL. |
| `vram_samples.csv` | Mẫu VRAM theo thời gian. |

VRAM chỉ đo được khi client chạy trên chính máy GPU (`--vram-mode auto`).

### 10.2. Năng lực tải

Mọi lời gọi model được xếp hàng trên một luồng, nên N stream bận sẽ lâu gấp N lần. Mục tiêu 20 stream đồng thời cần `processing_rtf` không quá **0,035** (giải thích ở `docs/Giai_phap_fix_bug_server_hien_tai.md`, mục 7).

```bash
# 1. Trên server thử: trong /u01/ims/audio-engine/config/server.yaml đặt
#      server.ingress_max_ms: 0   và   asr.partial_max_backlog_ms: 0
#    rồi
/u01/ims/audio-engine/bin/stop.sh && /u01/ims/audio-engine/bin/start.sh   # chờ dòng "listening on" trong log

# 2. Gửi nhanh nhất có thể
python3 eval_wav_grpc.py \
  --wav /u01/ims/audio-engine/data/ghi_am.wav \
  --truth /u01/ims/audio-engine/data/ground_truth.txt \
  --fast --output-dir /u01/ims/audio-engine/data/eval_fast

# 3. Đọc processing_rtf trong /u01/ims/audio-engine/data/eval_fast/metrics.json
# 4. Đặt lại server.ingress_max_ms: 3000, asr.partial_max_backlog_ms: 500 và khởi động lại
```

Không dùng RTF của lần chạy không có `--fast`: con số đó gồm cả thời gian client chờ theo nhịp thực. Nếu để `asr.partial_max_backlog_ms` khác `0` khi chạy `--fast`, server bỏ hết các lần nhận dạng PARTIAL và RTF chỉ còn tính phần FINAL.

---

## 11. Cập nhật phiên bản

Up source mới đè lên `/u01/ims/namcht/audio-engine`, rồi:

```bash
cd /u01/ims/namcht/audio-engine/asr_server
bash scripts/update.sh --restart
```

`--restart` dừng server, thay mã, rồi chạy lại. Các stream đang mở bị rớt; Gateway phải mở lại stream. Không có `--restart` mà server đang chạy thì `update.sh` từ chối và không đổi gì.

Thứ tự `update.sh` thực hiện:

1. Build từ source.
2. Kiểm tra cấu hình đang dùng trong APP bằng mã **mới**. Không hợp lệ (ví dụ phiên bản mới bỏ một khóa) thì dừng: báo lỗi, mã cũ còn nguyên, server không bị dừng. So `config/server.yaml` với `config/server.yaml.dist` để sửa rồi chạy lại.
3. Dừng server (khi có `--restart`).
4. Thay `lib/asr_server` và các script trong `bin/`; bản cũ chuyển thành `lib/asr_server.prev`.
5. Chạy lại server (khi có `--restart`).

`config/server.yaml`, `config/logging.yaml`, `bin/env.sh`, `logs/`, `model/`, `data/` không bị đụng tới. Khóa cấu hình mới của phiên bản mới nhận giá trị mặc định; xem trong `config/server.yaml.dist`.

**Quay lại phiên bản trước:**

```bash
cd /u01/ims/audio-engine
bin/stop.sh
mv lib/asr_server lib/asr_server.bad && mv lib/asr_server.prev lib/asr_server
bin/start.sh
```

Chỉ giữ một phiên bản trước. `bin/` không được quay lại theo; nếu cần, cài lại từ source của phiên bản cũ.

Chỉ đổi cấu hình thì không cần `update.sh`: sửa `config/server.yaml`, chạy `--check-config` (mục 7), rồi `bin/stop.sh && bin/start.sh`.

---

## 12. Vận hành và xử lý sự cố

### 12.1. Dòng log thường gặp

| Dòng log | Mức | Ý nghĩa |
|---|---|---|
| `start call=... leg=... rate=...` | INFO | Một stream mới. |
| `commit call=... utterance=... reason=... words=... skipped_partials=N` | INFO | Đã phát một FINAL. `N` khác 0 nghĩa là utterance này đã bị bỏ N lần nhận dạng PARTIAL vì model chậm hơn audio. |
| `asr backlog call=...: ... ms of audio waiting; PARTIAL inference is skipped ...` | WARNING | Model không theo kịp stream này; PARTIAL thưa đi. Ghi một lần cho mỗi stream. |
| `asr error call=... consecutive=N: ...` | ERROR | Model lỗi; mất utterance đó, stream vẫn mở. |
| `audio_timeout call=...: no audio for ... ms, committing` | WARNING | Đang có utterance mở mà Gateway ngừng gửi. |
| `ingress overflow call=...` | ERROR | Audio chờ ASR vượt `server.ingress_max_ms`; stream bị đóng. |
| `protocol error: ...` | WARNING | Client vi phạm hợp đồng; stream bị đóng. |
| `chunk_seq gap ...: expected ..., got ...` | WARNING | rtpgw đã bỏ chunk vì hàng đợi gửi của nó đầy; phần audio đó mất, stream vẫn xử lý tiếp. Ghi 3 lần đầu của mỗi stream, sau đó mỗi 500 lần. |
| `rtp_timestamp jump ... chunk_seq=...: expected ..., got ...` | WARNING | `rtp_timestamp` của chunk không nối tiếp audio đã nhận; từ chunk đó `ts_start`/`ts_end` tính theo giá trị mới. Ghi 3 lần đầu của mỗi stream, sau đó mỗi 500 lần. |
| `model reported language ...` | WARNING | Model báo ngôn ngữ ngoài `vi`/`en`; `language` để rỗng. |
| `punctuation not ready after ... ms; using raw text` | WARNING | Punctuation quá hạn; kết quả dùng text chưa có dấu câu. |

### 12.2. Sự cố

| Hiện tượng | Nguyên nhân | Xử lý |
|---|---|---|
| `update.sh`: `the configuration in ... is not valid for this version` | Cấu hình trong APP có khóa lạ hoặc giá trị sai với mã mới | Sửa theo thông báo, so với `config/server.yaml.dist`, chạy lại `update.sh`. |
| `update.sh`: `asr_server is running in ...` | Server đang chạy | Thêm `--restart`, hoặc `bin/stop.sh` trước. |
| `update.sh`: `Permission denied` | User không có quyền ghi vào APP | Chạy bằng user chạy app. |
| `configuration error: ...`, mã thoát 2 | Khóa lạ hoặc giá trị sai trong YAML | Sửa theo thông báo, chạy lại `--check-config`. |
| `missing local ASR checkpoint` / `punctuation checkpoint` / `Silero VAD checkpoint` | Thiếu model hoặc sai đường dẫn | Kiểm tra `APP/model` và ba khóa đường dẫn model. |
| `asr_server exited during startup; see .../logs/asr-server.out` | Lỗi khi khởi động | Đọc `logs/asr-server.out`; chạy lại ở tiền cảnh (mục 8.1). |
| `No module named ...` khi chạy | Sai Python, hoặc thiếu gói | Đặt `PYTHON` trong `bin/env.sh`; kiểm tra lại mục 5. |
| vLLM báo không nhận ra model hoặc kiến trúc | Bản vLLM quá cũ | Nâng vLLM (lưu ý ở mục 1). |
| Hết bộ nhớ GPU khi nạp model | GPU đang bị tiến trình khác dùng, hoặc VRAM không đủ | Xem `nvidia-smi`; giảm `asr.gpu_memory_utilization`; thử `asr.enforce_eager: true`. |
| Lỗi không `bind` được `0.0.0.0:50051` | Cổng đang bị dùng | `ss -lntp \| grep 50051`; dừng tiến trình cũ hoặc đổi `server.port`. |
| `bad interpreter: ... bash^M` | Script có ký tự xuống dòng kiểu Windows | Trong SRC: `sed -i 's/\r$//' bin/*.sh bin/env.sh.example scripts/*.sh`, rồi chạy lại `update.sh`. |
| Client nhận `RESOURCE_EXHAUSTED` | Gửi nhanh hơn thời gian thực, hoặc GPU quá tải tới mức một vòng nhận dạng cho mọi stream đang bận lâu hơn khoảng 2,5 giây | Kiểm tra nhịp gửi của Gateway; tìm `asr backlog` và `skipped_partials` trong log; đo `processing_rtf` (mục 10.2). |
| Client nhận `UNAVAILABLE` | Model lỗi 3 lần liên tiếp (`asr.max_consecutive_errors`) | Tìm `asr error` trong log. |
| Client nhận `INVALID_ARGUMENT` | Vi phạm hợp đồng: thiếu `session_id`/`stream_id`, `sample_rate` sai, `task` lạ, `pcm` lẻ byte, tham số stream đổi giữa chừng. rtpgw không kết nối lại và đóng session | Tìm `protocol error` trong log. |
| Client nhận `UNIMPLEMENTED` | `task` là `realtime_translation`, `keyword` hoặc `voice_controller_emoji` | Server chỉ làm `speech_to_text`. rtpgw không kết nối lại và đóng session. |
| Có audio nhưng không có kết quả | VAD không thấy tiếng nói, hoặc utterance ngắn hơn `vad.min_voiced_ms` | Bật log `DEBUG`, tìm `discard`; kiểm tra mức âm lượng và `vad.threshold`. |

### 12.3. Mạng

Cổng gRPC không mã hóa và không xác thực. Chỉ cho Gateway truy cập (firewall hoặc `server.host` gắn vào giao diện nội bộ).

---

## 13. Cần xác nhận khi cài lần đầu trên GPU

Các điểm dưới đây chưa kiểm chứng được nếu không có máy GPU. Ghi lại kết quả để cập nhật tài liệu.

| # | Điểm cần xác nhận | Cách kiểm tra |
|---|---|---|
| 1 | Bản vLLM đang cài nạp được Qwen3-ASR-1.7B | Mục 8.1 chạy tới dòng `listening on`. |
| 2 | Output của model có dạng `language X<asr_text>...` | Mục 9: `text` không còn marker và `lang=vi`. |
| 3 | `torch.jit.load` còn dùng được với bản `torch` trên máy GPU | Mục 8.1 có dòng `Silero VAD (CPU) from ...` và không có lỗi. |
| 4 | Silero nhận đúng tiếng nói qua G.711 | Mục 9 với ghi âm cuộc gọi thật. |
| 5 | VRAM thực dùng | Mục 10.1. |
| 6 | `processing_rtf` so với ngưỡng 0,035 | Mục 10.2. |
| 7 | Chạy binary rtpgw thật nối gRPC với server | Mã giải mã, resample, chunk và mã hóa của rtpgw đã được chạy thử với server trên một cuộc gọi thật; phần truyền gRPC và jitter buffer thì chưa. Đối chiếu `ts_start`/`ts_end` trong callback với RTP gốc. |

---

## Phụ lục A. Cài gói Python khi server không có Internet

Chỉ cần khi bước kiểm tra ở mục 5 báo thiếu gói.

Trên một máy có Internet, **cùng hệ điều hành, cùng phiên bản Python và cùng bản CUDA** với server GPU:

```bash
mkdir -p wheelhouse
python -m pip download -d wheelhouse "vllm[audio]>=0.16.0"
python -m pip download -d wheelhouse -r requirements.txt
```

Chép `wheelhouse/` lên server, rồi cài `vllm` **trước** để `torch` và `transformers` theo đúng bản vLLM cần:

```bash
python3 -m pip install --no-index --find-links=/duong/dan/wheelhouse "vllm[audio]>=0.16.0"
python3 -m pip install --no-index --find-links=/duong/dan/wheelhouse -r requirements.txt
```

vLLM nên cài vào một môi trường riêng, vì bản dựng sẵn của nó gắn chặt với một phiên bản CUDA và PyTorch. Một container image đã dựng và thử sẵn (`docker save` / `docker load`) thường an toàn hơn bộ wheel rời.

Tài liệu liên quan: `README.md` (hợp đồng gRPC đầy đủ ở mục "gRPC contract"), `tools/eval/README.md`, `docs/Giai_phap_fix_bug_server_hien_tai.md`.
