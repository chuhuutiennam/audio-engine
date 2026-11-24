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
| Khử nhiễu (tùy chọn) | Gói `sherpa-onnx==2.0.2` và file `dpdfnet2.onnx` | Chỉ cần khi bật `noise_reduction.enabled` (mục 7.1). Chạy trên CPU, không cần GPU. |
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
| `bin/rollback.sh`, `wait_ready.sh`, `_upgrade.sh` | Quay lại phiên bản đã lưu, chờ server sẵn sàng | Thay bằng bản mới; rollback không đưa các file này về bản cũ |
| `bin/env.sh` | Thiết lập riêng của máy: đường dẫn Python, GPU | Tạo một lần, **không ghi đè** |
| `config/server.yaml`, `logging.yaml` | Cấu hình đang dùng | Tạo một lần, **không ghi đè** |
| `config/*.dist` | Cấu hình mặc định của phiên bản vừa cài, để so sánh | Thay bằng bản mới |
| `lib/asr_server/` | Mã của ứng dụng | Thay bằng bản mới |
| `lib/asr_server.prev/` | Mã của phiên bản ngay trước, để đối chiếu | Thay bằng bản vừa bị thay |
| `backup/<ngày>-<giờ>/` | Các phiên bản đã bị thay: mã, `bin/`, `config/`, `VERSION`. Tên theo giờ UTC (giờ Việt Nam trừ 7); bản lưu trước khi đổi sang UTC mang giờ địa phương, bản mới luôn được đặt tên sau chúng | Thêm một bản, giữ tối đa 10 bản |
| `backup/rolled-back/<ngày>-<giờ>/` | Các phiên bản bị thay khi rollback, để xem lại. Tên theo giờ UTC như `backup/` | Thêm một bản mỗi lần rollback, giữ tối đa 10 bản |
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

Chỉ khi dùng khử nhiễu (`noise_reduction`): tải model DPDFNet-2 16 kHz (khoảng 10 MB), kiểm tra SHA-256, và tải wheel `sherpa-onnx` cùng hệ điều hành và phiên bản Python với server:

```bash
curl -fL --retry 3 \
  'https://github.com/k2-fsa/sherpa-onnx/releases/download/speech-enhancement-models/dpdfnet2.onnx' \
  -o model/dpdfnet2.onnx
echo 'ce35d6025fc71df0ef10d1540e1b7916837bbfe5f6896deb744508d2cad487a9  model/dpdfnet2.onnx' | sha256sum -c
python -m pip download -d wheelhouse sherpa-onnx==2.0.2
```

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
├── silero_vad.jit                     (file)
└── dpdfnet2.onnx                      (file, chỉ khi bật noise_reduction)
```

### 6.3. Thiết lập riêng của máy

Lần cài đầu, `/u01/ims/audio-engine/bin/env.sh` được tạo sẵn với các biến chạy offline và `CUDA_VISIBLE_DEVICES=0`. Chỉ cần sửa file này khi Python không phải `python3` mặc định, hoặc khi cần chọn GPU khác:

```bash
PYTHON=/duong/dan/toi/venv/bin/python
export CUDA_VISIBLE_DEVICES=0
```

Máy đã cài từ trước vẫn giữ `bin/env.sh` cũ. Sau khi cập nhật, so sánh với `bin/env.sh.example` rồi thêm các dòng còn thiếu.

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
| `asr.chunk_sec` | `1.0` | Đặt `0.5` khi bật hiện từng từ. Lượng audio đưa vào model tăng khoảng 1,8–1,9 lần. |
| `asr.log_results` | `false` | Đặt `true` để ghi ngôn ngữ và text của từng PARTIAL, COMMIT vào log (mục 12.1). Chỉ bật trong lúc phân tích: log chứa nội dung cuộc gọi, và các file đã ghi vẫn còn trên đĩa tối đa 14 ngày sau khi tắt cờ. |
| `asr.default_language` | `vi` | Ngôn ngữ của stream khi rtpgw không gửi `language` hoặc gửi mã lạ: `vi`, `en` hoặc `auto` (model tự đoán). Thay cho giá trị `Vietnamese` viết cứng trước đây. |
| `asr.force_language` | `false` | Đặt `true` để ép model nhận dạng theo ngôn ngữ của stream (`vi`, `en`) thay vì tự đoán; stream `auto` vẫn tự đoán. Chỉ áp dụng cho `asr.backend: native_vllm`. Thư mục model phải có chat template (`chat_template.json`, `chat_template.jinja` hoặc trong `tokenizer_config.json`), thiếu thì server không khởi động. |
| `word_reveal.enabled` | `false` | Đặt `true` để chữ của PARTIAL hiện dần từng từ. |
| `word_reveal.window_ms` | `400` | Thời gian tối đa để hiện hết các từ mới. Không được lớn hơn `asr.chunk_sec`. |
| `word_reveal.min_interval_ms` | `120` | Khoảng cách tối thiểu giữa hai kết quả gửi đi trên một stream. |
| `asr.enabled` | `true` | Đặt `false` khi server chỉ làm khử nhiễu: không nạp Qwen3-ASR, VAD, punctuation (không cần GPU); `speech_to_text` trả `UNIMPLEMENTED`. Phải bật ít nhất một trong `asr.enabled`, `noise_reduction.enabled`. |
| `noise_reduction.*` | `enabled: false` | Khử nhiễu DPDFNet-2, xem mục 7.1. |

Các khóa còn lại (VAD, punctuation, semantic) có chú thích ngay trong file; giữ mặc định cho lần chạy đầu.

Bật hiện từng từ: sửa `asr.chunk_sec: 0.5` và `word_reveal.enabled: true` trong `config/server.yaml` của thư mục chạy, rồi chạy `bin/stop.sh` và `bin/start.sh`. `scripts/update.sh` không ghi đè file này. Muốn tắt, đặt lại hai giá trị cũ và khởi động lại.

Bật ép ngôn ngữ: **sau khi** nâng cấp xong (code cũ từ chối khóa lạ), thêm `force_language: true` vào mục `asr` của `config/server.yaml` ở thư mục chạy, đổi `default_language` nếu cần, rồi chạy `bin/stop.sh` và `bin/start.sh`. Kiểm tra trong `logs/asr-server.log` (lệnh ở cuối mục 12.1):

1. Có dòng `asr.force_language on`.
2. Sau cuộc gọi đầu tiên có dòng `model prompt language=Vietnamese ends with '...<|im_start|>assistant\nlanguage Vietnamese<asr_text>'`.
3. Các stream rtpgw gửi `vi` không còn dòng `model reported language`. Với `asr.log_results: true`, `model_output` không còn bắt đầu bằng `language ...<asr_text>`.

Muốn tắt, đặt lại `force_language: false` và khởi động lại.

### 7.1. Bật khử nhiễu (`noise_reduction`, DPDFNet-2)

Gateway gửi stream có `task=noise_reduction` khi DCSF chọn dịch vụ này. Server lọc nhiễu từng chunk rồi trả audio về cho Gateway phát tiếp; Gateway tự phát audio gốc khi server tắt dịch vụ, lỗi, hoặc không trả audio quá 500 ms. Thiết kế: `docs/DPDFNet2_Task_Routing_Design_VI.md` ở gốc repo.

| Khóa | Mặc định | Ý nghĩa |
|---|---|---|
| `noise_reduction.enabled` | `false` | `true`: phục vụ `noise_reduction`. Cần gói `sherpa-onnx` và file model. |
| `noise_reduction.model` | `/u01/ims/audio-engine/model/dpdfnet2.onnx` (máy cài mới) | File model trên đĩa; server không tự tải. |
| `noise_reduction.frame_ms` | `20` | Lượng audio mỗi kết quả trả về: `20`, `40` hoặc `80`. |
| `noise_reduction.num_threads` | `1` | Số luồng ONNX của mỗi bộ lọc. |
| `noise_reduction.workers` | `2` | Số luồng chạy model, dùng chung cho mọi stream. |
| `noise_reduction.max_backlog_ms` | `500` | Audio chờ model vượt mức này (CPU không theo kịp) thì stream bị đóng (`RESOURCE_EXHAUSTED`) và Gateway phát audio gốc đến hết cuộc gọi, thay vì nhận audio ngày càng trễ. `0` = không giới hạn; chỉ đặt `0` trên server thử khi client gửi nhanh hơn thời gian thực. Tối đa `5000`. Ngắt nhầm thì mất khử nhiễu đến hết cuộc gọi, nên đừng đặt quá thấp: Gateway có thể gửi dồn nhiều chunk sau một lần nghẽn. Với `500`, trong lúc một chunk đang qua model, đợt dồn 7 chunk 80 ms (6 chunk chờ = 480 ms) vẫn qua, 8 chunk thì bị đóng. Audio cũng dồn khi Gateway đọc kết quả chậm. |
| `noise_reduction.max_streams` | `16` | Số bộ lọc nạp sẵn lúc khởi động (mỗi bộ khoảng 20 MB RAM, 0,3 s). Stream thứ `max_streams + 1` bị từ chối ngay (`RESOURCE_EXHAUSTED`) và Gateway phát audio gốc. Không được lớn hơn `server.max_streams`. Chọn theo số đo trên CPU thật, không theo số cuộc gọi. |

Các bước:

1. Cài gói trên server: `python3 -m pip install --no-index --find-links=/duong/dan/wheelhouse sherpa-onnx==2.0.2`, rồi kiểm tra `python3 -c "import sherpa_onnx; print(sherpa_onnx.__version__, hasattr(sherpa_onnx, 'OnlineSpeechDenoiser'))"` in `2.0.2 True`.
2. Chép `dpdfnet2.onnx` vào `/u01/ims/audio-engine/model/` và kiểm tra SHA-256 như ở mục 3.
3. **Sau khi** nâng cấp lên bản có khử nhiễu (code cũ từ chối khóa lạ): máy cài mới đã có sẵn mục `noise_reduction` trong `config/server.yaml`; máy đã cài từ trước thì chép nguyên mục `noise_reduction` từ `config/server.yaml.dist` (đường dẫn model ở đó đã trỏ vào `APP/model`). Đặt `enabled: true`.
4. Chạy `--check-config`, rồi `bin/stop.sh` và `bin/start.sh`. `scripts/update.sh` chỉ kiểm tra model của các task đang bật: thiếu thì báo `MISSING  DPDFNet-2 model file: ...` hoặc `MISSING  Python package sherpa-onnx==2.0.2 ...: <lý do>` (chưa cài, bản cũ không có `OnlineSpeechDenoiser`, hoặc thư viện native không nạp được); với `asr.enabled: false` không còn đòi model ASR, punctuation, Silero. Việc kiểm tra chạy trước khi thay đổi bất cứ thứ gì: nếu server đang chạy mà thiếu model thì `update.sh --restart` dừng lại, không dừng server, không cài (chép model rồi chạy lại). Đường dẫn model tương đối được tính từ thư mục chạy app.
5. Kiểm tra trong `logs/asr-server.log`: lúc khởi động có dòng `noise_reduction on: 16 denoisers loaded in ... s`; sau mỗi cuộc gọi có dòng `noise_reduction stream ended ... outcome=final` (mục 12.1).

Muốn tắt, đặt lại `enabled: false` và khởi động lại: Gateway nhận `UNIMPLEMENTED` và phát audio gốc.

Kiểm tra trên máy cài, không cần Gateway (cần `pytest`; không cần GPU):

```bash
cd /u01/ims/namcht/audio-engine/asr_server
ASR_TEST_DPDFNET2_MODEL=/u01/ims/audio-engine/model/dpdfnet2.onnx python3 -m pytest tests/test_noise_real_model.py -s
```

Kết quả `5 passed` và một dòng `noise_reduction stream ended ...` có `first_audio_ms`, `model_ms_p95`: số đo trên CPU của máy đó.

### 7.2. Khử nhiễu trước nhận dạng (`asr_denoise`, DPDFNet-2)

Khi bật, audio của các stream `speech_to_text` đi qua DPDFNet-2 rồi mới vào VAD và ASR. Kết quả trả cho Gateway giữ nguyên dạng: không thêm trường proto, `ts_start`/`ts_end` không lệch. Mặc định tắt.

| Khóa | Mặc định | Ý nghĩa |
|---|---|---|
| `asr_denoise.enabled` | `false` | `true`: khử nhiễu trước VAD/ASR. Cần `asr.enabled: true`, gói `sherpa-onnx` và file model. |
| `asr_denoise.model` | `/u01/ims/audio-engine/model/dpdfnet2.onnx` (máy cài mới) | File model trên đĩa, dùng chung file với `noise_reduction` được. Lúc khởi động server kiểm tra độ trễ của model bằng một xung: khác 640 mẫu (40 ms) thì từ chối khởi động, vì timestamp sẽ lệch. |
| `asr_denoise.apply_to` | `wideband` | `wideband`: chỉ các cuộc gọi Gateway chạy G.722/L16 (chunk đầu có `output_audio_format: PCM_S16LE`). Cuộc gọi G.711 đi qua nguyên vẹn, vì DPDFNet-2 làm mất tiếng nói trên audio băng hẹp. `all`: mọi stream 16 kHz, chỉ dùng khi đo thử (mục 10.3). |
| `asr_denoise.num_threads` | `1` | Số luồng ONNX của mỗi bộ lọc. |
| `asr_denoise.workers` | `2` | Số luồng chạy model, dùng chung cho mọi stream. |
| `asr_denoise.max_streams` | `16` | Số bộ lọc nạp sẵn lúc khởi động (mỗi bộ khoảng 20 MB RAM). Pool này riêng, không dùng chung với `noise_reduction`. Stream không còn bộ lọc thì được nhận dạng từ audio gốc. Không được lớn hơn `server.max_streams`. |
| `asr_denoise.max_backlog_ms` | `500` | Khi một lần gọi model, hoặc lượng audio chờ model, vượt mức này, stream chuyển sang audio gốc **từ mẫu kế tiếp** đến hết stream (không mất, không lặp mẫu); bộ lọc được trả ngay cho cuộc gọi khác. Tối đa `5000`. `0` = không giới hạn, chỉ dùng trên server thử khi client gửi nhanh hơn thời gian thực (`--fast`): khi đó một lần gọi model bị kẹt sẽ giữ cả stream. |

Các bước bật:

1. Cài `sherpa-onnx==2.0.2` và chép `dpdfnet2.onnx` như mục 7.1, bước 1–2.
2. **Sau khi** nâng cấp lên bản có `asr_denoise`: máy đã cài từ trước thì chép nguyên mục `asr_denoise` từ `config/server.yaml.dist` sang `config/server.yaml`, rồi đặt `enabled: true`.
3. Chạy `--check-config`, rồi `bin/stop.sh` và `bin/start.sh`. Khi model hoặc `sherpa-onnx` còn thiếu, `scripts/update.sh` báo `MISSING  DPDFNet-2 model file (asr_denoise.model): ...` hoặc `MISSING  Python package sherpa-onnx==2.0.2 (asr_denoise.enabled; ...)`.
4. Kiểm tra `logs/asr-server.log`: lúc khởi động có dòng `asr_denoise on: N denoisers loaded ...`; sau mỗi cuộc gọi G.722/L16 có dòng `stt_denoise stream ended ... outcome=denoised`.

**Trước khi bật ở production**, phải so WER khi tắt và khi bật trên cùng bộ ghi âm băng rộng (mục 10.3). Lý do: trên 6 mẫu VoiceBank, khử nhiễu giảm VAD báo nhầm từ 31% xuống 5%, nhưng ở 2 mẫu VAD bỏ sót nhiều tiếng nói hơn (0,6% lên 8,3%; 5,6% lên 9,3%).

Về CPU, trên laptop 12 luồng, khử nhiễu chạy được khoảng 10 stream đồng thời. Trên server phải đo lại, sau khi đã trừ phần CPU ASR dùng, rồi mới chọn `max_streams` và `workers`.

Muốn tắt, đặt `enabled: false` và khởi động lại: ASR nhận audio gốc như trước.

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

### 10.3. So WER khi tắt và bật `asr_denoise`

Client đo không gửi `output_audio_format`. Vì vậy trên server thử phải đặt `asr_denoise.apply_to: all`, nếu không thì stream của client đo không được khử nhiễu. Dùng bản ghi âm **băng rộng** 16 kHz, đúng loại cuộc gọi G.722/L16 sẽ được khử nhiễu.

```bash
# 1. Tắt: asr_denoise.enabled: false; khởi động lại; đo
python3 eval_wav_grpc.py --wav /u01/ims/audio-engine/data/ghi_am.wav \
  --truth /u01/ims/audio-engine/data/ground_truth.txt --output-dir /u01/ims/audio-engine/data/eval_denoise_off

# 2. Bật: asr_denoise.enabled: true, asr_denoise.apply_to: all; khởi động lại; đo lại cùng file
python3 eval_wav_grpc.py --wav /u01/ims/audio-engine/data/ghi_am.wav \
  --truth /u01/ims/audio-engine/data/ground_truth.txt --output-dir /u01/ims/audio-engine/data/eval_denoise_on

# 3. Đặt lại asr_denoise.apply_to: wideband trước khi chạy thật
```

So `metrics.json` của hai thư mục, dùng WER trước dấu câu. Đọc thêm `stream_events.jsonl` để xem có câu nào mất hẳn, hoặc mất âm đầu/cuối câu (so `ts_start`/`ts_end` của từng FINAL). Trong log bản bật phải có `stt_denoise stream ended ... outcome=denoised`. Chỉ bật ở production khi WER không tăng và không có câu nào bị mất. Nên đo trên nhiều bản ghi có nhiễu thật (đường phố, văn phòng, xe); một file chưa đủ.

---

## 11. Cập nhật phiên bản

Up source mới đè lên `/u01/ims/namcht/audio-engine`, rồi:

```bash
cd /u01/ims/namcht/audio-engine/asr_server
bash scripts/update.sh --restart
```

`--restart` dừng server, thay mã, chạy lại rồi chờ server sẵn sàng. Các stream đang mở bị rớt; Gateway phải mở lại stream. Server đang dừng thì `--restart` cũng chạy server, trừ khi còn thiếu model. Không có `--restart` mà server đang chạy thì `update.sh` từ chối và không đổi gì.

Thứ tự `update.sh` thực hiện:

1. Build từ source.
2. Kiểm tra cấu hình đang dùng trong APP bằng mã **mới**. Không hợp lệ (ví dụ phiên bản mới bỏ một khóa) thì dừng: báo lỗi, mã cũ còn nguyên, server không bị dừng. So `config/server.yaml` với `config/server.yaml.dist.new` để sửa rồi chạy lại.
3. Kiểm tra cấu hình đó bằng mã **đang cài**. Không hợp lệ thì dừng như trên, vì bản lưu sẽ không rollback được.
4. Lưu phiên bản đang cài vào `backup/<ngày>-<giờ>/`, xóa các bản cũ vượt quá 10 bản. Bản gần nhất chạy tốt luôn được giữ lại, kể cả sau nhiều lần nâng cấp lỗi liên tiếp.
5. Dừng server (khi có `--restart`).
6. Thay `lib/asr_server` và các script trong `bin/`; bản cũ chuyển thành `lib/asr_server.prev`.
7. Chạy lại server và chờ cổng gRPC mở (khi có `--restart`). Cổng chỉ mở sau khi nạp xong model.

`config/server.yaml`, `config/logging.yaml`, `bin/env.sh`, `logs/`, `model/`, `data/` không bị đụng tới. Khóa cấu hình mới của phiên bản mới nhận giá trị mặc định; xem trong `config/server.yaml.dist`.

**Khóa cấu hình mới phải thêm sau khi nâng cấp xong.** Thêm trước thì bước 3 báo `unknown key` và dừng. Nếu mã đang cài bị hỏng nên không đọc được cấu hình, thêm `--ignore-old-config` để vẫn nâng cấp; bản lưu lần đó được đánh dấu `failed` và `rollback.sh` mặc định bỏ qua nó.

**Khi nâng cấp lỗi:** `update.sh` báo `did not become ready; it was NOT rolled back` và trả mã thoát 1. Script **không tự rollback**, để bạn xem log trước:

```bash
tail -n 100 /u01/ims/audio-engine/logs/asr-server.log
tail -n 100 /u01/ims/audio-engine/logs/asr-server.out
```

Server chưa mở cổng sau thời gian chờ vẫn được giữ chạy. Kiểm tra lại bằng `bin/wait_ready.sh`.

**Quay lại phiên bản đã lưu:**

```bash
cd /u01/ims/audio-engine
bin/rollback.sh --list                # xem các bản đã lưu
bin/rollback.sh                       # về bản đánh dấu "<- default"
bin/rollback.sh 20261119-103000       # về bản chỉ định
```

- Bản mặc định là bản mới nhất có `health=ok`. Nếu không có bản nào như vậy, script chọn bản mới nhất chưa có kết quả kiểm tra. Bản `failed` không bao giờ được chọn tự động. Muốn về bản khác, ghi rõ tên bản.
- `rollback.sh` khôi phục mã, `bin/`, `config/`, `VERSION`, chạy lại server và chờ server sẵn sàng.
- Phiên bản vừa bị thay, gồm mã và cấu hình, được giữ ở `backup/rolled-back/<ngày>-<giờ>/` để xem lại.
- Cấu hình đã lưu không hợp lệ với mã đã lưu thì script dừng và không đổi gì.
- Không quay lại được gói Python và model.

Thiết lập trong `bin/env.sh`:

| Biến | Mặc định | Ý nghĩa |
|---|---|---|
| `ASR_BACKUP_KEEP` | `10` | Số phiên bản giữ trong `backup/`, tối thiểu 1 |
| `ASR_START_WAIT_SEC` | `300` | Số giây chờ server sẵn sàng sau khi chạy lại. Đặt lớn hơn thời gian nạp model thật. `0` là chỉ kiểm tra tiến trình còn sống |

Kiểm tra này chỉ phát hiện server chết hoặc không mở cổng. Lỗi nhận dạng sai text phải tự phát hiện rồi chạy `rollback.sh`.

Chỉ đổi cấu hình thì không cần `update.sh`: sửa `config/server.yaml`, chạy `--check-config` (mục 7), rồi `bin/stop.sh && bin/start.sh`.

---

## 12. Vận hành và xử lý sự cố

### 12.1. Dòng log thường gặp

| Dòng log | Mức | Ý nghĩa |
|---|---|---|
| `start call=... leg=... rate=... codec=... language=...` | INFO | Một stream mới. `language`: ngôn ngữ của stream (`Vietnamese`, `English`) hoặc `auto`. Model chỉ bị ép theo ngôn ngữ này khi `asr.force_language: true`. |
| `asr.force_language on: ...` | INFO | Lúc khởi động, khi bật ép ngôn ngữ. |
| `model prompt language=... ends with '...'` | INFO | Chỉ khi `asr.force_language: true`, mỗi ngôn ngữ một lần: phần cuối prompt vLLM đã dựng. Ép đúng thì dòng `Vietnamese` kết thúc bằng `<\|im_start\|>assistant\nlanguage Vietnamese<asr_text>`, và phần trước đó giống dòng `auto` (nếu có). |
| `commit call=... utterance=... reason=... words=... skipped_partials=N` | INFO | Đã phát một FINAL. `N` khác 0 nghĩa là utterance này đã bị bỏ N lần nhận dạng PARTIAL vì model chậm hơn audio. |
| `asr backlog call=...: ... ms of audio waiting; PARTIAL inference is skipped ...` | WARNING | Model không theo kịp stream này; PARTIAL thưa đi. Ghi một lần cho mỗi stream. |
| `asr error call=... consecutive=N: ...` | ERROR | Model lỗi; mất utterance đó, stream vẫn mở. |
| `audio_timeout call=...: no audio for ... ms, committing` | WARNING | Đang có utterance mở mà Gateway ngừng gửi. |
| `ingress overflow call=...` | ERROR | Audio chờ ASR vượt `server.ingress_max_ms`; stream bị đóng. |
| `protocol error: ...` | WARNING | Client vi phạm hợp đồng; stream bị đóng. |
| `chunk_seq gap ...: expected ..., got ...` | WARNING | rtpgw đã bỏ chunk vì hàng đợi gửi của nó đầy; phần audio đó mất, stream vẫn xử lý tiếp. Ghi 3 lần đầu của mỗi stream, sau đó mỗi 500 lần. |
| `rtp_timestamp jump ... chunk_seq=...: expected ..., got ...` | WARNING | `rtp_timestamp` của chunk không nối tiếp audio đã nhận; từ chunk đó `ts_start`/`ts_end` tính theo giá trị mới. Ghi 3 lần đầu của mỗi stream, sau đó mỗi 500 lần. |
| `model reported language ...` | WARNING | Model báo ngôn ngữ ngoài `vi`/`en`; `language` để rỗng. Chỉ xảy ra khi model tự đoán: `asr.force_language: false`, hoặc stream `auto`. |
| `punctuation not ready after ... ms; using raw text` | WARNING | Punctuation quá hạn; kết quả dùng text chưa có dấu câu. |
| `noise_reduction on: N denoisers loaded in ... s, ... worker threads, ... ms of audio per result, max_backlog_ms=...` | INFO | Lúc khởi động, khi bật khử nhiễu. |
| `noise_reduction stream ended session=... stream=... outcome=... output=... chunks=... samples_in=... samples_out=... samples_sent=... clipped=... results=... first_audio_ms=... model_ms_p50=... model_ms_p95=... model_ms_max=... queue_ms_p95=... queue_ms_max=... backlog_ms_max=... chunk_seq_gaps=...` | INFO | Một stream khử nhiễu kết thúc. `outcome`: `final` (bình thường), `cancelled` (Gateway hủy, ví dụ khi chuyển sang audio gốc), `invalid`, `model_error`, `backlog` (audio chờ xử lý vượt `max_backlog_ms`), `read_error` (lỗi khi đọc luồng request, ví dụ server đang dừng hoặc message hỏng; Gateway mất kết nối thì ghi là `cancelled`). `backlog_ms_max`: lượng audio chờ model lớn nhất. `first_audio_ms` phải dưới 500. `model_ms`: thời gian model cho mỗi chunk 80 ms, phải nhỏ hơn 80 nhiều; `queue_ms` tăng là thiếu `workers`. `clipped` khác 0 là audio ra bị bão hòa. |
| `noise_reduction refused session=... stream=...: all N denoisers in use (M streams refused since start)` | WARNING | Hết bộ lọc; stream bị từ chối, Gateway phát audio gốc. Ghi 3 lần đầu, sau đó mỗi 100 lần. Tăng `noise_reduction.max_streams` nếu CPU còn dư. |
| `noise_reduction backlog session=... stream=...: N ms of audio waiting to be processed, above noise_reduction.max_backlog_ms ...; stream closed (M streams closed since start)` | WARNING | Audio chờ xử lý vượt ngưỡng; stream bị đóng, Gateway phát audio gốc. Ghi 3 lần đầu, sau đó mỗi 100 lần. Nếu `model_ms`, `queue_ms` của dòng tóm tắt cao thì CPU không theo kịp: giảm `max_streams` hoặc tăng `workers` nếu còn nhân CPU. Nếu chúng thấp thì nguyên nhân là Gateway gửi dồn hoặc đọc kết quả chậm. |
| `noise_reduction model failed ...` / `denoiser reset failed ...` | ERROR | Model lỗi; stream đó kết thúc (`UNAVAILABLE`), bộ lọc được nạp lại. |
| `asr_denoise on: N denoisers loaded in ... s, ... worker threads, apply_to=..., max_backlog_ms=...` | INFO | Lúc khởi động, khi bật khử nhiễu trước ASR. |
| `asr_denoise cannot start: ...` | ERROR | Thiếu model, thiếu `sherpa-onnx`, hoặc model có độ trễ khác 640 mẫu (`the denoiser delays audio by N samples`). Server không khởi động. |
| `stt_denoise stream ended session=... stream=... outcome=... reason=... chunks=... samples_in=... samples_out=... model_ms_p50=... model_ms_p95=... model_ms_max=... backlog_ms_max=...` | INFO | Một stream STT có khử nhiễu kết thúc. `outcome`: `denoised` (khử nhiễu đến hết), `fallback` (đã chuyển sang audio gốc, xem `reason`), `cancelled` (Gateway hủy hoặc mất kết nối), `error` (stream bị từ chối giữa chừng, ví dụ lỗi giao thức). `samples_out` bằng `samples_in` là không mất mẫu. `model_ms`: thời gian model cho mỗi chunk, phải nhỏ hơn độ dài chunk nhiều. |
| `stt_denoise fallback session=... stream=... reason=... at_sample=N (...)` | WARNING | Từ mẫu N, stream dùng audio gốc đến hết. `reason`: `model_error` (model lỗi, bộ lọc được nạp lại), `model_slow` (một lần gọi quá `max_backlog_ms`), `busy` (mọi luồng `workers` đều bận quá `max_backlog_ms`: thiếu `workers` hoặc CPU), `backlog` (audio chờ quá `max_backlog_ms`). Gặp thường xuyên thì giảm `max_streams` hoặc tăng `workers` nếu còn nhân CPU. |
| `stt_denoise: no denoiser free (asr_denoise.max_streams=N) ...; the stream is recognised from the audio as sent (M such streams so far)` | WARNING | Hết bộ lọc; stream đó không được khử nhiễu. Ghi 3 lần đầu, sau đó mỗi 500 lần. |
| `asr_result call=... leg=... utterance=... revision=... event=... language=... han_chars=N model_output=... raw=... text=...` | INFO | Chỉ có khi `asr.log_results: true`. Mỗi PARTIAL và COMMIT server tạo ra một dòng, ghi trước bước hiện từng từ. Câu trả lời của model không đổi text thì không tạo kết quả nên không có dòng này. |

Các trường của dòng `asr_result`:

- `revision`: số thứ tự trong server. Khi bật hiện từng từ, số này khác `revision` mà client nhận.
- `language`: ngôn ngữ server đang dùng cho câu đó.
- `han_chars`: số chữ Hán trong `raw`.
- `model_output`: câu trả lời gốc của model, còn nguyên marker ngôn ngữ. Trường này rỗng khi model lỗi, hoặc khi dùng `asr.backend: qwen_asr`.
- `raw`, `text`: text trước và sau punctuation.

Tìm các kết quả có chữ Hán, gồm cả các file log đã xoay vòng:

```bash
grep -h 'asr_result' logs/asr-server.log* | grep -v 'han_chars=0 '
```

**Kiểm tra ép ngôn ngữ** sau khi bật `asr.force_language: true` và khởi động lại (mục 7). `L` in log từ lần khởi động gần nhất. Nếu server khởi động từ trước nửa đêm, dòng `loading` đã sang file `logs/asr-server.log.<ngày>` nên `L` in cả file hôm nay.

```bash
cd /u01/ims/audio-engine
grep -nE 'force_language|default_language|log_results' config/server.yaml
N=$(grep -n 'loading Qwen3-ASR' logs/asr-server.log | tail -1 | cut -d: -f1)
L() { tail -n +"${N:-1}" logs/asr-server.log; }

# 1. Khởi động
L | grep -E 'loading Qwen3-ASR|asr.force_language on|listening on'
L | grep -E 'ERROR|Traceback' | head -20

# 2. Prompt vLLM đã dựng (có sau cuộc gọi đầu tiên)
L | grep 'model prompt language='

# 3. Ngôn ngữ của các stream, và các lần model tự đoán
L | grep 'start call=' | grep -o 'language=[A-Za-z]*' | sort | uniq -c
L | grep -o "model reported language '[^']*'" | sort | uniq -c

# 4. Kết quả nhận dạng (cần asr.log_results: true)
L | grep 'asr_result' | grep -c "model_output='language "
L | grep 'asr_result' | grep -o ' language=[^ ]*' | sort | uniq -c
L | grep 'asr_result' | grep -v 'han_chars=0 ' | tail -20
L | grep 'asr_result' | grep 'event=COMMIT' | tail -30

# Theo dõi trực tiếp khi gọi thử (Ctrl+C để dừng)
tail -f logs/asr-server.log | grep --line-buffered -E 'start call=|model prompt|model reported language|asr_result|ERROR'
```

| Bước | Kết quả đúng | Nếu khác |
|---|---|---|
| 1 | Có `asr.force_language on: ... default vi` (giá trị của `asr.default_language`), rồi `listening on ...`. Không có `ERROR`, `Traceback`. | Server không khởi động: đọc `logs/asr-server.out`. Thiếu chat template thì có `no chat template in ...`. |
| 2 | Dòng `Vietnamese` kết thúc bằng `<\|im_start\|>assistant\nlanguage Vietnamese<asr_text>'`. | Đã có cuộc gọi mà không có dòng nào: vLLM không trả về prompt. Dùng bước 4 để kiểm tra. |
| 3 | Các stream có `language=Vietnamese`. Lệnh thứ hai không in gì. | Còn `model reported language`: xem stream đó có phải `auto` không (dòng `start` ghi `language=auto`). |
| 4 | Lệnh đầu in `0`. Chỉ có `language=Vietnamese`. Lệnh `han_chars` không in gì. | `model_output` còn bắt đầu bằng `language `: model vẫn tự chọn ngôn ngữ, ép không có tác dụng. Đặt lại `force_language: false`, khởi động lại, giữ log bước 2 và 4. |

Dòng `asr_result` chứa nội dung cuộc gọi: che bớt trước khi gửi ra ngoài.

### 12.2. Sự cố

| Hiện tượng | Nguyên nhân | Xử lý |
|---|---|---|
| `update.sh`: `the configuration in ... is not valid for this version` | Cấu hình trong APP có khóa lạ hoặc giá trị sai với mã mới | Sửa theo thông báo, so với `config/server.yaml.dist.new`, chạy lại `update.sh`. |
| `update.sh`: `asr_server is running in ...` | Server đang chạy | Thêm `--restart`, hoặc `bin/stop.sh` trước. |
| `update.sh`: `Permission denied` | User không có quyền ghi vào APP | Chạy bằng user chạy app. |
| `update.sh`: `the installed version cannot read ...` | Đã thêm khóa của phiên bản mới trước khi nâng cấp | Bỏ khóa đó, chạy lại `update.sh --restart`, rồi thêm lại khóa và khởi động lại. |
| `update.sh`: `did not become ready; it was NOT rolled back` | Server chết hoặc chưa mở cổng sau `ASR_START_WAIT_SEC` giây | Xem `logs/asr-server.log` và `logs/asr-server.out`. Sửa lỗi rồi nâng cấp lại, hoặc chạy `bin/rollback.sh`. |
| `rollback.sh`: `the saved configuration of ... is not valid` | Cấu hình trong bản lưu không hợp lệ với mã của bản đó | Chọn bản khác trong `bin/rollback.sh --list`. |
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
