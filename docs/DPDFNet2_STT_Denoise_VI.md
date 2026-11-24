# THIẾT KẾ: DPDFNet-2 LÀM TIỀN XỬ LÝ CHO `speech_to_text` (`asr_denoise`)

**Phiên bản:** 1.0, ngày 24/11/2026. Nhánh `feature/stt-denoise` của AI Engine, tạo từ `main` `c9e752c`.

**Liên quan:**
- `DPDFNet2_Task_Routing_Design_VI.md`: task `noise_reduction`, hợp đồng codec, field 14. Bản 1.4 nằm ở nhánh `docs/noise-reduction-wideband-only`, mục 3.3 nói về G.711.
- `asr_server/README.md`: mục `asr_denoise`.
- `asr_server/HUONG_DAN_CAI_DAT.md`: mục 7.2 (cấu hình) và 10.3 (đo WER).

## 1. Mục tiêu và phạm vi

Đưa DPDFNet-2 vào làm bước **tùy chọn** trước VAD và Qwen3-ASR của `speech_to_text`, để nhận dạng tốt hơn khi có nhiễu.

Giữ nguyên:
- cơ chế commit, segmentation, partial/final, punctuation;
- giao thức protobuf;
- task `noise_reduction` và các task khác.

Ưu tiên cao nhất: pipeline ASR ổn định và **không làm mất lời nói**.

## 2. Kiến trúc

```text
Gateway ──AudioChunk (PCM16 16 kHz)──▶ SpeechStreamService (speech_service.py)
                                         │ kiểm hợp đồng trên chunk gốc (_open/_check/_accept)
                                         ▼ StreamRequest(config/audio/end_stream)
                         ┌── asr_denoise bật và stream thuộc diện áp dụng ──┐
                         │ SttDenoiser (stt_denoise.py)                     │
                         │   hàng đợi → DPDFNet2Processor(align=True)       │
                         │   trên thread pool riêng                         │
                         └──────────────────────────────────────────────────┘
                                         ▼ StreamRequest (audio đã khử nhiễu, sequence đánh lại)
                         ASRService (server.py): VAD → Qwen3-ASR → punctuation — KHÔNG SỬA
```

- `denoise.DPDFNet2Processor` là đường model dùng chung cho `noise_reduction` (`align=False`, output trùng từng byte với trước) và cho STT (`align=True`). Không có implementation DPDFNet-2 thứ hai.
- Không gọi RPC `noise_reduction` từ bên trong STT.
- Lõi ASR (`ASRService`) không sửa dòng nào. Tầng khử nhiễu chỉ thay audio trong các `StreamRequest`.

## 3. Quyết định đã chốt (người dùng duyệt ngày 24/11/2026)

| # | Quyết định | Lý do |
|---|---|---|
| 1 | `apply_to: wideband` là mặc định: chỉ khử nhiễu stream có chunk đầu mang `output_audio_format = PCM_S16LE` và `sample_rate = 16000`, tức cuộc gọi Gateway chạy G.722/L16. `all` dùng để đo thử. Stream 8 kHz không bao giờ khử nhiễu. | Trên audio G.711 băng hẹp, DPDFNet-2 làm nhỏ hẳn nhiều đoạn có tiếng (đo trên cuộc gọi G.711 thật). Ngoài field 14, proto không có trường nào phân biệt G.711 với G.722, vì hai codec cùng clock 8000. |
| 2 | Hết bộ lọc, model lỗi hoặc quá tải thì chuyển sang **audio gốc từ mẫu kế tiếp**; không kết thúc stream. | Gateway đóng session khi stream STT kết thúc. Khử nhiễu chỉ là tùy chọn, ASR phải chạy tiếp. |
| 3 | Mục cấu hình riêng `asr_denoise:`. | Loader cấu hình chỉ hỗ trợ hai cấp. |
| 4 | Pool bộ lọc riêng, không chung với `noise_reduction`. | Hai task không giành bộ lọc của nhau. |
| 5 | Nhánh `feature/stt-denoise` tạo từ `main`. | — |

## 4. Timeline và độ trễ

Đo trên model thật (`dpdfnet2.onnx`, sherpa-onnx 2.0.2, trên tiếng nói thật, chunk 20/40/80 ms):
- `run()` trả theo bước 160 mẫu và giữ lại một bước; `flush()` trả nốt phần còn lại. Tổng mẫu ra bằng tổng mẫu vào.
- Nội dung output trễ đúng **640 mẫu (40 ms)** so với input, đo bằng tương quan chéo và bằng xung đơn.

Với `align=True`:
- bỏ 640 mẫu đầu model trả ra;
- lúc kết thúc stream, đưa thêm 640 mẫu im lặng vào model rồi mới `flush()`.

Kết quả: mẫu ra thứ k là mẫu vào thứ k đã khử nhiễu, số mẫu bằng nhau, tương quan chéo cho lag 0 trên tiếng nói thật. VAD, ASR và `ts_start`/`ts_end` dùng nguyên timeline của input. `check_model_delay()` chạy lúc khởi động: model có độ trễ khác 640 thì bị từ chối.

Độ trễ thật của kết quả PARTIAL/FINAL vẫn tăng khoảng 40 ms cộng thời gian chạy model (15–25 ms mỗi chunk 80 ms trên laptop).

## 5. Streaming, trạng thái và kết thúc stream

- Mỗi stream mượn một bộ lọc từ pool nạp sẵn (`max_streams`) và trả lại khi stream kết thúc hoặc bị hủy. Nếu một lần gọi model còn đang chạy, bộ lọc được trả sau khi lần gọi đó xong.
- State của model giữ suốt stream, qua cả khoảng lặng và các lần commit; **không** reset theo `is_final` của ASR.
- Một tác vụ đọc riêng kéo request ngay khi tới, nên đo được lượng audio đang chờ model (`backlog`).
- Model chạy trên thread pool riêng (`workers`); mỗi stream gọi model lần lượt, theo thứ tự.
- Khi nhận `end_stream` hoặc client half-close: xử lý hết input, `flush()` đúng một lần, chuyển toàn bộ audio còn lại cho lõi, **rồi mới** gửi `end_stream`. Lõi trả FINAL theo cơ chế sẵn có.

## 6. Chính sách lỗi

Không bao giờ bỏ chunk. Tầng STT giữ riêng một bản input "đã đưa vào model nhưng model chưa trả" (`raw`). Khi chuyển sang audio gốc, phần này được trả trước, sau đó đến các chunk mới nguyên vẹn, nên không mất mẫu và không lặp mẫu. Bản sao phải nằm ở tầng STT chứ không trong processor, vì một lần gọi model chậm có thể vẫn đang dùng processor.

| Tình huống | Hành vi | Log / bộ đếm |
|---|---|---|
| Không còn bộ lọc | Stream nhận dạng từ audio gốc | WARNING `no denoiser free` (3 lần đầu, sau đó mỗi 500 lần), `stats['no_denoiser']` |
| Model lỗi (kể cả lúc `flush`) | Chuyển sang audio gốc từ mẫu kế tiếp; bộ lọc bị loại và nạp bộ mới | WARNING `reason=model_error`, `fallback_error` |
| Một lần gọi model > `max_backlog_ms` | Như trên | `reason=model_slow`, `fallback_slow` |
| Lần gọi chưa bắt đầu sau `max_backlog_ms` vì mọi `workers` đều bận | Như trên; huỷ lần gọi đang xếp hàng | `reason=busy`, `fallback_busy` |
| Audio chờ model > `max_backlog_ms` | Như trên | `reason=backlog`, `fallback_backlog` |
| Lỗi giao thức (`INVALID_ARGUMENT`…) | Giữ nguyên mã lỗi như khi tắt khử nhiễu; trả bộ lọc | — |
| Client hủy, mất kết nối | Trả bộ lọc (sau khi lần gọi đang chạy xong) | `outcome=cancelled` |
| Stream bị từ chối giữa chừng (lỗi giao thức) | Giữ mã lỗi; trả bộ lọc | `outcome=error` |

Sau mỗi lần chuyển sang audio gốc, bộ lọc được trả về pool ngay, hoặc ngay khi lần gọi đang chạy xong, để cuộc gọi khác dùng được.

Mỗi stream ghi một dòng INFO `stt_denoise stream ended`, gồm `outcome`, `reason`, `samples_in`, `samples_out`, `model_ms` và `backlog_ms_max`.

`max_backlog_ms: 0` tắt mọi giới hạn trên, chỉ dùng cho server thử gửi nhanh hơn thời gian thực. Khi đó một lần gọi model bị kẹt sẽ giữ cả stream, và audio chờ không có giới hạn; bộ đếm `server.ingress_max_ms` của lõi không thấy phần audio này.

## 7. Cấu hình

```yaml
asr_denoise:
  enabled: false             # mặc định: hành vi cũ
  model: /opt/models/dpdfnet2.onnx
  apply_to: wideband         # wideband | all
  num_threads: 1
  workers: 2
  max_streams: 16            # ≤ server.max_streams
  max_backlog_ms: 500        # 0..5000, 0 = không giới hạn
```

Ràng buộc:
- `enabled: true` cần `asr.enabled: true`.
- Thiếu model hoặc `sherpa-onnx`: server không khởi động, và `scripts/update.sh` báo `MISSING ... (asr_denoise...)`.
- Proto không đổi.

## 8. Kiểm thử

Kiểm thử đơn vị và gRPC dùng model giả có đúng đặc tính thời gian của DPDFNet-2:
- **`tests/test_denoise_processor.py`:**
  - căn trễ với chunk 20/40/80 ms, chunk lẻ và stream ngắn hơn độ trễ;
  - `flush` không mất đuôi;
  - giữ state qua các chunk; các stream độc lập;
  - model trả thiếu mẫu ở cuối thì được bù bằng input;
  - `check_model_delay`.
- **`tests/test_stt_denoise.py`:**
  - **Test then chốt:** với model "identity" (gain 1, trễ 640), kết quả STT khi bật **trùng hệt** khi tắt (text, `ts`, `utterance_id`, `revision`, `reason`, `seq`). Kiểm cho G.722 và L16, chunk 20/40/80 ms, có và không có `end_of_stream`.
  - VAD/ASR nhận đúng audio đã khử nhiễu.
  - G.711 và 8 kHz không bị đụng tới; hết bộ lọc.
  - Lỗi model ở giữa stream, ở chunk đầu và lúc kết thúc; model chậm; worker bận; tồn đọng.
  - Lỗi giao thức giữ nguyên mã; hủy qua gRPC thật trả lại bộ lọc.
  - Phần nối `serve()`; từ chối model có độ trễ khác 640.
- **Mutation:** 6 mutation đều bị bắt (bỏ bù trễ, bỏ đệm im lặng, bỏ phần input giữ lại khi fallback, bỏ flush lúc kết thúc, bỏ điều kiện wideband, không trả bộ lọc).
- **Hồi quy:** toàn bộ test `noise_reduction`, STT và gRPC cũ qua. Output `noise_reduction` trên model thật trùng từng byte với trước refactor.

Kiểm thử với model thật (`tests/test_noise_real_model.py`):
- căn trễ (xung đơn ở mẫu 1000 ra đúng mẫu 1000);
- tầng STT với model thật: số mẫu ra bằng vào, `sequence` liên tục, `end_stream` ở cuối.

## 9. Số đo trên laptop (12 luồng CPU)

**Thông lượng:**

| Stream / workers | Thông lượng | Thời gian model mỗi chunk 80 ms (p50 / p95) |
|---|---|---|
| 1 / 1 | 4,1× thời gian thực | 19 / 24 ms |
| 4 / 4 | 9,4× | 31 / 47 ms |
| 8 / 4 | 10,1× | 28 / 46 ms |
| 16 / 8 | 10,2× | 45 / 105 ms |

**VAD** (webrtcvad mode 2, khung 20 ms), 6 mẫu VoiceBank, lấy VAD trên bản sạch làm chuẩn:

| | Bỏ sót tiếng nói | Báo nhầm |
|---|---|---|
| Audio gốc có nhiễu | 7,6% | 31,2% |
| Sau khử nhiễu | 6,2% | 4,8% |

Ở 2/6 mẫu, khử nhiễu làm VAD bỏ sót nhiều hơn: 0,6% lên 8,3% và 5,6% lên 9,3%.

## 10. Giới hạn và việc còn mở

1. **WER/CER chưa đo.** Cần GPU và Qwen3-ASR; quy trình ở HUONG_DAN mục 10.3, dùng `apply_to: all` vì client đo không gửi field 14. **Không bật ở production khi chưa có số WER bật/tắt.**
2. **Silero VAD chưa đo** (máy dev không có checkpoint). Số ở mục 9 là của webrtcvad.
3. **Năng lực CPU của server chưa đo.** Chọn `max_streams`/`workers` sau khi trừ phần CPU ASR dùng.
4. **Phụ thuộc Gateway gửi field 14 cho STT.** Gateway hiện tại gửi trên mọi task; Gateway cũ không gửi thì không stream nào được khử nhiễu (an toàn).
5. **Một lần treo khi chạy test.** Đã thấy một lần khi chạy `test_stt_denoise.py` và `test_serve_noise.py` chung một tiến trình; không tái hiện được qua 5 lượt chạy sau. `run_tests.sh` chạy hai file này tách tiến trình.
6. **Khi đã chuyển sang audio gốc,** stream không quay lại khử nhiễu trong phần còn lại của stream.
7. **Nhãn `busy` có thể sai trong một khe thời gian rất hẹp.** Nếu một luồng vừa rảnh đúng lúc hết hạn chờ thì lần gọi vẫn chạy, nhưng được ghi `busy` thay vì `model_slow`. Hành vi (chuyển sang audio gốc) không đổi. Chưa tái hiện được nên chưa sửa.
