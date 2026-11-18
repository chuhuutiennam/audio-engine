# GIẢI PHÁP SỬA LỖI SERVER HIỆN CÓ
## `qwen3_asr_grpc_vad_commit` (Qwen3-ASR + WebRTC VAD + punctuation, `asr.proto`)

> **Phiên bản:** 1.6 (đã hiện thực; kết quả ở mục 11; backend native vLLM và Silero ở mục 12; chuyển sang `speech.proto` ở mục 13; sửa sau khi review luồng nhận `AudioChunk` ở mục 14; đối chiếu với mã nguồn rtpgw ở mục 15)  
> **Ngày:** 16/11/2026  
> **Trạng thái:** đã sửa B4, B5, B2, B7, B1, B3, B8 và đưa `strip_asr_markers` vào server. Mới kiểm tra bằng model giả; **chưa chạy trên GPU với Qwen thật**.  
> **Mã nguồn:** `audio-engine/asr_server/`. Số dòng ghi trong tài liệu là của `server.py` trong zip gốc; ở `src/asr_server/server.py` các dòng này lệch lên 1.  
> **Phạm vi:** sửa trên server hiện có. Các mục 1 đến 12 viết theo `asr.proto`; từ bản 1.4 server chỉ còn phục vụ `speech.proto`, bảng đổi tên trường ở mục 13.3.  
> **Bằng chứng:** `docs/test_repro_bugs_server_hien_tai.py`, 11 test tái hiện trên code chưa sửa, dùng ASR giả, không cần GPU.

---

## 1. Các quyết định đã chốt

| # | Câu hỏi | Trả lời |
|---|---|---|
| 1 | Sửa trên server hiện có, giữ `asr.proto`? | Đúng. |
| 2 | Cách sửa B2 | Một worker thread cho mọi lời gọi SDK. |
| 3 | Gateway khi im lặng | Gửi chunk liên tục, `sequence` không nhảy số. |
| 4 | Gateway chấp nhận COMMIT có `text` rỗng và `reason` mới? | Có. |
| 5 | Hàng đợi audio đầy | Đóng stream bằng `RESOURCE_EXHAUSTED`. |
| 6 | PARTIAL có cần dấu câu? | Có. |
| 7 | Client có được thấy COMMIT rỗng và `reason` mới? | Không. Các trường hợp bất thường chỉ ghi log; PARTIAL đã hiển thị được xóa bằng một PARTIAL có `text` rỗng. |
| 8 | Mức log khi model lỗi | `ERROR`. Các trường hợp `discard`, text rỗng ở `DEBUG`; hết hạn chờ audio ở `WARNING`. |
| 9 | Mục tiêu tải | 20 cuộc gọi đồng thời, nhận dạng một chiều, tức 20 stream; ngưỡng cần đạt là `processing_rtf` ≤ 0,035. Có cần thêm biện pháp tăng tải hay không sẽ quyết sau phép đo ở mục 7. |
| 11 | Đưa `strip_asr_markers` vào server? | Có. |
| 13 | Server chạy thật | Bản native vLLM + Silero (`qwen3_asr_grpc_native_vllm_silero.zip`), vLLM 0.16.0 trở lên. Model tự nhận diện ngôn ngữ; server lấy ngôn ngữ từ marker. |
| 12 | Ngôn ngữ và punctuation | Truyền ngôn ngữ của utterance vào bước punctuation, để sau này chọn model hoặc ngưỡng theo ngôn ngữ. |
| 14 | Giao thức gRPC | Thay hẳn `asr.proto` bằng `speech.proto` (`SpeechStream.Recognize`). |
| 15 | Mở rộng `RecognitionResult` | Thêm tag 12 đến 16, vì thiếu `raw_text` thì client không đo được WER trước dấu câu. |
| 16 | Model báo ngôn ngữ ngoài `vi`/`en` | Ghi log và xử lý tiếp. |
| 17 | `ts_start`/`ts_end` | RTP timestamp theo clock nguồn, lấy modulo 2^32 như RTP gốc. |
| 18 | Gateway | Luôn gửi `chunk_seq` từ 0 và `rtp_clock_rate = 8000`. |
| 19 | Model chậm hơn audio | Bỏ lần nhận dạng PARTIAL khi stream đang tồn audio; FINAL luôn nhận dạng; giới hạn 3000 ms giữ làm chốt chặn cuối. |
| 20 | `rtp_timestamp` nhảy giữa stream | Lấy lại mốc RTP ở mỗi lần nhảy; `ts_start`/`ts_end` theo chunk chứa audio đó. |
| 21 | Tag mở rộng của `RecognitionResult` | Dời từ 12–16 sang 20–24; tag 12–14 là `topic_label`, `topic_info`, `topic_score` của rtpgw. **Thay quyết định 15 về số tag.** |
| 22 | PARTIAL rỗng (tín hiệu xóa) | Chấp nhận MF không nhận được: rtpgw bỏ kết quả không có text và không có topic. Không sửa rtpgw. |
| 23 | `chunk_seq` nhảy | Ghi `WARNING` và xử lý tiếp. **Thay quyết định 3 (B6 chặt).** |
| 24 | Task khác `speech_to_text` | Mọi task đều tới audio-engine; vẫn từ chối `realtime_translation`, `keyword`, `voice_controller_emoji`. |
| 10 | Nơi đặt mã nguồn | Giải nén zip thành thư mục mới trong `audio-engine`, tổ chức lại thành project Python chuẩn có `config`, `logs`, `bin`, `data`, `scripts`. |

## 2. Tổng hợp

| ID | Lỗi | Vị trí | Quyết định |
|---|---|---|---|
| B2 | Hủy RPC giữa lúc suy luận cho phép hai lời gọi model chạy đồng thời | `server.py` dòng 107–108, 132–137 | Sửa |
| B7 | Ngoại lệ từ model làm rơi cả RPC | `server.py` `feed`, `finish` | Sửa |
| B4 | Sau forced split, mọi đoạn mang `start_ms` của utterance gốc | `server.py` dòng 157–161, `vad.py` dòng 88 | Sửa |
| B5 | PARTIAL không có COMMIT đi kèm | `server.py` dòng 140–142, 177–180 | Sửa |
| B1 | Vòng đọc gRPC chờ model | `server.py` dòng 183–234 | Sửa |
| B8 | Punctuation chạy tuần tự dưới khóa toàn cục, nằm trong đường đọc | `server.py` dòng 47–51, 119 | Sửa |
| B3 | Ngừng nhận audio thì không bao giờ chốt | `vad.py` dòng 67 | Thêm timer an toàn |
| B6 | Lệch `sequence` là hủy stream | `server.py` dòng 211–213 | **Không sửa**: Gateway không nhảy số, giữ `strict` |

Thứ tự trong bảng là thứ tự ưu tiên.

## 3. Chi tiết từng lỗi

### B2. Hai lời gọi model đồng thời sau khi hủy RPC

**Tái hiện:** `test_B2_cancel_during_inference_allows_concurrent_model_calls` đo được 2 lời gọi model cùng lúc.

**Nguyên nhân:** `async with self.infer_lock: await asyncio.to_thread(...)`. Khi client hủy, task nhận `CancelledError` và nhả khóa, trong khi thread vẫn đang chạy model. Stream khác lấy được khóa và gọi model song song.

**Giải pháp:**

- Thay `infer_lock` và `asyncio.to_thread` bằng một `ThreadPoolExecutor(max_workers=1)` dùng chung. Mọi lời gọi SDK (`streaming_transcribe`, `finish_streaming_transcribe`) đều chạy trên thread này.
- Server tự gom audio 16 kHz của utterance; chỉ khi đủ ít nhất một chunk (`chunk_sec`) mới gửi một job cho worker. Nhờ đó không còn lời gọi SDK nào chạy trên event loop, và server không phải đọc `asr_state.buffer`, `chunk_size_samples`, `chunk_id`.
- Khi chốt utterance: phần audio còn lại và `finish_streaming_transcribe` chạy trong **một** job.
- Mỗi job kiểm tra cờ `alive` của stream khi bắt đầu; stream đã hủy thì job bị bỏ qua.

```python
class InferWorker:
    def __init__(self):
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='asr-infer')

    async def run(self, fn, *args, alive=lambda: True):
        def job():
            if not alive():
                return False
            fn(*args)
            return True
        return await asyncio.get_running_loop().run_in_executor(self._pool, job)
```

**Vì sao không lẫn kết quả giữa các stream:** mỗi utterance có `asr_state` riêng và truyền chính object đó vào job; mỗi job trả về future riêng mà chỉ stream gửi job mới chờ; worker chạy từng job một. Mô phỏng 16 stream, 5 stream bị hủy, 143 lời gọi: 0 lần lẫn, tối đa 1 lời gọi cùng lúc.

**Nghiệm thu:** test B2 đảo chiều (tối đa 1 lời gọi đồng thời); thêm test nhiều stream nội dung khác nhau, mỗi stream nhận đúng text của mình.

### B7. Ngoại lệ từ model làm rơi cả RPC

**Tái hiện:** `test_B7_model_exception_kills_stream`: RPC kết thúc `UNKNOWN` hoặc `INTERNAL`, không COMMIT nào tới client.

**Giải pháp:**

- Bắt ngoại lệ quanh mọi job của worker.
- Khi lỗi: **không phát COMMIT**. Ghi log mức `ERROR` kèm `call_id`, `leg_id`, `utterance_index` và ngoại lệ; bỏ state của utterance.
- Nếu utterance đó đã phát PARTIAL thì phát một PARTIAL có `text = ""` và `raw_text = ""` để client xóa phần đang hiển thị.
- Bỏ qua audio cho tới event `endpoint` hoặc `discard` kế tiếp của VAD, rồi nhận utterance mới bình thường.
- Đếm lỗi liên tiếp; một lần suy luận thành công thì về 0. Đạt `asr.max_consecutive_errors` (mặc định 3) thì đóng stream bằng `UNAVAILABLE`.

**Nghiệm thu:** test B7 đảo chiều: stream sống tiếp, không có COMMIT cho utterance lỗi, utterance sau đó vẫn có COMMIT bình thường; log có một dòng `ERROR`; test 3 lỗi liên tiếp cho `UNAVAILABLE`.

### B4. `start_ms` sai sau forced split

**Tái hiện:** `test_B4_forced_split_reuses_original_start_ms`: 3 giây nói liên tục với giới hạn 1 giây cho các COMMIT `(0–1000)`, `(0–2000)`, `(0–3000)`, `(0–3600)`.

**Nguyên nhân:** sau COMMIT `max_duration`, VAD vẫn đang trong utterance và event `audio` tiếp tục mang `start_ms` cũ; `new_state(ev.start_ms)` nhận lại giá trị đó.

**Giải pháp:** thêm vào `WebRTCEndpointer`:

```python
def rebase_start(self, start_ms):
    """Forced split: the continuing audio belongs to a new utterance starting here."""
    self.start_ms = int(start_ms)
```

`process_event` gọi `vad.rebase_start(ev.end_ms)` ngay sau COMMIT `max_duration`. Bộ đếm `voiced_frames` giữ nguyên để phần đuôi ngắn sau điểm cắt không bị `discard` nhầm.

**Nghiệm thu:** test B4 đảo chiều: các đoạn nối tiếp nhau `(0–1000)`, `(1000–2000)`, ..., không chồng lấn.

### B5. PARTIAL không có COMMIT đi kèm

**Tái hiện:** `test_B5a` (cấu hình mặc định, hai tiếng động ngắn cách nhau 0,5 giây: có PARTIAL rồi `discard`), `test_B5b` (text cuối rỗng), `test_B5c` (utterance kế tiếp dùng lại `utterance_index`, `revision` lặp giá trị 1).

**Giải pháp:**

- **Sửa tận gốc trường hợp `discard`:** không phát PARTIAL khi tổng voiced của utterance chưa đạt `min_voiced_ms`. VAD chỉ `discard` utterance có voiced dưới ngưỡng này, nên utterance đã có PARTIAL thì không thể bị `discard`.
- Thêm cờ `partial_sent` cho mỗi utterance.
- Text cuối rỗng và `partial_sent`: không phát COMMIT; phát một PARTIAL có `text = ""` để xóa; ghi log mức `DEBUG`. Chưa phát PARTIAL thì không phát gì, như hiện nay.
- `utterance_index` chỉ tăng sau COMMIT, như hiện nay. Khi một index được dùng lại cho utterance kế tiếp, **`revision` tiếp tục tăng, không quay về 1**, để client không coi PARTIAL mới là bản cũ.

**Nghiệm thu:** ba test B5 đảo chiều: không còn PARTIAL nào trước khi đủ `min_voiced_ms`; sau PARTIAL có text luôn là một COMMIT hoặc một PARTIAL rỗng; `revision` tăng đơn điệu trong cùng `utterance_index`.

### B1. Vòng đọc gRPC chờ model

**Tái hiện:** `test_B1_reader_is_blocked_while_model_runs` (giữa hai lần đọc chunk có khoảng dừng bằng thời gian suy luận), `test_B1_streams_are_serialized_by_global_lock` (4 stream song song chậm gấp 4 lần 1 stream).

**Giải pháp:**

- Mỗi stream có một task đọc riêng: chỉ lấy message từ `request_iterator` và đẩy vào hàng đợi.
- Hàng đợi giới hạn theo **thời lượng audio** đang chờ, `server.ingress_max_ms` (mặc định 3000). Vượt giới hạn thì đóng stream bằng `RESOURCE_EXHAUSTED`.
- Vòng xử lý (VAD, ASR, punctuation, phát kết quả) lấy từ hàng đợi; logic bên trong giữ nguyên.
- Khi RPC kết thúc hoặc bị hủy: hủy task đọc, đặt `alive = False`.

**Giới hạn cần nói rõ:** sửa này không tăng năng lực tải. Suy luận vẫn tuần tự trên một worker, nên 4 stream vẫn chậm gấp 4. Cái đạt được là server luôn đọc audio đúng nhịp, và khi quá tải thì báo lỗi rõ ràng thay vì nghẽn ngầm ở tầng gRPC.

**Nghiệm thu:** test đọc không còn khoảng dừng theo thời gian suy luận; test hàng đợi đầy cho `RESOURCE_EXHAUSTED`. Test "4 stream chậm gấp 4" được giữ làm test mô tả hành vi, không đảo chiều.

### B8. Punctuation tuần tự trong đường đọc

**Tái hiện:** `test_B8_partial_punctuation_serialized_and_in_read_path`: 4 stream, 8 lần gọi, tổng thời gian bằng tổng các lần gọi.

**Giải pháp:**

- PARTIAL vẫn có dấu câu.
- Thay `punct_lock` và `to_thread` bằng executor riêng, `punctuation.workers` (mặc định 1).
- PARTIAL: chờ tối đa `punctuation.partial_timeout_ms` (mặc định 300); quá hạn thì phát text thô. Khi đó `last_mark` rỗng nên `semantic_pause` không kích hoạt.
- COMMIT: chờ tối đa `punctuation.commit_timeout_ms` (mặc định 1000); quá hạn thì `text = raw_text`.
- Mỗi stream có tối đa một job punctuation cho PARTIAL đang chờ; nếu job trước chưa xong thì PARTIAL mới dùng text thô, để hàng đợi không phình khi quá tải.

Sau B1, punctuation không còn chặn việc đọc audio; phần này chỉ còn chặn độ trễ của kết quả.

**Nghiệm thu:** punctuation chậm hơn hạn mức thì PARTIAL và COMMIT vẫn ra đúng hạn với text thô; `raw_text` luôn có.

### B3. Ngừng nhận audio thì không chốt

**Tái hiện:** `test_B3_no_commit_while_audio_stops_arriving`: nói 1 giây rồi ngừng gửi 2 giây, COMMIT chỉ tới sau 2 giây với `reason = end_stream`.

**Đánh giá:** Gateway gửi liên tục nên tình huống này không xảy ra khi vận hành bình thường. Chỉ thêm lưới an toàn cho trường hợp Gateway treo.

**Giải pháp:** vòng xử lý chờ hàng đợi có hạn `server.audio_timeout_ms` (mặc định 2000) khi đang có utterance mở. Quá hạn thì gọi `vad.finish()` và chốt bằng COMMIT với `reason = "vad_endpoint"` sẵn có; nguyên nhân thật (`audio_timeout`) ghi log mức `WARNING`. Stream vẫn mở.

**Nghiệm thu:** test B3 đảo chiều: COMMIT tới trong khoảng 2 giây sau audio cuối, `reason = vad_endpoint`, log có dòng `audio_timeout`.

### B6. Lệch `sequence` là hủy stream (không sửa)

Hành vi này có chủ ý và có test `test_reject_nonmonotonic_sequence`. Gateway không nhảy `sequence`, nên giữ nguyên.

Ghi nhận khi tái hiện: nếu client còn đang gửi, nó có thể nhận `INTERNAL` thay vì `INVALID_ARGUMENT`. Gateway không nên dựa vào mã lỗi cụ thể để phân biệt nguyên nhân.

## 4. Thay đổi nhìn từ phía client

`asr.proto` không đổi, không có giá trị `reason` mới và không có COMMIT rỗng. Các thay đổi hành vi còn lại:

| Thay đổi | Trước | Sau |
|---|---|---|
| PARTIAL có `text` rỗng | Không bao giờ | Có, để xóa PARTIAL đang hiển thị khi utterance kết thúc mà không có COMMIT (text cuối rỗng, model lỗi) |
| PARTIAL đầu tiên của utterance | Có thể phát khi voiced còn dưới `min_voiced_ms` | Chỉ phát khi voiced đã đạt `min_voiced_ms` |
| `revision` khi `utterance_index` được dùng lại | Quay về 1 | Tiếp tục tăng |
| `start_ms` sau forced split | Lặp giá trị gốc | Bằng `end_ms` của đoạn trước |
| Model lỗi | RPC rơi | Stream tiếp tục, utterance lỗi không có COMMIT; 3 lỗi liên tiếp mới `UNAVAILABLE` |
| Quá tải | Nghẽn ngầm, độ trễ tăng dần | `RESOURCE_EXHAUSTED` khi audio chờ vượt `server.ingress_max_ms` |
| PARTIAL khi punctuation chậm | Trễ | Ra đúng hạn, có thể không có dấu câu |

Eval client `eval_wav_grpc.py` chạy được không cần sửa: PARTIAL rỗng chỉ ghi đè text tạm của utterance.

Log thay cho thông báo tới client:

| Sự kiện | Mức log |
|---|---|
| Model lỗi trong một utterance | `ERROR` |
| Hết hạn chờ audio (`audio_timeout`) | `WARNING` |
| Utterance bị `discard`; text cuối rỗng | `DEBUG` |

## 5. Tham số cấu hình mới (`config/server.yaml`)

| Khóa | Mặc định | Dùng cho |
|---|---:|---|
| `server.ingress_max_ms` | 3000 | B1 |
| `server.audio_timeout_ms` | 2000 | B3 |
| `asr.max_consecutive_errors` | 3 | B7 |
| `punctuation.workers` | 1 | B8 |
| `punctuation.partial_timeout_ms` | 300 | B8 |
| `punctuation.commit_timeout_ms` | 1000 | B8 |

Các giá trị mặc định là đề xuất, chưa đo trên hệ thống thật.

## 6. Kế hoạch thực hiện

| Bước | Nội dung | File |
|---|---|---|
| 0 | Dựng project Python chuẩn từ zip, không đổi hành vi (**đã xong**) | `asr_server/` |
| 1 | B4, B5 | `src/asr_server/vad.py` (thêm `rebase_start`), `src/asr_server/server.py` (`finish`, `process_event`) |
| 2 | B2, B7 | `src/asr_server/server.py` (`InferWorker`, `feed`, `finish`) |
| 3 | B1, B3 | `src/asr_server/server.py` (`StreamRecognize`: task đọc, hàng đợi, timer) |
| 4 | B8 | `src/asr_server/server.py` (`punctuate`) |
| 5 | Cấu hình và tài liệu | `src/asr_server/config.py`, `config/server.yaml` (khóa mới ở mục 5), `README.md` |

Sau mỗi bước: toàn bộ test trong `tests/` phải pass (hiện là 32 test, gồm 13 test gốc và 19 test cấu hình), và các test tái hiện tương ứng được đảo chiều thành test hồi quy trong `tests/`.

## 7. Năng lực tải (mục tiêu 20 cuộc gọi đồng thời)

Suy luận chạy tuần tự trên một worker vì API streaming của SDK không gom batch được. B1 không thay đổi điều này. Chưa có số đo thời gian suy luận trên GPU thật, nên chưa biết 20 cuộc gọi có vượt năng lực hay không.

**Phép đo, không cần code mới:** trên server dùng để đo, đặt `server.ingress_max_ms: 0` (client `--fast` gửi nhanh hơn thời gian thực nên sẽ chạm giới hạn hàng đợi và bị đóng bằng `RESOURCE_EXHAUSTED`); rồi chạy eval client ở chế độ `--fast` trên một file ghi âm đại diện cho **một chiều thoại** (có cả khoảng lặng tự nhiên) và đọc `processing_rtf` trong `metrics.json`. Số này là phần thời gian thực mà server cần để xử lý một stream.

```text
số stream tối đa ≈ 0,7 / processing_rtf        (0,7 = mức tải nên giữ để hàng đợi không dài, là đề xuất)
```

| Cần phục vụ | `processing_rtf` phải không lớn hơn |
|---|---:|
| 20 stream (20 cuộc, một chiều) | 0,035 |
| 40 stream (20 cuộc, hai chiều) | 0,0175 |

**Các hướng nếu không đủ:**

| Hướng | Tăng tải thật? | Cần gì | Đánh đổi |
|---|---|---|---|
| A. Worker ưu tiên job chốt utterance trước job PARTIAL | Không; giữ độ trễ COMMIT khi đông | Khoảng 20 dòng trong `InferWorker` | PARTIAL trễ hơn khi quá tải |
| B. `asr.chunk_sec: 2.0` | Giảm khoảng 30–40% lượng suy luận | Tham số có sẵn | PARTIAL thưa gấp đôi |
| C. Nhiều tiến trình server, mỗi tiến trình một GPU | Gần tuyến tính theo số GPU | Thêm GPU, cân bằng tải theo stream | Chi phí phần cứng |
| D. Gom nhiều stream vào một lần suy luận | Có, trên cùng một GPU | Bỏ API streaming, dùng `transcribe` theo lô | Sửa lớn, phải đo lại WER; thuộc bản thiết kế 1.1 |

Hướng A **chưa đưa vào** đợt sửa này; quyết sau khi có `processing_rtf`.

**Lưu ý khi triển khai B1:** hiện nay quá tải chỉ làm kết quả trễ dần. Sau B1, quá tải kéo dài làm stream bị đóng bằng `RESOURCE_EXHAUSTED`. Nếu phép đo cho thấy hệ thống đang chạy sát giới hạn thì phải xử lý tải (hướng B, C hoặc D) trước hoặc cùng lúc với B1, hoặc tạm đặt `server.ingress_max_ms` lớn hơn.

## 8. Không làm trong đợt này

- Đưa cả 600 ms im lặng cuối vào ASR: là tinh chỉnh, cần đo WER trước khi đổi.
- Metrics, graceful drain, giới hạn tải theo năng lực thật (`server.max_streams: 64` hiện không phản ánh năng lực).
- Tăng năng lực tải: xem mục 7.
- Chuyển sang `speech.proto` và timeline RTP.

## 9. Điểm còn mở

- Chọn model hoặc ngưỡng punctuation theo ngôn ngữ: chưa làm. Ngôn ngữ đã được truyền tới bước punctuation (mục 11) nhưng model đa ngữ hiện tại chưa dùng tới.
- Hướng A ở mục 7 (ưu tiên job chốt utterance): chờ số đo `processing_rtf`.

## 10. Giả định chưa kiểm chứng

- Model thật không giữ dữ liệu riêng của stream ngoài `asr_state`. Server hiện tại cũng đang dựa vào giả định này.
- SDK thật suy luận ngay khi nhận đủ một chunk, giống ASR giả trong test. Điều kiện `will_infer` trong code hiện tại cũng dựa vào điều này.
- Chưa biết vLLM thật phản ứng thế nào khi bị gọi đồng thời (B2); test chỉ chứng minh hai lời gọi có xảy ra cùng lúc.
- Mọi test dùng ASR và punctuation giả. Sau khi sửa cần chạy lại trên GPU với eval client và so với kết quả trước khi sửa.

## 11. Kết quả hiện thực

**File thay đổi trong `asr_server/`:** `src/asr_server/server.py`, `src/asr_server/punctuation.py` (thêm tham số `language`), `src/asr_server/vad.py` (thêm `rebase_start`), `src/asr_server/text_clean.py` (mới, chép từ bộ eval), `src/asr_server/config.py`, `config/server.yaml`, `tests/test_config.py`, `tests/test_grpc.py`, `tests/test_regression.py` (mới), `README.md`.

**Kiểm tra:**

| Hạng mục | Kết quả |
|---|---|
| Test server (`tests/`) | 57 pass: 13 test gốc, 22 test cấu hình, 21 test hồi quy, 1 test punctuation mới |
| Test eval client (`tools/eval/tests/`) | 19 pass |
| Bộ test tái hiện lỗi cũ chạy trên server đã sửa | B1 (đọc bị chặn), B2, B4, B5a, B7 không còn tái hiện |
| Chạy thử đầu cuối, model giả, WebRTC VAD thật | `start.sh`, client demo và eval client ở chế độ thời gian thực, `stop.sh` chạy đúng |

Các test tái hiện vẫn pass theo đúng thiết kế: các stream vẫn tuần tự (B1, mục 7); `sequence` vẫn `strict` (B6); utterance có text cuối rỗng vẫn không có COMMIT và `utterance_index` vẫn được dùng lại (B5, nay kèm PARTIAL rỗng và `revision` tăng dần).

**Khác với tài liệu hoặc phát sinh khi làm:**

- **Client gửi nhanh hơn thời gian thực sẽ bị đóng stream.** Đã kiểm chứng: eval client `--fast` trên file 36 giây nhận `RESOURCE_EXHAUSTED` với `server.ingress_max_ms: 3000`. Chế độ thời gian thực không bị ảnh hưởng. Vì lý do này `tests/test_grpc.py` (test gốc) được thêm một dòng đặt `ingress_max_ms=0`.
- Punctuation ném ngoại lệ thì COMMIT dùng `raw_text` (trước đây làm rơi RPC). Tài liệu chỉ nêu trường hợp quá hạn.
- `server.ingress_max_ms: 0` nghĩa là không giới hạn, `server.audio_timeout_ms: 0` nghĩa là tắt timer.
- `init_streaming_state` vẫn được gọi trên event loop như trước; chỉ `streaming_transcribe` và `finish_streaming_transcribe` chạy trên worker.
- `raw_text` nay là text đã loại marker.
- **Ngôn ngữ truyền vào punctuation:** server gọi `punctuator.restore(text, language=...)` cho cả PARTIAL và COMMIT. Giá trị là ngôn ngữ model báo về cho utterance, nếu không có thì lấy ngôn ngữ client gửi; đây cũng là giá trị trả cho client trong `Transcript.language`. `HFPunctuator` nhận tham số này nhưng chưa dùng, nên kết quả chấm câu không đổi. Punctuator nào chưa có tham số `language` vẫn chạy được.

## 12. Backend native vLLM và Silero VAD

`asr_server` ban đầu được dựng từ `qwen3_asr_grpc_vad_commit.zip` (WebRTC VAD, SDK `qwen_asr`). Server chạy thật là `qwen3_asr_grpc_native_vllm_silero.zip`. Lớp `ASRService` của hai bản giống nhau, nên các lỗi ở mục 2 có ở cả hai: bộ test tái hiện chạy trên bản native cho B1, B2, B3, B4, B5, B7, B8 đều tái hiện.

**Đã đưa vào project:**

| Thành phần | File | Ghi chú |
|---|---|---|
| `VLLMASRBackend` | `src/asr_server/asr_backend.py` | Lấy nguyên từ zip native; thêm bước tách marker. |
| `SileroEndpointer` | `src/asr_server/vad.py` | Lấy nguyên từ zip native; thêm `rebase_start` (B4). Nằm cạnh `WebRTCEndpointer`. |
| Chọn backend | `config/server.yaml` | `asr.backend: native_vllm` (mặc định) hoặc `qwen_asr`; `vad.backend: silero` (mặc định) hoặc `webrtc`. |
| Tách marker và ngôn ngữ | `src/asr_server/text_clean.py` (`split_asr_output`) | Text sạch; ngôn ngữ trong marker đầu tiên thành `Transcript.language` và được truyền cho punctuation. Marker ghi `None` thì giữ ngôn ngữ client gửi. |
| Silero ngoài event loop | `src/asr_server/server.py` | Nạp checkpoint và chạy `push()` trên pool `vad.workers` (mặc định 2 thread). |

**Khóa cấu hình mới:** `asr.backend`, `asr.max_model_len`, `asr.max_num_seqs`, `asr.dtype`, `asr.enforce_eager`, `vad.backend`, `vad.silero_model`, `vad.threshold`, `vad.neg_threshold`, `vad.workers`. Mặc định `asr.gpu_memory_utilization` đổi từ 0,8 thành 0,75 theo bản native.

**Kiểm tra:**

| Hạng mục | Kết quả |
|---|---|
| Test server (`tests/`) | 83 pass: thêm 8 test của zip native (3 backend, 5 Silero), 12 test mới cho marker, ngôn ngữ và Silero, 6 test cấu hình |
| Test eval client | 19 pass |
| Chạy thử đầu cuối, cấu hình mặc định | `silero_vad.jit` chính thức, vLLM giả: khởi động, nhận audio, COMMIT, dừng đều đúng |
| Chạy thử đầu cuối, `qwen_asr` + `webrtc` | Vẫn chạy được khi chọn bằng config |

**Số đo Silero trên máy thử 2 nhân (không phải máy GPU):** nạp checkpoint khoảng 43 ms mỗi stream; xử lý VAD khoảng 2% thời gian thực mỗi stream; khoảng 4,4 MiB RAM mỗi stream.

**Chưa kiểm chứng:**

- Chưa chạy với vLLM và checkpoint Qwen thật. Dạng output `language X<asr_text>...` lấy theo tài liệu công khai của model, chưa xác nhận trên máy GPU.
- Chưa có audio tiếng nói thật để đánh giá độ chính xác của Silero.
- Bản `torch` trên máy thử cảnh báo `torch.jit.load` sắp bị loại bỏ; cần kiểm tra với bản `torch` trên máy GPU.

**Chưa làm:** gom batch nhiều stream vào một lần `LLM.chat()` (`asr.max_num_seqs` vẫn là 1); ép ngôn ngữ cho model.

## 13. Chuyển sang `speech.proto`

Theo các quyết định 14 đến 18 ở mục 1. Hợp đồng đầy đủ nằm ở `asr_server/README.md`, mục "gRPC contract"; mục này chỉ ghi quyết định, cách hiện thực và kết quả.

### 13.1. Cách hiện thực

| Thành phần | File | Ghi chú |
|---|---|---|
| Lớp gRPC | `src/asr_server/speech_service.py` | `SpeechStreamService.Recognize`: kiểm tra hợp đồng, đổi `AudioChunk` thành yêu cầu nội bộ, đổi kết quả thành `RecognitionResult`. |
| Lõi nhận dạng | `src/asr_server/server.py` | `ASRService` không còn phụ thuộc gRPC binding. Logic VAD, ASR, punctuation và các bản sửa B1 đến B8 giữ nguyên. |
| Kiểu dữ liệu nội bộ | `src/asr_server/messages.py` | Thay cho các message của `asr.proto`. |
| Proto và binding | `src/asr_server/proto/speech.proto`, `speech_pb2.py`, `speech_pb2_grpc.py` | `speech.proto` gốc cộng tag 12 đến 16. Sinh lại `speech_pb2.py` bằng `scripts/build_proto.py`. |
| Client thử | `scripts/client.py` | Gửi `AudioChunk` 16 kHz, 80 ms. |
| Client đo WER | `tools/eval/eval_wav_grpc.py` | Gửi như Gateway; đọc `raw_text`, `utterance_id`, `reason`. |

Đã xóa khỏi project: `asr.proto`, `asr_pb2.py`, `asr_pb2_grpc.py` (ở server và ở `tools/eval`), `tests/test_grpc.py`, `tools/eval/generate_bindings.py`.

Khóa cấu hình mới: `asr.use_language_source` (mặc định `false`).

### 13.2. Hành vi

| Nội dung | Hành vi | Nguồn |
|---|---|---|
| Cấu hình stream | Lấy từ chunk đầu tiên. `session_id`, `stream_id`, `sample_rate`, `task`, `rtp_clock_rate` đổi giữa chừng thì `INVALID_ARGUMENT`. | Hệ quả của việc proto không có message cấu hình |
| `task` | Rỗng hoặc `speech_to_text`. `realtime_translation` trả `UNIMPLEMENTED`. | Server không có chức năng dịch |
| `chunk_seq` | Phải tăng đúng 1, sai thì `INVALID_ARGUMENT` (giữ B6 chặt). `-1` ở chunk đầu thì tắt kiểm tra. | Quyết định 3 và 18 |
| `end_of_stream` | Xử lý audio của chunk đó, commit utterance đang mở, đóng stream. | Chú thích trong proto |
| `ts_start`, `ts_end` | `(mốc RTP + round(ms × rtp_clock_rate / 1000)) mod 2^32`, với `ms` là vị trí trong audio tính từ mốc. Mốc là `rtp_timestamp` của chunk đầu; từ bản 1.5 mốc được lấy lại ở mỗi lần `rtp_timestamp` nhảy (mục 14.2). Sau khi quay vòng `ts_end` nhỏ hơn `ts_start`. | Quyết định 17 |
| Ngôn ngữ model báo ngoài `vi`/`en` | Log `WARNING` một lần cho mỗi ngôn ngữ trong stream, `language` để rỗng, utterance xử lý bình thường. | Quyết định 16 |
| `raw_text` | Có ở cả PARTIAL và FINAL. | Quyết định 15 |
| `status` | Luôn `RESULT_STATUS_OK`; utterance lỗi không sinh FINAL mà chỉ ghi log. | Quyết định 7 |
| `COMMIT_REASON_AUDIO_GAP` | Để dành, chưa phát. Hết hạn chờ audio vẫn báo `SILENCE` và ghi log `WARNING`. | Quyết định 7 |

**Các điểm tôi tự chọn khi hiện thực, cần anh xem lại:**

| # | Điểm | Đã chọn | Lý do |
|---|---|---|---|
| a | Giá trị đầu của `seq` | Bắt đầu từ 1 | Proto không quy định; 0 là giá trị mặc định của proto3 nên không phân biệt được với "không gửi". |
| b | Giá trị đầu của `chunk_seq` | Nhận giá trị bất kỳ, chỉ bắt buộc tăng 1 | Gateway gửi từ 0 vẫn đúng; không phải từ chối stream vì lệch quy ước. |
| c | `channels = 0` | Chấp nhận như mono | 0 là giá trị khi trường không được gán. |
| d | `rtp_clock_rate = 0` | Dùng `sample_rate` làm clock và ghi `WARNING` | Đúng chú thích trong proto. Với Gateway gửi 16 kHz mà không gán trường này, tick trả về sẽ theo 16000 chứ không phải 8000. |
| e | `rtp_timestamp` không liên tục | Bản 1.4: chỉ ghi `WARNING`, `ts_start`/`ts_end` vẫn tính theo mốc của chunk đầu. **Đã thay ở bản 1.5** bằng quyết định 20 (mục 14.2). | Thử nghiệm cho thấy tick lệch tới hết cuộc gọi sau một lần nhảy. |
| f | Mã ngôn ngữ đầu vào không nhận ra | Ghi `WARNING`, coi như không gửi | Cùng tinh thần quyết định 16. |
| g | Client đo WER | Mặc định gửi 16 kHz thay cho 8 kHz | Cho giống Gateway; `--rate 8000` nếu cần. |

### 13.3. Đổi tên trường so với `asr.proto`

| `asr.proto` (mục 1 đến 12) | `speech.proto` |
|---|---|
| `event = PARTIAL` / `COMMIT` | `is_final = false` / `true` |
| `utterance_index` | `utterance_id` (tag 12) |
| `revision` | `revision` (tag 13) |
| `raw_text` | `raw_text` (tag 14) |
| `reason` (chuỗi) | `reason` (enum `CommitReason`, tag 15) |
| `start_ms`, `end_ms` | `ts_start`, `ts_end` (tick RTP) |
| `Transcript.language` (tên, ví dụ `Vietnamese`) | `language` (mã `vi` / `en`) |
| `commit_id` | Không còn; dùng `session_id` + `stream_id` + `utterance_id` |
| `sequence` | `chunk_seq` |
| `call_id`, `leg_id` | `session_id`, `stream_id` |
| `EndUtterance` | Không còn |

### 13.4. Kiểm tra

| Hạng mục | Kết quả |
|---|---|
| Test server (`tests/`) | 114 pass, trong đó 39 test mới ở `tests/test_speech_grpc.py`: ánh xạ trường, `end_of_stream`, tick RTP kể cả quay vòng 2^32, `chunk_seq`, ngôn ngữ, kiểm tra đầu vào, gRPC thật |
| Test eval client | 19 pass |
| Chạy thử đầu cuối | `silero_vad.jit` chính thức, vLLM giả, qua `speech.proto`: client thử (có quay vòng RTP) và client đo WER đều nhận kết quả đúng |

**Chưa kiểm chứng:**

- Chưa chạy với Gateway thật, vLLM và checkpoint Qwen thật.
- Đường `--g711` của client thử chạy không lỗi nhưng mới thử bằng tín hiệu tổng hợp, VAD không phát hiện tiếng nói; chưa thử bằng ghi âm thật.

### 13.5. Việc phía Gateway

- Gateway đang dùng `speech.proto` gốc vẫn chạy được: tag mới bị bỏ qua. Khi đó ghép PARTIAL với FINAL theo vị trí: mọi PARTIAL thuộc về FINAL kế tiếp.
- Muốn đọc `utterance_id`, `revision`, `raw_text`, `reason` thì sinh lại binding từ `asr_server/src/asr_server/proto/speech.proto`. `docs/speech_v1_1_proposed.proto` đã được cập nhật cho trùng với file này.
- PARTIAL có `text` rỗng nghĩa là xóa phần text đang hiển thị của utterance đó.
- `Thiet_ke_Speech_Engine_v1.1.md` mục 3 (hợp đồng gRPC) viết trước khi có các quyết định này; chỗ nào khác với `asr_server/README.md` thì README đúng.

## 14. Sửa sau khi review luồng nhận `AudioChunk`

Review luồng từ lúc nhận `AudioChunk` tới lúc phát `RecognitionResult` cho ra hai vấn đề, đều tái hiện được bằng model giả. Sửa theo quyết định 19 và 20 ở mục 1.

### 14.1. Model chậm hơn audio thì stream bị đóng (quyết định 19)

**Hiện tượng.** Mỗi `asr.chunk_sec` audio luôn kích một lần gọi model, kể cả khi stream đang tồn audio chờ. Vòng xử lý của stream đứng chờ lần gọi đó, hàng đợi dài dần tới `server.ingress_max_ms` rồi stream bị đóng bằng `RESOURCE_EXHAUSTED`.

**Sửa.** Trong `feed()` (`src/asr_server/server.py`): khi đã đủ một chunk mà audio chờ của stream vượt `asr.partial_max_backlog_ms` (mặc định 500), bỏ lần gọi cho PARTIAL và giữ audio lại. Lần gọi kế tiếp, khi stream đã bắt kịp, nhận dạng toàn bộ audio đã gom. FINAL (kết thúc utterance, cắt cưỡng bức, kết thúc stream) không bao giờ bị bỏ. `0` nghĩa là không bỏ lần nào.

**Kết quả thử.** Người gửi đúng nhịp thực, 12 giây tiếng nói liên tục, `ingress_max_ms = 3000`, model giả kiểu native vLLM:

| Số stream | Mỗi lần gọi model | Trước khi sửa | Sau khi sửa |
|---|---|---|---|
| 1 | 1,5 s | `RESOURCE_EXHAUSTED` | FINAL đủ |
| 8 | 0,2 s | Cả 8 `RESOURCE_EXHAUSTED` | 8/8 FINAL |
| 20 | 0,1 s | Cả 20 `RESOURCE_EXHAUSTED` | 20/20 FINAL |
| 20 | 0,2 s | Cả 20 `RESOURCE_EXHAUSTED` | Cả 20 `RESOURCE_EXHAUSTED` |

**Giới hạn còn lại.**

- Stream vẫn phải chờ tới lượt trên luồng model duy nhất. Khi một vòng gọi model cho mọi stream đang bận lâu hơn khoảng `ingress_max_ms − partial_max_backlog_ms` (2,5 giây) thì stream vẫn bị đóng, như dòng cuối của bảng. Muốn vượt mức này phải tách hẳn việc nhận dạng khỏi vòng đọc audio, hoặc gom batch nhiều stream vào một lần gọi; cả hai chưa làm.
- Chỉ có tác dụng với `asr.backend: native_vllm`, vì backend này nhận dạng một lần cho mỗi lần gọi. Gói `qwen_asr` tự nhận dạng từng chunk bên trong mỗi lần gọi, nên với backend đó công việc chỉ bị dời lại.
- Đo tải bằng `--fast` phải đặt `asr.partial_max_backlog_ms: 0` cùng với `server.ingress_max_ms: 0`. Nếu không, mọi lần nhận dạng PARTIAL bị bỏ và `processing_rtf` chỉ còn tính phần FINAL.

**Log.** `WARNING asr backlog ...` một lần cho mỗi stream khi bắt đầu bỏ; dòng `commit` có thêm `skipped_partials=N`.

### 14.2. `ts_start`/`ts_end` lệch sau khi `rtp_timestamp` nhảy (quyết định 20)

**Hiện tượng.** Mốc RTP chỉ lấy ở chunk đầu. Khi `rtp_timestamp` nhảy +80000 tick giữa stream, `ts_end` vẫn như khi không nhảy, tức lệch 80000 tick so với RTP gốc cho tới hết cuộc gọi.

**Sửa.** Trong `src/asr_server/speech_service.py`: mỗi chunk có `rtp_timestamp` không nối tiếp audio đã nhận mở một mốc mới tại vị trí đó. Tick của kết quả tính theo mốc gần nhất ở trước vị trí audio.

| Trường hợp | Kết quả |
|---|---|
| Nhảy trước utterance | `ts_start` và `ts_end` đều theo mốc mới |
| Nhảy giữa utterance | `ts_start` theo mốc cũ, `ts_end` theo mốc mới; hiệu hai giá trị không còn bằng độ dài audio |
| Nhảy sau khi utterance kết thúc | Không đổi |
| Vị trí trùng đúng điểm nhảy | Là `ts_end` thì theo mốc cũ, là `ts_start` thì theo mốc mới (khoảng nửa mở) |
| Nhảy lùi, nhảy qua 2^32 | Xử lý như trên, modulo 2^32 |
| Chunk đầu không có audio | Mốc lấy ở chunk đầu tiên có audio; không tính là nhảy |
| `rtp_timestamp` lệch ở mọi chunk | Mỗi chunk một mốc; chỉ giữ mốc của 60 giây audio gần nhất |

**Log.** `WARNING rtp_timestamp jump ... chunk_seq=... expected ..., got ...` cho 3 lần nhảy đầu của mỗi stream, sau đó mỗi 500 lần.

### 14.3. Kiểm tra

| Hạng mục | Kết quả |
|---|---|
| Test server (`tests/`) | 133 pass: thêm 5 test ở `tests/test_backlog.py`, thay 1 test RTP cũ bằng 5 test mới trong `tests/test_speech_grpc.py`; đã gồm 10 test của script triển khai |
| Test eval client | 19 pass |
| Chạy thử đầu cuối | `silero_vad.jit` chính thức, vLLM giả, client gửi đúng nhịp thực qua điểm quay vòng RTP: kết quả đúng, `skipped_partials=0` |

**Chưa kiểm chứng:** hành vi với thời gian nhận dạng thật của Qwen trên GPU; ngưỡng 500 ms là giá trị khởi đầu, cần xem lại sau khi đo.

### 14.4. Điểm đã thấy nhưng chưa sửa

- Mỗi frame VAD 32 ms là một lần chuyển việc sang VAD pool (khoảng 625 lần mỗi giây với 20 stream). Chưa đo được là vấn đề.
- Khi hàng đợi tràn, client chỉ nhận `RESOURCE_EXHAUSTED` sau khi server xử lý xong phần audio đã xếp hàng trước đó.

## 15. Đối chiếu với mã nguồn rtpgw (`media_ai`)

rtpgw là project `D:\NAMCHT\ims\SRC\media_ai` (Go). Đối chiếu ở commit `990bd60`. Sửa theo các quyết định 21 đến 24 ở mục 1. **Các mục 13 và 14 ghi tag 12–16 và `chunk_seq` chặt là theo bản trước; số tag và hành vi đúng hiện nay là ở mục này.**

### 15.1. Cách kiểm

Biên dịch chính các file của rtpgw rồi chạy trên một cuộc gọi thật trong `media_ai\pcap` (luồng PCMU 144 giây, mất một gói RTP):

| Phần của rtpgw | File | Cách dùng |
|---|---|---|
| Giải mã G.711 | `internal/codec/g711.go` | Nguyên văn |
| Resampler 8k→16k (sinc) | `internal/audio/resampler.go` | Nguyên văn |
| Chunker 80 ms, `chunk_seq`, `rtp_timestamp` | `internal/audio/chunker.go` | Nguyên văn |
| Mã hóa `AudioChunk`, giải mã `RecognitionResult` | `internal/ai/grpc_dialer.go` | Nguyên văn hai hàm `marshalAudioChunk`, `unmarshalRecognitionResult` |
| Vòng `process()` và `emitTail()` | `internal/pipeline/audio_pipeline.go` | Viết lại theo đúng trình tự |

Đầu ra đưa vào audio-engine với `silero_vad.jit` chính thức và vLLM giả. **Chưa chạy:** binary rtpgw nối gRPC thật với audio-engine, jitter buffer, đường callback tới MF.

### 15.2. Chiều rtpgw → audio-engine

| Hạng mục | rtpgw gửi | Kết quả ở audio-engine |
|---|---|---|
| Âm thanh | PCM16 mono 16 kHz, 1800 chunk 2560 byte | Nhận đúng |
| Chunk cuối | 104 byte (đuôi resampler), `end_of_stream = true` | Phát FINAL, đóng stream |
| `chunk_seq` | Từ 0, tăng 1 | Nhận đúng |
| `rtp_timestamp` | Bằng timestamp gói RTP trừ 26 tick trễ resampler; mỗi chunk +640 | Nhận đúng |
| Mất một gói RTP | Một bước +800 thay cho +640 | Lấy lại mốc, một `WARNING` |
| `rtp_clock_rate` | 8000 (mọi service đều là PCMU 8000, `handler.go: serviceToCodec`) | Nhận đúng |
| `language`, `task` | Gửi trên mọi chunk | Nhận đúng |
| Kiểu trường | `rtp_clock_rate` là `int32`, `chunk_seq` là `uint64` | Cùng định dạng wire với `uint32`/`int64` của audio-engine |

Silero trên audio cuộc gọi thật: 23 utterance, dài 0,9 đến 3,5 giây, tổng 40 giây tiếng nói trên 144 giây.

RTP từ MF trong hai file pcap (16 luồng): toàn bộ là PCMU 160 byte, không có CN, không có khoảng ngừng gửi; 2 luồng mất đúng một gói. rtpgw không chèn im lặng cho gói mất hay gói CN, nên nếu MF bật DTX thì audio sẽ bị dồn lại; các file pcap hiện có không cho thấy trường hợp này.

### 15.3. Chiều audio-engine → rtpgw

| # | Vấn đề | Bằng chứng | Xử lý |
|---|---|---|---|
| 1 | Tag 12 (`utterance_id`) bị rtpgw đọc thành `topic_label` | Bộ giải mã của rtpgw cho 42/42 kết quả có `topic_label` sai (`WEATHER`, `TRAFFIC`, … `TOPIC_23`), giá trị này vào callback gửi MF | Dời năm trường sang tag 20–24; thêm `topic_label`, `topic_info`, `topic_score` và enum `TopicLabel` vào proto của audio-engine. Sau khi sửa: 42/42 kết quả có `topic_label` rỗng |
| 2 | PARTIAL rỗng không tới MF | `coordinator.go:405` bỏ kết quả không có text và không có topic | Chấp nhận (quyết định 22). audio-engine vẫn gửi; rtpgw bỏ |
| 3 | `chunk_seq` nhảy làm đóng cả session | rtpgw bỏ chunk khi hàng đợi gửi đầy (`grpc_client.go:278`) sau khi đã cấp `chunk_seq`; `INVALID_ARGUMENT` là lỗi vĩnh viễn với rtpgw, nó đóng session (`coordinator.go:252`) | Ghi `WARNING`, xử lý tiếp. Thử bỏ 25 chunk giữa luồng: một `WARNING chunk_seq gap`, mốc RTP được lấy lại, stream chạy tới hết |
| 4 | Task khác `speech_to_text` | Mọi task đều tới audio-engine | Giữ từ chối: `realtime_translation`, `keyword`, `voice_controller_emoji` trả `UNIMPLEMENTED`; task lạ trả `INVALID_ARGUMENT`. rtpgw đóng session ngay, không thử lại |

### 15.4. rtpgw phản ứng với mã lỗi

| Mã từ audio-engine | rtpgw |
|---|---|
| `INVALID_ARGUMENT`, `UNIMPLEMENTED` | Không kết nối lại, đóng session |
| `RESOURCE_EXHAUSTED`, `UNAVAILABLE`, lỗi khác | Kết nối lại tối đa 3 lần (chờ 1, 2, 4 giây), rồi đóng session. Stream mới bắt đầu bằng `chunk_seq` khác 0; audio-engine nhận |

rtpgw không dùng `ts_start`/`ts_end` để xử lý; hai giá trị này được chuyển nguyên vào callback JSON. Cấu hình hiện tại của rtpgw không bật hạn chờ kết quả (`RecvIdleTimeout = 0`), nên một chiều im lặng lâu không làm đứt stream.

### 15.5. Kiểm tra

| Hạng mục | Kết quả |
|---|---|
| Test server (`tests/`) | 139 pass: thêm test số tag và nội dung wire, test đọc `AudioChunk` đúng như rtpgw mã hóa, test `chunk_seq` nhảy, test từ chối `keyword` và `voice_controller_emoji` |
| Test eval client | 19 pass |
| Mã rtpgw + audio-engine trên cuộc gọi thật | 42 kết quả, 23 FINAL; rtpgw giải mã đủ, `topic_label` rỗng, cả 42 được chuyển tiếp |

`docs/speech.proto` trong `audio-engine` cũ hơn proto của rtpgw (thiếu tag 12–14 của `RecognitionResult`, các task `keyword` và `voice_controller_emoji`). File dùng để sinh binding cho audio-engine là `asr_server/src/asr_server/proto/speech.proto`.
