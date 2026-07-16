# -*- coding: utf-8 -*-
"""Aliyun-compatible FunASR websocket service."""

import asyncio
import io
import json
import logging
from collections import deque
from contextlib import suppress
from dataclasses import dataclass, field
from enum import IntEnum
from typing import TYPE_CHECKING, Any, Deque, Dict, Optional, cast

import numpy as np
import soundfile as sf

from fastapi import WebSocketDisconnect

if TYPE_CHECKING:
    from .asr.engines import FunASREngine, BaseASREngine

from ..core.config import settings
from ..core.executor import run_sync
from ..core.security import validate_websocket_token
from ..utils.text_processing import apply_itn_to_text
from ..utils.audio_filter import is_nearfield_voice
from ..models.websocket_asr import (
    AliyunASRWSHeader,
    AliyunASRNamespace,
    AliyunASRMessageName,
    AliyunASRStatus,
)
from .asr.runtime import RuntimeEngineLease, get_runtime_router
from .asr.engines import (
    get_global_punc_model,
    get_global_punc_realtime_model,
    get_punc_inference_lock,
    get_punc_realtime_inference_lock,
)

logger = logging.getLogger(__name__)
MAX_PENDING_AUDIO_SECONDS = 10


class ConnectionState(IntEnum):
    """连接状态"""

    READY = 1
    STARTED = 2


@dataclass
class _AudioSession:
    """Mutable recognition state owned by one sequential worker."""

    task_id: str
    params: dict[str, Any]
    audio_cache: Dict[Any, Any]
    audio_buffer: np.ndarray
    sentence_index: int = 0
    audio_time: int = 0
    sentence_active: bool = False
    sentence_start_time: int = 0
    last_sentence_text: str = ""
    sentence_texts: list[str] = field(default_factory=list)
    sentence_texts_raw: list[str] = field(default_factory=list)
    empty_result_count: int = 0


class _BoundedAudioQueue:
    """FIFO queue that applies backpressure by total queued audio samples."""

    def __init__(self, max_samples: int):
        self._max_samples = max(1, max_samples)
        self._queued_samples = 0
        self._items: Deque[np.ndarray] = deque()
        self._finished = False
        self._aborted = False
        self._condition = asyncio.Condition()

    async def put(self, audio: np.ndarray) -> float:
        sample_count = len(audio)
        if sample_count == 0:
            return 0.0
        if sample_count > self._max_samples:
            raise ValueError(
                f"Audio frame has {sample_count} samples, exceeding the {self._max_samples} sample limit"
            )

        async with self._condition:
            blocked_at: Optional[float] = None
            while (
                not self._finished
                and not self._aborted
                and self._queued_samples + sample_count > self._max_samples
            ):
                if blocked_at is None:
                    blocked_at = asyncio.get_running_loop().time()
                await self._condition.wait()

            if self._finished or self._aborted:
                raise RuntimeError("Audio queue is closed")

            self._items.append(audio)
            self._queued_samples += sample_count
            self._condition.notify()
            if blocked_at is None:
                return 0.0
            return asyncio.get_running_loop().time() - blocked_at

    async def get(self) -> Optional[np.ndarray]:
        async with self._condition:
            while not self._items and not self._finished and not self._aborted:
                await self._condition.wait()

            if self._aborted:
                return None
            if not self._items:
                return None

            audio = self._items.popleft()
            self._queued_samples -= len(audio)
            self._condition.notify_all()
            return audio

    async def finish(self) -> None:
        async with self._condition:
            self._finished = True
            self._condition.notify_all()

    async def abort(self) -> None:
        async with self._condition:
            self._aborted = True
            self._items.clear()
            self._queued_samples = 0
            self._condition.notify_all()


class AliyunWebSocketASRService:
    """阿里云WebSocket实时ASR服务"""

    def __init__(self):
        self.asr_engine = None
        self.engine_lease: Optional[RuntimeEngineLease] = None

    async def cleanup(self):
        """清理资源"""
        try:
            if self.engine_lease is not None:
                await self.engine_lease.close()
                self.engine_lease = None
            if self.asr_engine is not None:
                self.asr_engine = None
                logger.info("WebSocket ASR引擎资源已清理")
        except Exception as e:
            logger.warning(f"清理WebSocket ASR资源异常: {e}")

    def _ensure_asr_engine(self) -> "BaseASREngine":
        """确保ASR引擎已加载"""
        assert self.asr_engine is not None, "ASR引擎初始化失败"
        return self.asr_engine

    async def handle_connection(self, websocket, task_id: str) -> None:
        """Handle protocol messages while one worker processes audio in order."""
        state = ConnectionState.READY
        session_id = f"session_{task_id}"
        audio_queue: Optional[_BoundedAudioQueue] = None
        audio_worker: Optional[asyncio.Task[None]] = None
        session: Optional[_AudioSession] = None

        logger.info("[%s] WebSocket ASR connected", task_id)

        try:
            is_valid, validation_message = validate_websocket_token(websocket, task_id)
            if not is_valid:
                await self._send_task_failed(websocket, task_id, validation_message)
                return

            self.engine_lease = await get_runtime_router().acquire_engine(
                "paraformer-large"
            )
            self.asr_engine = self.engine_lease.engine
            if not self.asr_engine.supports_realtime:
                raise RuntimeError(
                    "The configured ASR engine does not support realtime recognition"
                )
            logger.info("WebSocket ASR engine: paraformer-large")

            while True:
                message = await self._receive_message_or_worker_failure(
                    websocket, audio_worker
                )
                if message.get("type") == "websocket.disconnect":
                    raise WebSocketDisconnect(
                        code=message.get("code", 1006),
                        reason=message.get("reason", ""),
                    )

                if "text" in message:
                    data = json.loads(message["text"])
                    header = data.get("header", {})
                    message_name = header.get("name", "")
                    message_task_id = header.get("task_id", "")
                    namespace = header.get("namespace", "")

                    if namespace != AliyunASRNamespace.SPEECH_TRANSCRIBER:
                        await self._send_task_failed(
                            websocket, task_id, "Invalid namespace"
                        )
                        continue

                    if message_name == AliyunASRMessageName.START_TRANSCRIPTION:
                        if state != ConnectionState.READY:
                            await self._send_task_failed(
                                websocket, task_id, "Connection already started"
                            )
                            continue

                        params = self._parse_start_transcription(data, task_id)
                        task_id = message_task_id or task_id
                        session = _AudioSession(
                            task_id=task_id,
                            params=params,
                            audio_cache={},
                            audio_buffer=np.array([], dtype=np.float32),
                        )
                        max_samples = (
                            self._get_sample_rate(params) * MAX_PENDING_AUDIO_SECONDS
                        )
                        audio_queue = _BoundedAudioQueue(max_samples)
                        audio_worker = asyncio.create_task(
                            self._run_audio_worker(websocket, audio_queue, session),
                            name=f"funasr-audio-{task_id}",
                        )
                        await self._send_transcription_started(
                            websocket, task_id, session_id
                        )
                        state = ConnectionState.STARTED
                        continue

                    if message_name == AliyunASRMessageName.STOP_TRANSCRIPTION:
                        if (
                            state != ConnectionState.STARTED
                            or session is None
                            or audio_queue is None
                        ):
                            await self._send_task_failed(
                                websocket, task_id, "Connection not started"
                            )
                            continue
                        if message_task_id != task_id:
                            await self._send_task_failed(
                                websocket, task_id, "Task ID not match"
                            )
                            continue

                        await audio_queue.finish()
                        assert audio_worker is not None
                        await audio_worker
                        await self._finish_session(websocket, session)
                        await self._send_transcription_completed(websocket, task_id)
                        logger.info("[%s] Recognition completed", task_id)
                        return

                    await self._send_task_failed(
                        websocket, task_id, f"Invalid message name: {message_name}"
                    )
                    continue

                if "bytes" not in message:
                    continue
                if (
                    state != ConnectionState.STARTED
                    or session is None
                    or audio_queue is None
                ):
                    await self._send_task_failed(
                        websocket, task_id, "Connection not started"
                    )
                    continue
                if audio_worker is not None and audio_worker.done():
                    await audio_worker

                audio_format = session.params.get("format", "pcm")
                sample_rate = self._get_sample_rate(session.params)
                incoming_audio = self._convert_audio_bytes_to_array(
                    message["bytes"], audio_format, sample_rate, task_id
                )
                blocked_seconds = await audio_queue.put(incoming_audio)
                if blocked_seconds > 0:
                    logger.warning(
                        "[%s] Applied audio ingress backpressure for %.3fs",
                        task_id,
                        blocked_seconds,
                    )

        except WebSocketDisconnect as exc:
            logger.warning(
                "[%s] WebSocket disconnected: code=%s reason=%s",
                task_id,
                exc.code,
                exc.reason or "-",
            )
        except Exception as exc:
            logger.exception("[%s] WebSocket ASR connection failed", task_id)
            with suppress(Exception):
                await self._send_task_failed(websocket, task_id, str(exc))
            with suppress(Exception):
                await websocket.close(code=1011, reason="Audio processing failed")
        finally:
            if audio_queue is not None:
                await audio_queue.abort()
            if audio_worker is not None and not audio_worker.done():
                audio_worker.cancel()
                with suppress(asyncio.CancelledError):
                    await audio_worker

    @staticmethod
    async def _receive_message_or_worker_failure(
        websocket, audio_worker: Optional[asyncio.Task[None]]
    ) -> dict[str, Any]:
        if audio_worker is None:
            return await websocket.receive()

        receive_task = asyncio.create_task(websocket.receive())
        done, _ = await asyncio.wait(
            {receive_task, audio_worker}, return_when=asyncio.FIRST_COMPLETED
        )
        if audio_worker in done:
            if not receive_task.done():
                receive_task.cancel()
                with suppress(asyncio.CancelledError):
                    await receive_task
            await audio_worker
            raise RuntimeError("Audio worker stopped unexpectedly")

        return receive_task.result()

    async def _run_audio_worker(
        self,
        websocket,
        audio_queue: _BoundedAudioQueue,
        session: _AudioSession,
    ) -> None:
        try:
            while (audio := await audio_queue.get()) is not None:
                await self._process_audio_samples(websocket, session, audio)
        except Exception:
            await audio_queue.abort()
            raise

    async def _process_audio_samples(
        self,
        websocket,
        session: _AudioSession,
        incoming_audio: np.ndarray,
    ) -> None:
        session.audio_buffer = np.concatenate([session.audio_buffer, incoming_audio])
        sample_rate = self._get_sample_rate(session.params)
        selected_chunk_size = 9600 if len(session.audio_buffer) >= 9600 else 3840

        while len(session.audio_buffer) >= selected_chunk_size:
            chunk_start_time = session.audio_time
            audio_chunk = session.audio_buffer[:selected_chunk_size]
            session.audio_buffer = session.audio_buffer[selected_chunk_size:]

            threshold = settings.ASR_NEARFIELD_RMS_THRESHOLD
            if session.sentence_active:
                threshold *= 0.6
            is_nearfield, filter_metrics = is_nearfield_voice(
                audio_chunk,
                sample_rate=sample_rate,
                rms_threshold=threshold,
                enable_filter=settings.ASR_ENABLE_NEARFIELD_FILTER,
            )

            is_sentence_end = False
            if not is_nearfield:
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(
                        "[%s] Filtered far-field audio: rms=%.6f threshold=%.6f",
                        session.task_id,
                        filter_metrics["rms_energy"],
                        threshold,
                    )
                session.audio_time += int(len(audio_chunk) / sample_rate * 1000)
                if not session.sentence_active:
                    continue
                result_text = ""
                result_text_raw = ""
                is_silence_frame = False
            else:
                (
                    result_text,
                    result_text_raw,
                    is_silence_frame,
                    session.audio_cache,
                    session.audio_time,
                ) = await self._process_audio_chunk(
                    audio_chunk,
                    session.audio_cache,
                    session.params,
                    session.audio_time,
                    session.task_id,
                )

            max_empty_count = max(
                3, (session.params.get("max_sentence_silence", 800) * 2) // 600
            )
            if not result_text:
                session.empty_result_count += 1
                is_sentence_end = (
                    session.sentence_active
                    and session.empty_result_count >= max_empty_count
                )
            else:
                session.empty_result_count = 0

            if (
                is_silence_frame
                and session.sentence_active
                and session.sentence_texts_raw
            ):
                is_sentence_end = True

            if is_sentence_end and session.sentence_active:
                await self._complete_sentence(websocket, session)
                continue

            if result_text and result_text != session.last_sentence_text:
                session.last_sentence_text = result_text
                if (
                    not session.sentence_texts
                    or result_text != session.sentence_texts[-1]
                ):
                    session.sentence_texts.append(result_text)
                if (
                    not session.sentence_texts_raw
                    or result_text_raw != session.sentence_texts_raw[-1]
                ):
                    session.sentence_texts_raw.append(result_text_raw)

                if not session.sentence_active:
                    session.sentence_active = True
                    session.sentence_start_time = chunk_start_time
                    session.sentence_texts = [result_text]
                    session.sentence_texts_raw = [result_text_raw]
                    session.empty_result_count = 0
                    await self._send_sentence_begin(
                        websocket,
                        session.task_id,
                        session.sentence_index + 1,
                        session.sentence_start_time,
                    )

                if session.params.get("enable_intermediate_result", True):
                    await self._send_transcription_result_changed(
                        websocket,
                        session.task_id,
                        session.sentence_index + 1,
                        session.audio_time,
                        "".join(session.sentence_texts),
                    )

    async def _complete_sentence(self, websocket, session: _AudioSession) -> None:
        (
            _,
            flush_result_text_raw,
            _,
            session.audio_cache,
            session.audio_time,
        ) = await self._process_audio_chunk(
            np.array([], dtype=np.float32),
            session.audio_cache,
            session.params,
            session.audio_time,
            session.task_id,
            is_final=True,
        )
        if flush_result_text_raw and (
            not session.sentence_texts_raw
            or flush_result_text_raw != session.sentence_texts_raw[-1]
        ):
            session.sentence_texts_raw.append(flush_result_text_raw)

        session.sentence_index += 1
        text = "".join(session.sentence_texts_raw)
        if session.params.get("enable_punctuation_prediction", True):
            text = await self._apply_final_punctuation_to_sentence(
                text, session.task_id
            )
        await self._send_sentence_end(
            websocket,
            session.task_id,
            session.sentence_index,
            session.audio_time,
            text,
            session.sentence_start_time,
            enable_itn=session.params.get("enable_inverse_text_normalization", True),
        )
        session.sentence_active = False
        session.sentence_start_time = 0
        session.last_sentence_text = ""
        session.sentence_texts = []
        session.sentence_texts_raw = []
        session.empty_result_count = 0
        session.audio_cache = {}

    async def _finish_session(self, websocket, session: _AudioSession) -> None:
        if session.sentence_active and session.sentence_texts_raw:
            await self._complete_sentence(websocket, session)

    def _parse_start_transcription(self, data: dict, task_id: str) -> dict:
        """解析StartTranscription消息参数"""
        payload = data.get("payload", {})
        params = {
            "format": payload.get("format", "pcm"),
            "sample_rate": payload.get("sample_rate", 16000),
            "enable_intermediate_result": payload.get(
                "enable_intermediate_result", True
            ),
            "enable_punctuation_prediction": payload.get(
                "enable_punctuation_prediction", True
            ),
            "enable_inverse_text_normalization": payload.get(
                "enable_inverse_text_normalization", True
            ),
            "max_sentence_silence": payload.get("max_sentence_silence", 800),
        }
        logger.info(f"[{task_id}] StartTranscription参数解析成功: {params}")
        return params

    @staticmethod
    def _get_sample_rate(params: dict[str, Any]) -> int:
        sample_rate_value = params.get("sample_rate", 16000)
        if isinstance(sample_rate_value, (list, tuple)):
            sample_rate_value = sample_rate_value[0] if sample_rate_value else 16000
        if isinstance(sample_rate_value, str):
            sample_rate_value = sample_rate_value.strip()
            if not sample_rate_value.isdigit():
                raise ValueError(f"Invalid sample rate: {sample_rate_value}")
        try:
            sample_rate = int(sample_rate_value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid sample rate: {sample_rate_value}") from exc
        if sample_rate <= 0:
            raise ValueError(f"Invalid sample rate: {sample_rate}")
        return sample_rate

    def _is_silence_frame(
        self, audio_array: np.ndarray, threshold: float = 0.001
    ) -> bool:
        """检测音频帧是否为静音（优化版）

        Args:
            audio_array: 音频数据数组（float32，范围-1.0到1.0）
            threshold: 静音阈值，低于此值视为静音

        Returns:
            True表示静音帧，False表示有效语音
        """
        if len(audio_array) == 0:
            return True

        # 使用更快的最大振幅检测，避免计算RMS
        max_amplitude = np.max(np.abs(audio_array))

        # 如果最大振幅都很低，直接判定为静音，无需进一步计算
        return max_amplitude < threshold * 2

    @staticmethod
    def _should_apply_realtime_punc(text: str) -> bool:
        """Realtime PUNC currently uses a Chinese punctuation model."""
        return any("\u4e00" <= char <= "\u9fff" for char in text)

    async def _process_audio_chunk(
        self,
        audio_array: np.ndarray,
        cache: Dict,
        params: dict,
        current_audio_time: int,
        task_id: str,
        is_final: bool = False,
    ) -> tuple[str, str, bool, Dict, int]:
        """处理音频块，返回带标点文本、无标点文本、是否静音帧、缓存、音频时长"""
        try:
            asr_engine = self._ensure_asr_engine()

            sample_rate = self._get_sample_rate(params)

            audio_array = np.asarray(audio_array, dtype=np.float32)

            chunk_duration_ms = int(len(audio_array) / sample_rate * 1000)
            new_audio_time = current_audio_time + chunk_duration_ms

            # 计算音频能量用于调试
            max_amplitude = np.max(np.abs(audio_array)) if len(audio_array) > 0 else 0
            mean_amplitude = np.mean(np.abs(audio_array)) if len(audio_array) > 0 else 0
            logger.debug(
                f"[{task_id}] 音频块信息: samples={len(audio_array)}, "
                f"duration={chunk_duration_ms}ms, max={max_amplitude:.4f}, mean={mean_amplitude:.6f}"
            )

            # 只在音频块足够大（>=400ms）时才检测静音帧，避免对小块音频进行检测增加延迟
            # 静音帧检测主要用于主动结束句子，不需要对每个小块都检测
            is_silence = False
            if chunk_duration_ms >= 400:
                is_silence = self._is_silence_frame(audio_array)
                logger.debug(f"[{task_id}] 静音帧检测: is_silence={is_silence}")

            # 根据实际音频样本数自适应调整chunk_size
            # chunk_stride = chunk_size[1], FunASR期望: samples ≈ chunk_stride * 960
            # 支持的标准chunk_stride: 4 (3840 samples, 240ms), 10 (9600 samples, 600ms)
            num_samples = len(audio_array)

            # 计算最接近的chunk_stride
            if num_samples == 3840:
                chunk_stride = 4
            elif num_samples == 9600:
                chunk_stride = 10
            else:
                # 自动选择最接近的标准stride
                chunk_stride = round(num_samples / 960)
                chunk_stride = max(4, min(chunk_stride, 10))  # 限制在4-10之间

            chunk_size = [0, chunk_stride, 5]
            encoder_chunk_look_back = 4
            decoder_chunk_look_back = 1

            logger.debug(
                f"[{task_id}] 使用chunk_size={chunk_size} (samples={num_samples}, "
                f"stride={chunk_stride}, expected={chunk_stride * 960})"
            )

            # 使用线程池执行模型推理，避免阻塞事件循环
            # 将 asr_engine 转换为 FunASREngine 以访问 realtime_model
            funasr_engine = cast("FunASREngine", asr_engine)
            realtime_model = funasr_engine.realtime_model
            if realtime_model is None:
                raise Exception("实时模型未加载")
            # 主ASR推理加全局锁，避免并发连接串音
            result = await run_sync(
                realtime_model.generate,
                input=audio_array,
                cache=cache,
                is_final=is_final,
                chunk_size=chunk_size,
                encoder_chunk_look_back=encoder_chunk_look_back,
                decoder_chunk_look_back=decoder_chunk_look_back,
            )

            logger.debug(f"[{task_id}] ASR模型返回结果: {result}")

            result_text_raw = ""
            result_text_with_punc = ""

            if result and len(result) > 0:
                result_text_raw = result[0].get("text", "").strip()
                result_text_with_punc = result_text_raw

                # Apply realtime punctuation to intermediate results when requested.
                if (
                    result_text_raw
                    and params.get("enable_punctuation_prediction", True)
                    and not params.get("_disable_realtime_punc", False)
                    and self._should_apply_realtime_punc(result_text_raw)
                ):
                    try:
                        punc_realtime_model = get_global_punc_realtime_model(
                            asr_engine.device
                        )
                        if punc_realtime_model:

                            def _apply_realtime_punc():
                                with get_punc_realtime_inference_lock():
                                    return punc_realtime_model.generate(
                                        input=result_text_raw,
                                        cache={},
                                    )

                            punc_result = await run_sync(_apply_realtime_punc)
                            if punc_result and len(punc_result) > 0:
                                result_text_with_punc = (
                                    punc_result[0].get("text", result_text_raw).strip()
                                )
                    except Exception as e:
                        params["_disable_realtime_punc"] = True
                        logger.warning(f"[{task_id}] 实时标点恢复失败: {e}")

            if result_text_with_punc:
                logger.debug(f"[{task_id}] 识别: '{result_text_with_punc}'")

            return (
                result_text_with_punc,
                result_text_raw,
                is_silence,
                cache,
                new_audio_time,
            )

        except Exception:
            logger.exception("[%s] Audio chunk processing failed", task_id)
            raise

    async def _apply_final_punctuation_to_sentence(
        self, text: str, task_id: str
    ) -> str:
        """对完整句子应用最终标点恢复（使用离线标点模型添加完整标点包括句末标点）"""
        if not text:
            return text

        try:
            asr_engine = self._ensure_asr_engine()
            punc_model = get_global_punc_model(asr_engine.device)

            if punc_model is None:
                logger.info(f"[{task_id}] 标点模型未加载，返回原文本")
                return text

            logger.debug(f"[{task_id}] 应用标点恢复: '{text}'")

            def _apply_final_punc():
                with get_punc_inference_lock():
                    return punc_model.generate(input=text, cache={})

            result = await run_sync(_apply_final_punc)

            if result and len(result) > 0:
                punctuated_text = result[0].get("text", text).strip()
                logger.debug(f"[{task_id}] 标点恢复结果: '{punctuated_text}'")
                return punctuated_text
            else:
                return text

        except Exception as e:
            logger.warning(f"[{task_id}] 标点恢复失败: {e}")
            return text

    def _convert_audio_bytes_to_array(
        self, audio_bytes: bytes, audio_format: str, sample_rate: int, task_id: str
    ) -> np.ndarray:
        """将音频字节转换为numpy数组

        Args:
            audio_bytes: 音频字节数据
            audio_format: 音频格式（pcm或wav）
            sample_rate: 采样率
            task_id: 任务ID

        Returns:
            float32的numpy数组，范围-1.0到1.0
        """
        if audio_format == "pcm":
            audio_array = (
                np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            )
        elif audio_format == "wav":
            audio_io = io.BytesIO(audio_bytes)
            audio_array, sr = sf.read(audio_io)
            if sr != sample_rate:
                logger.warning(
                    f"[{task_id}] WAV采样率 {sr} 与配置 {sample_rate} 不一致"
                )
        else:
            raise Exception(f"暂不支持的音频格式: {audio_format}")

        return np.asarray(audio_array, dtype=np.float32)

    def _build_event(
        self,
        task_id: str,
        name: str,
        payload: Optional[dict[str, Any]] = None,
        status: int = AliyunASRStatus.SUCCESS,
        status_text: Optional[str] = None,
    ) -> dict[str, Any]:
        response = {
            "header": {
                "message_id": AliyunASRWSHeader.generate_message_id(),
                "task_id": task_id,
                "namespace": AliyunASRNamespace.SPEECH_TRANSCRIBER,
                "name": name,
                "status": status,
            }
        }
        if status == AliyunASRStatus.SUCCESS:
            response["header"]["status_message"] = "GATEWAY|SUCCESS|Success."
        elif status_text:
            response["header"]["status_text"] = status_text
        if payload is not None:
            response["payload"] = payload
        return response

    async def _send_event(self, websocket, task_id: str, response: dict) -> None:
        try:
            await websocket.send_text(json.dumps(response, ensure_ascii=False))
        except Exception:
            logger.warning(
                "[%s] Failed to send WebSocket event", task_id, exc_info=True
            )
            raise

    async def _send_transcription_started(
        self, websocket, task_id: str, session_id: str
    ):
        await self._send_event(
            websocket,
            task_id,
            self._build_event(
                task_id,
                AliyunASRMessageName.TRANSCRIPTION_STARTED,
                {"session_id": session_id},
            ),
        )

    async def _send_sentence_begin(
        self, websocket, task_id: str, index: int, time: int
    ):
        await self._send_event(
            websocket,
            task_id,
            self._build_event(
                task_id,
                AliyunASRMessageName.SENTENCE_BEGIN,
                {"index": index, "time": time},
            ),
        )

    async def _send_transcription_result_changed(
        self, websocket, task_id: str, index: int, time: int, result: str
    ):
        await self._send_event(
            websocket,
            task_id,
            self._build_event(
                task_id,
                AliyunASRMessageName.TRANSCRIPTION_RESULT_CHANGED,
                {"index": index, "time": time, "result": result},
            ),
        )

    async def _send_sentence_end(
        self,
        websocket,
        task_id: str,
        index: int,
        time: int,
        result: str,
        begin_time: int = 0,
        enable_itn: bool = False,
    ):
        if enable_itn and result:
            logger.debug(f"[{task_id}] 应用ITN: {result}")
            # run_sync, not a direct call: apply_itn_to_text takes the blocking
            # process-wide _wetext_lock, which must never be acquired on the
            # event loop thread or a worker holding it freezes the server.
            result = await run_sync(apply_itn_to_text, result)
            logger.debug(f"[{task_id}] ITN结果: {result}")

        await self._send_event(
            websocket,
            task_id,
            self._build_event(
                task_id,
                AliyunASRMessageName.SENTENCE_END,
                {
                    "index": index,
                    "time": time,
                    "result": result,
                    "begin_time": begin_time,
                },
            ),
        )

    async def _send_transcription_completed(self, websocket, task_id: str):
        await self._send_event(
            websocket,
            task_id,
            self._build_event(task_id, AliyunASRMessageName.TRANSCRIPTION_COMPLETED),
        )

    async def _send_task_failed(self, websocket, task_id: str, reason: str):
        with suppress(Exception):
            await websocket.send_text(
                json.dumps(
                    self._build_event(
                        task_id,
                        AliyunASRMessageName.TASK_FAILED,
                        status=AliyunASRStatus.TASK_FAILED,
                        status_text=reason,
                    ),
                    ensure_ascii=False,
                )
            )
            logger.error(f"[{task_id}] 发送TaskFailed: {reason}")
