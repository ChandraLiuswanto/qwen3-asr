# -*- coding: utf-8 -*-
"""
基于wetext的ITN（逆文本标准化）工具模块
使用wetext库提供高质量的中文ITN处理
"""

import logging
import threading

logger = logging.getLogger(__name__)

# wetext导入 - 延迟导入以避免初始化问题
_wetext_normalizer = None
_wetext_lock = threading.Lock()  # guards init AND normalize(): FST thread safety unproven


def _get_normalizer():
    """获取wetext标准化器实例（单例模式，caller must hold _wetext_lock）

    The lock contract is enforced, not advisory: _wetext_normalizer is a
    process-wide singleton and normalize() is not proven FST-thread-safe,
    so an unlocked caller would race both the init and the normalize.
    """
    assert _wetext_lock.locked(), "_get_normalizer() requires _wetext_lock to be held"
    global _wetext_normalizer
    if _wetext_normalizer is None:
        try:
            from wetext import Normalizer
            _wetext_normalizer = Normalizer(lang="zh", operator="itn")
            logger.info("WeText ITN模块初始化成功")
        except ImportError as e:
            logger.error(f"导入wetext失败: {e}")
            raise ImportError("请安装wetext库: pip install wetext")
        except Exception as e:
            logger.error(f"初始化wetext失败: {e}")
            raise
    return _wetext_normalizer


def apply_itn_to_text(text: str) -> str:
    """
    对文本应用逆文本标准化（ITN）
    使用wetext库进行高质量的中文ITN处理

    Args:
        text: 语音识别结果文本

    Returns:
        应用ITN后的文本
    """
    if not text or not text.strip():
        return text

    try:
        with _wetext_lock:
            normalizer = _get_normalizer()
            result = normalizer.normalize(text)
        logger.debug(f"ITN处理: '{text}' -> '{result}'")
        return result
    except Exception as e:
        logger.warning(f"ITN处理失败: {text}, 错误: {str(e)}")
        return text


def warmup_itn() -> bool:
    """预热ITN标准化器（在应用启动时调用）

    Constructing Normalizer(lang="zh", operator="itn") loads FSTs and takes
    seconds. Without this warmup that cost is paid by the first request,
    while holding _wetext_lock -- stalling every other ITN caller behind it.
    Doing it at boot keeps the lock's hold time to a single normalize().
    """
    with _wetext_lock:
        _get_normalizer()
        # Exercise normalize() once too: the first call can lazily finish
        # FST setup that construction defers.
        _wetext_normalizer.normalize("一百二十三")
    return True


def normalize_asr_text(text: str, enable_itn: bool) -> str:
    if not enable_itn:
        return text
    return apply_itn_to_text(text)
