# Unsupported Language → 400 (Invalid Parameter), Not 500

## Spec

**Problem.** `_normalize_language_name` (`app/services/asr/qwen3_vllm.py`) raises a bare
`ValueError` on languages outside the canonical 30-set. The blanket `@_handle_asr_error`
decorator (`app/services/asr/qwen3_engine.py:88-98`) re-wraps **every** exception into
`DefaultServerErrorException` (50000000), which `get_http_status_code` maps to HTTP 500.
A client typo (`language=xx-nonsense`) therefore surfaces as a server error: wrong retry
semantics for callers, noise in server-error alerting. Found by the final whole-branch
review of the native-prompt-format branch; tracked in bd (the "Unsupported language
returns HTTP 500" issue).

The decorator defect is more general than the language path: **any** already-typed
`APIException` raised inside a decorated engine method (e.g. `InvalidParameterException`
from `audio_validation.py`) is also mangled into a 500 today.

**Decision (user-approved, Option A of three considered).** Keep upstream qwen-asr's
contract — unsupported languages are rejected, never guessed (upstream `validate_language`
raises; title-case normalization is only its pre-validation step, verified against
QwenLM/Qwen3-ASR main). Fix only the error typing/mapping:

1. `_handle_asr_error` passes `APIException` subclasses through unchanged (`except
   APIException: raise` ahead of the blanket catch). Typed errors keep their status
   code; only untyped exceptions get wrapped as `DefaultServerErrorException`.
2. `_normalize_language_name` raises `InvalidParameterException` (40000003) instead of
   bare `ValueError`, following the existing service-layer precedent
   (`audio_validation.py`, `model_selection.py`). Message keeps its content: the
   offending input, its canonical form, and the sorted supported list.

**Resulting behavior.** HTTP endpoints: 40000003 → HTTP 400 via the existing
`get_http_status_code` and registered `APIException` handlers — no endpoint changes.
WebSocket: the generic handler already sends `str(exc)` in an error frame; the frame now
carries the meaningful "Unsupported language ..." message instead of a generic 转写失败
wrap. Connection-close-after-error behavior is deliberately unchanged (scope).

**Explicitly out of scope.** No fallback/pass-through of unvalidated languages into the
assistant prefill (rejected: diverges from upstream, reopens an injection surface since
`language` is unsanitized by design — validation is its sanitizer). No auto-detect
fallback (rejected by user in favor of the strict contract). No WS keep-alive-on-error.
No change to the 30-language set, aliases, temperature, or prompt construction.

**Success criteria.**
- `InvalidParameterException` raised through a `@_handle_asr_error`-decorated method
  reaches the caller unchanged (same instance/status code), pinned by a test.
- Untyped exceptions still wrap to `DefaultServerErrorException`, pinned by a test.
- `_normalize_language_name("xx-nonsense")` raises `InvalidParameterException` with
  status 40000003; `get_http_status_code(40000003) == 400`, pinned.
- Existing suite stays green (182 today; ValueError assertions in
  `tests/test_language_normalization.py` and `tests/test_chat_prompt.py` are updated to
  the new type as part of the change — they pin the same rejection behavior, only the
  exception type narrows).

## Plan

Constraints (inherited from the repo): tests are `unittest`, run
`DEVICE=cpu .venv/bin/python -m unittest discover -s tests`; TDD per task; workers make
NO git commits (orchestrator commits once at the end); in-place feature branch
`feat/unsupported-language-400` off main (no worktree — `.venv` lives in this checkout);
beads in the existing shared tracker, workers claim their assigned bead ID explicitly.

### Task A — decorator passes typed APIExceptions through

Files: `app/services/asr/qwen3_engine.py` (import `APIException` alongside the existing
`DefaultServerErrorException` import; add `except APIException: raise` before the
blanket `except Exception` in `_handle_asr_error`'s wrapper), new
`tests/test_asr_error_decorator.py`.

TDD: write the test first —
- a dummy function decorated with `_handle_asr_error("op")` raising
  `InvalidParameterException("boom")` propagates the SAME exception type with
  `status_code == 40000003` (fails today: arrives as `DefaultServerErrorException`);
- the same decorated function raising `RuntimeError` arrives as
  `DefaultServerErrorException` with `status_code >= 50000000` (already passes — pins
  no-regression);
- `get_http_status_code(40000003) == 400` and `get_http_status_code(50000000) == 500`.
Run the new module RED, implement, run GREEN, then the full suite.

### Task B — `_normalize_language_name` raises `InvalidParameterException` (blocked by A)

Files: `app/services/asr/qwen3_vllm.py` (add `from ...core.exceptions import
InvalidParameterException` matching `audio_validation.py`'s relative-import shape;
change the `raise ValueError(...)` in `_normalize_language_name` to
`raise InvalidParameterException(...)`, same message), plus test updates:
`tests/test_language_normalization.py` (`test_unsupported_language_raises` asserts
`InvalidParameterException` and `status_code == 40000003`),
`tests/test_chat_prompt.py` (the `assertRaises(ValueError)` for `"tl"` in
`test_alias_flow_id_to_indonesian_prefill` becomes `InvalidParameterException`).

Note: `InvalidParameterException` does NOT subclass `ValueError`, so the old assertions
would fail — updating them is part of the task, not test-weakening: the pinned behavior
(rejection with the offending input named in the message) is unchanged.

TDD: update/write tests first, verify RED for the type change, implement, GREEN, full
suite. Expected final count: 182 + 4 new decorator tests = 186.

### Sequencing

B blocked by A (Task B's propagation semantics depend on A's pass-through; also keeps
the shared-file test edits serialized). Single reviewer pass per task on the uncommitted
diff, whole-tree final review, then ONE orchestrator commit.
