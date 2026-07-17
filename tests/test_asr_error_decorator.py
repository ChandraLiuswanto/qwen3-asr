# -*- coding: utf-8 -*-
"""Tests for the `_handle_asr_error` decorator's exception typing.

`_handle_asr_error` (app/services/asr/qwen3_engine.py) used to re-wrap EVERY
exception raised inside a decorated method into `DefaultServerErrorException`
(50000000 -> HTTP 500), even exceptions that were already typed as an
`APIException` subclass with a meaningful status code (e.g.
`InvalidParameterException`, 40000003 -> HTTP 400). That mangled client
errors (bad parameters) into server errors. These tests pin the corrected
behavior: already-typed `APIException` subclasses pass through unchanged,
while untyped exceptions still get wrapped as `DefaultServerErrorException`.
"""

import unittest

from app.core.exceptions import (
    DefaultServerErrorException,
    InvalidParameterException,
    get_http_status_code,
)
from app.services.asr.qwen3_engine import _handle_asr_error


class HandleAsrErrorDecoratorTest(unittest.TestCase):
    def test_typed_api_exception_passes_through_unchanged(self) -> None:
        @_handle_asr_error("op")
        def raises_invalid_parameter():
            raise InvalidParameterException("boom")

        with self.assertRaises(InvalidParameterException) as ctx:
            raises_invalid_parameter()
        self.assertEqual(ctx.exception.status_code, 40000003)

    def test_untyped_exception_still_wraps_to_default_server_error(self) -> None:
        @_handle_asr_error("op")
        def raises_runtime_error():
            raise RuntimeError("x")

        with self.assertRaises(DefaultServerErrorException) as ctx:
            raises_runtime_error()
        self.assertGreaterEqual(ctx.exception.status_code, 50000000)

    def test_get_http_status_code_maps_invalid_parameter_to_400(self) -> None:
        self.assertEqual(get_http_status_code(40000003), 400)

    def test_get_http_status_code_maps_default_server_error_to_500(self) -> None:
        self.assertEqual(get_http_status_code(50000000), 500)


if __name__ == "__main__":
    unittest.main()
