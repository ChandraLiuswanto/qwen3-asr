"""DIARIZATION_POOL_SIZE must default to 4, be env-configurable, and reject
nonsense at boot via _positive_int_from_env (a 0-size pool means every
diarization call blocks forever)."""

import os
import unittest
from unittest import mock

from app.core.config import Settings


class DiarizationPoolConfigTest(unittest.TestCase):
    def test_default_is_4(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("DIARIZATION_POOL_SIZE", None)
            self.assertEqual(Settings().DIARIZATION_POOL_SIZE, 4)

    def test_env_override(self) -> None:
        with mock.patch.dict(os.environ, {"DIARIZATION_POOL_SIZE": "2"}):
            self.assertEqual(Settings().DIARIZATION_POOL_SIZE, 2)

    def test_zero_rejected_at_boot(self) -> None:
        with mock.patch.dict(os.environ, {"DIARIZATION_POOL_SIZE": "0"}):
            with self.assertRaises(ValueError):
                Settings()

    def test_negative_rejected_at_boot(self) -> None:
        with mock.patch.dict(os.environ, {"DIARIZATION_POOL_SIZE": "-1"}):
            with self.assertRaises(ValueError):
                Settings()


if __name__ == "__main__":
    unittest.main()
