import unittest

from app.services.asr.model_capabilities import get_slugged_assets


class SluggedAssetsTest(unittest.TestCase):
    def test_support_models_expose_stable_slugs(self) -> None:
        slugs = {asset.slug: asset.model_id for asset in get_slugged_assets()}

        self.assertEqual(
            slugs,
            {
                "VAD": "damo/speech_fsmn_vad_zh-cn-16k-common-pytorch",
                "PUNC": "iic/punc_ct-transformer_zh-cn-common-vocab272727-pytorch",
                "PUNC_REALTIME": (
                    "iic/punc_ct-transformer_zh-cn-common-vad_realtime-vocab272727"
                ),
                "CAMPP_DIARIZATION": "iic/speech_campplus_speaker-diarization_common",
                "CAMPP_SV": "damo/speech_campplus_sv_zh-cn_16k-common",
                "CAMPP_TRANSFORMER": (
                    "damo/speech_campplus-transformer_scl_zh-cn_16k-common"
                ),
            },
        )

    def test_every_slug_is_unique(self) -> None:
        slugs = [asset.slug for asset in get_slugged_assets()]

        self.assertEqual(len(slugs), len(set(slugs)))
