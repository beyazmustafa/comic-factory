import base64
import json
import httpx
import numpy as np
from google import genai
from google.genai import types
from factory.api import Api
from factory.config import Settings
from factory.voice import read_wave, synthesize
from tests.test_studio import TemporaryTest, style


class ProviderContractTests(TemporaryTest):
    def test_real_sdk_video_json_request(self):
        requests = []

        def handle(request):
            requests.append(json.loads(request.content))
            return httpx.Response(
                200,
                json={
                    "candidates": [
                        {
                            "content": {
                                "role": "model",
                                "parts": [{"text": '{"observed":true}'}],
                            }
                        }
                    ]
                },
            )

        client = genai.Client(
            api_key="fixture",
            http_options=types.HttpOptions(
                httpx_client=httpx.Client(transport=httpx.MockTransport(handle))
            ),
        )
        api = Api(Settings(), self.root, client=client)
        try:
            result = api.json(
                "Fixture",
                "Observe",
                videos=[
                    (
                        "CANDIDATE",
                        "https://generativelanguage.googleapis.com/v1beta/files/fixture",
                    ),
                ],
            )
        finally:
            api.close()
        self.assertEqual(result, {"observed": True})
        parts = [p for c in requests[0]["contents"] for p in c["parts"]]
        self.assertEqual(len([p for p in parts if "fileData" in p]), 1)
        self.assertEqual(
            requests[0]["generationConfig"]["responseMimeType"], "application/json"
        )

    def test_real_sdk_tts_returns_pcm_wave(self):
        requests = []
        samples = (np.sin(np.arange(24000) * 0.09) * 4000).astype("<i2")

        def handle(request):
            requests.append(json.loads(request.content))
            return httpx.Response(
                200,
                json={
                    "candidates": [
                        {
                            "content": {
                                "role": "model",
                                "parts": [
                                    {
                                        "inlineData": {
                                            "mimeType": "audio/L16;codec=pcm;rate=24000",
                                            "data": base64.b64encode(
                                                samples.tobytes()
                                            ).decode(),
                                        }
                                    }
                                ],
                            }
                        }
                    ]
                },
            )

        client = genai.Client(
            api_key="fixture",
            http_options=types.HttpOptions(
                httpx_client=httpx.Client(transport=httpx.MockTransport(handle))
            ),
        )
        api = Api(Settings(), self.root, client=client)
        try:
            output = synthesize(
                api, "Thor geri döndü.", "Orus", style(), self.root / "test.wav"
            )
        finally:
            api.close()
        self.assertEqual(len(read_wave(output)), 24000)
        config = requests[0]["generationConfig"]
        self.assertEqual(config["responseModalities"], ["AUDIO"])
        self.assertEqual(
            config["speechConfig"]["voice_config"]["prebuilt_voice_config"][
                "voice_name"
            ],
            "Orus",
        )
