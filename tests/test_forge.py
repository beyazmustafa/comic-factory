import unittest
from types import SimpleNamespace

from factory import forge


class FakeApi:
    def __init__(self, models, image_model=""):
        self._models = models
        self.settings = SimpleNamespace(image_model=image_model)

    def discover(self):
        return {"gemini_raw": self._models}


class ForgeTests(unittest.TestCase):
    def test_protected_names_are_rejected(self):
        self.assertTrue(forge.protected("Spider-Guy"))
        self.assertTrue(forge.protected("the dark BATMAN"))
        self.assertFalse(forge.protected("Ember Vane"))

    def test_image_models_filtered_and_ordered(self):
        api = FakeApi([
            "models/gemini-2.5-flash", "models/gemini-2.5-flash-image",
            "models/gemini-3-pro-image-preview", "models/text-embedding-004",
            "models/imagen-4.0-generate-001", "models/veo-3.0-image-to-video",
        ], image_model="models/gemini-2.5-flash-image")
        models = forge.image_models(api)
        self.assertEqual(models[0], "models/gemini-2.5-flash-image")
        self.assertIn("models/gemini-3-pro-image-preview", models)
        self.assertIn("models/imagen-4.0-generate-001", models)
        self.assertNotIn("models/gemini-2.5-flash", models)
        self.assertNotIn("models/text-embedding-004", models)
        self.assertNotIn("models/veo-3.0-image-to-video", models)
        self.assertEqual(len(models), len(set(models)))

    def test_next_hero_prefers_least_used(self):
        universe = {
            "heroes": [{"name": "A"}, {"name": "B"}, {"name": "C"}],
            "episodes": [{"hero": "A"}, {"hero": "A"}, {"hero": "C"}],
        }
        self.assertEqual(forge.next_hero(universe)["name"], "B")


if __name__ == "__main__":
    unittest.main()
