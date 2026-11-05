"""Small tests independent of ML dependencies."""
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import unittest

p = Path(__file__).resolve().parents[1] / "noScribe" / "voice_pause.py"
spec = spec_from_file_location("voice_pause", p)
mod = module_from_spec(spec)
spec.loader.exec_module(mod)

class TestVoicePauses(unittest.TestCase):
    def test_thresholds(self):
        f = mod.format_voice_pause
        self.assertIsNone(f(0.19))
        self.assertIsNone(f(-0.1))
        self.assertEqual(f(0.2), " (.)")
        self.assertEqual(f(0.6), " (.)")
        self.assertEqual(f(0.61), " (1)")
        self.assertEqual(f(1.8), " (2)")
        self.assertEqual(f(10.4), " (10)")

if __name__ == "__main__":
    unittest.main()
