from pathlib import Path
import shutil
import subprocess
import tempfile
import tomllib
import unittest
from unittest.mock import patch

import notices

ROOT = Path(__file__).resolve().parents[1]
RECIPES = [ROOT / "packaging/arch" / name / "PKGBUILD" for name in ("rawmakase", "rawmakase-git")]


class LicenseListTests(unittest.TestCase):
    def test_notices_accept_exactly_the_licenses_cargo_deny_allows(self):
        accepted = tomllib.loads(notices.CONFIG.read_text())["accepted"]
        allowed = tomllib.loads((ROOT / "deny.toml").read_text())["licenses"]["allow"]
        self.assertEqual(len(accepted), len(set(accepted)))
        self.assertEqual(set(accepted), set(allowed))


class GenerateTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.output = Path(temporary.name) / notices.NAME
        self.calls = []

    def generate(self, stderr="", returncode=0, offline=False):
        def run(command, **kwargs):
            self.calls.append(command)
            if command[:2] == ["cargo", "about"]:
                self.output.write_text("notices")
            return subprocess.CompletedProcess(command, returncode, stderr=stderr)
        host = "rustc 1.98.1\nhost: aarch64-apple-darwin\nrelease: 1.98.1\n"
        with patch("notices.subprocess.run", side_effect=run), \
                patch("notices.subprocess.check_output", return_value=host):
            notices.generate("source/Cargo.toml", self.output, offline)

    def test_fetches_every_platform_then_reads_only_local_sources_for_the_host(self):
        self.generate()
        fetch, about = self.calls
        self.assertEqual(fetch, ["cargo", "fetch", "--locked", "--manifest-path", "source/Cargo.toml"])
        self.assertEqual(about[:3], ["cargo", "about", "generate"])
        self.assertIn("--frozen", about)
        self.assertEqual(about[about.index("--manifest-path") + 1], "source/Cargo.toml")
        self.assertEqual(about[about.index("--target") + 1], "aarch64-apple-darwin")
        self.assertEqual(about[about.index("--config") + 1], str(notices.CONFIG))
        self.assertEqual(about[-1], str(notices.TEMPLATE))
        self.assertEqual(self.output.read_text(), "notices")

    def test_offline_does_not_fetch(self):
        self.generate(offline=True)
        self.assertEqual([command[:2] for command in self.calls], [["cargo", "about"]])

    def test_a_warning_fails_and_leaves_no_notices(self):
        # What cargo-about prints, exiting 0, when a pinned license file changed.
        warning = ("[WARN] failed to validate all files specified in clarification "
                   "for crate libm 0.2.16: checksum mismatch")
        with self.assertRaisesRegex(SystemExit, "checksum mismatch"):
            self.generate(stderr=warning)
        self.assertFalse(self.output.exists())

    def test_an_error_fails_and_leaves_no_notices(self):
        with self.assertRaisesRegex(SystemExit, "failed"):
            self.generate(returncode=1)
        self.assertFalse(self.output.exists())


@unittest.skipUnless(shutil.which("bash"), "requires bash")
class ArchRecipeTests(unittest.TestCase):
    """makepkg runs each recipe function from srcdir, the app's source in a subdirectory."""

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.src = Path(temporary.name, "src")
        self.pkg = Path(temporary.name, "pkg")
        (self.src / "rawmakase").mkdir(parents=True)
        (self.src / "rawmakase-0").mkdir()

    def recipe(self, recipe, script):
        return subprocess.run(["bash", "-c", f'source "$1"; pkgver=0; CARCH=x86_64; srcdir={self.src}; '
                                             f'pkgdir={self.pkg}; cd "$srcdir"; {script}', "-", recipe],
                              check=True, capture_output=True, text=True).stdout.splitlines()

    def test_prepare_fetches_every_platform_for_cargo_about(self):
        for recipe in RECIPES:
            with self.subTest(recipe=recipe.parent.name):
                self.assertEqual(self.recipe(recipe, 'cargo() { echo "$@"; }; prepare'),
                                 ["fetch --locked"])

    def test_build_writes_the_notices_offline(self):
        for recipe in RECIPES:
            with self.subTest(recipe=recipe.parent.name):
                built = self.recipe(recipe, 'cargo() { :; }; python() { echo "$@"; }; build')
                self.assertEqual(built, [f"packaging/notices.py --offline Cargo.toml {self.src}/{notices.NAME}"])

    def test_package_installs_the_notices(self):
        (self.src / notices.NAME).write_text("notices")
        # The ONNX Runtime archive makepkg extracts beside the source.
        runtime = self.src / "onnxruntime-linux-x64-1.23.2"
        for path in ("lib/libonnxruntime.so.1.23.2", "LICENSE", "ThirdPartyNotices.txt"):
            (runtime / path).parent.mkdir(parents=True, exist_ok=True)
            (runtime / path).write_text("")
        for recipe in RECIPES:
            with self.subTest(recipe=recipe.parent.name):
                self.recipe(recipe, '_ortver=1.23.2; make() { :; }; package')
                licenses = self.pkg / "usr/share/licenses" / recipe.parent.name
                self.assertEqual((licenses / notices.NAME).read_text(), "notices")


if __name__ == "__main__":
    unittest.main()
