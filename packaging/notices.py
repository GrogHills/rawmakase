#!/usr/bin/env python3
"""Write THIRD-PARTY-NOTICES.txt for the Rust crates built into the executable.

MIT, Apache-2.0 and the BSD licenses require the crates' copyright and license
notices to ship with it. cargo-about lists the crates of every platform we ship,
configured by packaging/about.toml, in the form of packaging/notices.hbs, both
taken from the source being packaged so that they match its lockfile. It reads
license files only from the crates' sources, so the same lockfile always gives
the same notices; it never fetches a missing one from a crate's repository.
Give it the manifest of the source the executable was built from: in a
release, that is the tag's checkout, not this one. A source from before the
notices (v0.4.8 and earlier) is packaged without them, as it was released.

  packaging/notices.py [--offline] MANIFEST OUTPUT

cargo-about reads every package in the lockfile, so this first downloads the
crates of every platform; --offline skips that, for a build whose crates are
already fetched (`cargo fetch --locked`, without --target).
"""
import argparse
from pathlib import Path
import subprocess
import sys

NAME = "THIRD-PARTY-NOTICES.txt"


def generate(manifest, output, offline=False):
    """Write the notices for `manifest`'s locked dependencies to `output`.

    Returns False, writing nothing, for a source without the notices' configuration.
    """
    manifest, output = Path(manifest), Path(output)
    config = manifest.parent / "packaging/about.toml"
    template = manifest.parent / "packaging/notices.hbs"
    if not config.is_file():
        print(f"{manifest.parent} has no {config.name}: packaging it without {NAME}", file=sys.stderr)
        return False
    if not offline:
        subprocess.run(["cargo", "fetch", "--locked", "--manifest-path", str(manifest)], check=True)
    result = subprocess.run(
        ["cargo", "about", "generate", "--frozen", "--fail", "--config", str(config),
         "--manifest-path", str(manifest), "--output-file", str(output), str(template)],
        stderr=subprocess.PIPE, text=True)
    # cargo-about only warns about a clarification whose license file changed,
    # and falls back to the generic text; a clean run prints nothing.
    if result.returncode != 0 or result.stderr.strip():
        output.unlink(missing_ok=True)
        raise SystemExit(f"cargo about generate failed:\n{result.stderr}")
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--offline", action="store_true",
                        help="use only crates already fetched for every platform")
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if generate(args.manifest, args.output, args.offline):
        print(args.output)


if __name__ == "__main__":
    sys.exit(main())
