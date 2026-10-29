#!/usr/bin/env python3
"""Write THIRD-PARTY-NOTICES.txt for the Rust crates built into the executable.

MIT, Apache-2.0 and the BSD licenses require the crates' copyright and license
notices to ship with it. cargo-about lists the crates the build target depends
on (rustc's host: the packages build natively), configured by
packaging/about.toml, in the form of packaging/notices.hbs. It reads license
files only from the crates' sources, so the same lockfile always gives the same
notices; it never fetches a missing one from a crate's repository. Give it the
manifest of the source the executable was built from: in a release, that is
the tag's checkout, not this one.

  packaging/notices.py [--offline] MANIFEST OUTPUT

cargo-about reads every package in the lockfile, so this first downloads the
crates of every platform; --offline skips that, for a build whose crates are
already fetched (`cargo fetch --locked`, without --target).
"""
import argparse
from pathlib import Path
import subprocess
import sys

CONFIG = Path(__file__).resolve().parent / "about.toml"
TEMPLATE = Path(__file__).resolve().parent / "notices.hbs"
NAME = "THIRD-PARTY-NOTICES.txt"


def host_target():
    """The target rustc builds for by default, which the packages build."""
    for line in subprocess.check_output(["rustc", "-vV"], text=True).splitlines():
        if line.startswith("host: "):
            return line.removeprefix("host: ")
    raise SystemExit("rustc -vV names no host target")


def generate(manifest, output, offline=False):
    """Write the notices for `manifest`'s locked dependencies to `output`."""
    manifest, output = Path(manifest), Path(output)
    if not offline:
        subprocess.run(["cargo", "fetch", "--locked", "--manifest-path", str(manifest)], check=True)
    result = subprocess.run(
        ["cargo", "about", "generate", "--frozen", "--fail", "--config", str(CONFIG),
         "--manifest-path", str(manifest), "--target", host_target(),
         "--output-file", str(output), str(TEMPLATE)],
        stderr=subprocess.PIPE, text=True)
    # cargo-about only warns about a clarification whose license file changed,
    # and falls back to the generic text; a clean run prints nothing.
    if result.returncode != 0 or result.stderr.strip():
        output.unlink(missing_ok=True)
        raise SystemExit(f"cargo about generate failed:\n{result.stderr}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--offline", action="store_true",
                        help="use only crates already fetched for every platform")
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    generate(args.manifest, args.output, args.offline)
    print(args.output)


if __name__ == "__main__":
    sys.exit(main())
