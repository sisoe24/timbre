#!/usr/bin/env python3

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / 'src'
PACKAGE_ROOT = SRC_ROOT / 'timbre'

# When this file is imported as `timbre`, expose src/timbre as the package path
# so `import timbre.config_loader` still resolves correctly.
__path__ = [str(PACKAGE_ROOT)]


def main() -> None:
    sys.path.insert(0, str(SRC_ROOT))
    from cli.main import main as cli_main
    cli_main()


if __name__ == '__main__':
    main()
