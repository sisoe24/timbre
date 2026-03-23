from __future__ import annotations

import sys
import subprocess
from pathlib import Path

_, *args = sys.argv
RELEASE_SCRIPT = Path(__file__).parent / 'release.sh'


def main():
    subprocess.run(['/bin/bash', RELEASE_SCRIPT, *args])


if __name__ == '__main__':
    main()
