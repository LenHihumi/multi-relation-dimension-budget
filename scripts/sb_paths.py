"""Where the measurement scripts read and write.

Every script in this directory used to carry an absolute path from the machine
the experiments ran on as its default.  That is wrong twice: nobody else can run
it, and the deposit would publish a username and a home directory.

Resolution order, for each of the four roots:

1. the environment variable (``SB_ROOT``, ``SB_OUT``, ``SB_CACHE``, ``SB_MODELS``)
2. a path derived from this file's location

``SB_ROOT`` is the project root that used to hold ``datasets/``, ``.cache/``
and ``outputs/``; ``SB_OUT`` is the run directory those scripts write into
(``outputs/diag`` in the original tree).  In an unpacked deposit that
directory does not exist, which is expected -- the measurement scripts are the
*re-measurement* path and need the corpora.  ``bash repro.sh``, the path a
reviewer runs, needs none of these.

    cp -r artifact ~/sb && export SB_ROOT=~/sb-data
    python3 scripts/sb_grid.py ...
"""
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
ARTIFACT = HERE.parent
REPO = ARTIFACT.parent

SB_ROOT = os.environ.get("SB_ROOT", str(REPO))
SB_OUT = os.environ.get("SB_OUT", str(ARTIFACT / "outputs" / "diag"))
SB_CACHE = os.environ.get("SB_CACHE", str(REPO / ".cache" / "sb"))
SB_MODELS = os.environ.get("SB_MODELS", str(REPO / "models"))

__all__ = ["SB_ROOT", "SB_OUT", "SB_CACHE", "SB_MODELS", "ARTIFACT", "REPO"]
