#!/usr/bin/env python
"""Single place where every filesystem root is resolved.

The released scripts originally carried absolute paths from the development machine.
They now read environment variables so the code runs anywhere (the roots below, plus PRIOREDIT_OUT
and the PRIOREDIT_PY_* interpreters further down):

  PRIOREDIT_ROOT   this repository's checkout            (default: repo root, inferred)
  PRIOREDIT_DATA   corpora root: MIMIC-CXR reports/metadata, MIMIC-RG4, MS-CXR-T,
                   Chest ImaGenome                        (default: $PRIOREDIT_ROOT/external/data)
  PRIOREDIT_LLMRG4 the LLM-RG4 checkout (base model, annotations, save/ dirs)
                   (default: $PRIOREDIT_ROOT/external/LLM-RG4)

None of these datasets are redistributed here: MIMIC-CXR, MIMIC-RG4, MS-CXR-T and
Chest ImaGenome are obtained from PhysioNet / the original authors under their own
licences. See docs/DATA.md for the expected layout under $PRIOREDIT_DATA.
"""
import os

ROOT = os.environ.get(
    "PRIOREDIT_ROOT",
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
)
DATA = os.environ.get("PRIOREDIT_DATA", os.path.join(ROOT, "external", "data"))
LLMRG4 = os.environ.get("PRIOREDIT_LLMRG4", os.path.join(ROOT, "external", "LLM-RG4"))

# in-repo artifacts shipped with the release
CONFIGS = os.path.join(ROOT, "configs")
LABELS = os.path.join(ROOT, "data", "labels")
GENERATIONS = os.path.join(ROOT, "data", "generations")
SCRIPTS = os.path.join(ROOT, "scripts")

# derived / regenerable outputs (gitignored)
OUT = os.environ.get("PRIOREDIT_OUT", os.path.join(ROOT, "out"))

# interpreters: development used separate conda envs (see docs/ENVIRONMENTS.md).
# Override if your metric/model envs are not the active interpreter.
PY_MODEL = os.environ.get("PRIOREDIT_PY_MODEL", "python")   # LLM-RG4 inference env
PY_METRIC = os.environ.get("PRIOREDIT_PY_METRIC", "python")  # RadEval env
PY_LABEL = os.environ.get("PRIOREDIT_PY_LABEL", "python")    # labeling / analysis env

# The scripts were originally one flat directory and import each other by bare module name
# (e.g. `from state_extractor import extract`). They are grouped into subdirectories here for
# readability, so put every group on sys.path to keep those imports working unchanged.
import sys as _sys
for _d in ("metric", "method", "baselines", "direction_probe", "analysis"):
    _p = os.path.join(SCRIPTS, _d)
    if _p not in _sys.path:
        _sys.path.append(_p)
