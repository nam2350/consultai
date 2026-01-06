#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Deprecated wrapper for batch regression runner."""

from __future__ import annotations

import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from run_batch_regression import main


if __name__ == "__main__":
    print("[deprecated] Use scripts/run_batch_regression.py instead.")
    main()
