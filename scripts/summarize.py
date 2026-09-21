#!/usr/bin/env python
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from trace_tf_target.summarize_seed_sweep import main

if __name__ == "__main__":
    main()
