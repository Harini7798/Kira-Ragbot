"""Force UTF-8 console output.

Scientific abstracts are full of characters Windows' default cp1252 console
encoding can't print — ≤, ≥, µ, °, ×, Greek letters — and printing one raises
UnicodeEncodeError, crashing the script. Calling ``use_utf8()`` at the start of
a script makes stdout/stderr emit UTF-8 (with a safe fallback) so that never
happens, on any platform.
"""
from __future__ import annotations

import sys


def use_utf8() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")
