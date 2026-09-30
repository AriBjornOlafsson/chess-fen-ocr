import ctypes.util
import os
import sys

if sys.platform == "darwin":
    # Homebrew's cairo isn't on the default dylib search path; cairosvg (via
    # cairocffi) needs this set before it's imported.
    for _prefix in ("/opt/homebrew/opt/cairo/lib", "/usr/local/opt/cairo/lib"):
        if os.path.isdir(_prefix):
            existing = os.environ.get("DYLD_FALLBACK_LIBRARY_PATH", "")
            if _prefix not in existing:
                os.environ["DYLD_FALLBACK_LIBRARY_PATH"] = (
                    f"{_prefix}:{existing}" if existing else _prefix
                )
            break
