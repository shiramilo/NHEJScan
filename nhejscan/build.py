"""Compile the bundled Messer & Arndt trace_ext program on first use."""
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def ensure_trace_ext():
    on_path = shutil.which("trace_ext")
    if on_path:
        return on_path
    cache = os.path.join(os.path.expanduser("~"), ".cache", "nhejscan")
    os.makedirs(cache, exist_ok=True)
    exe = os.path.join(cache, "trace_ext")
    if not os.path.exists(exe):
        cxx = shutil.which("g++") or shutil.which("c++") or shutil.which("clang++")
        if not cxx:
            sys.exit("No C++ compiler found; install g++/clang++ or provide --trace-ext")
        src = os.path.join(HERE, "cpp")
        subprocess.run([cxx, "-O2", "-o", exe, os.path.join(src, "align.cpp"),
                        os.path.join(src, "alignment.cpp")], check=True)
    return exe
