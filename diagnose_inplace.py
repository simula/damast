"""Find what holds the parquet file when `damast annotate --inplace` fails on Windows.

Run from the repository root:  python diagnose_inplace.py [--skip-test-adf] [--trace]

--skip-test-adf  do not keep the test's own AnnotatedDataFrame of the file alive
--trace          print every point at which this process opens/maps or releases the file (slow)
"""
import gc
import os
import sys
import tempfile
from pathlib import Path

import psutil

import damast.core.dataframe as dataframe_module
from damast.cli.main import run
from damast.core.dataframe import AnnotatedDataFrame

OPTIONS = set(sys.argv[1:])  # read before cli() replaces sys.argv
DATA = Path("tests/damast/data")
work = Path(tempfile.mkdtemp())  # created with mode 0o700, as pytest's tmp_path
out = work / "ais.barentswatch.parquet"


def holders():
    """The handles and memory maps of `out` this process holds."""
    process = psutil.Process()
    opened = [f.path for f in process.open_files() if Path(f.path).name == out.name]
    mapped = [m.path for m in process.memory_maps() if Path(m.path).name == out.name]
    return opened, mapped


def report(step):
    """Print whether `out` can be moved, and which handles / memory maps of it this process holds."""
    opened, mapped = holders()
    probe = out.with_name(out.name + ".probe")
    try:
        os.rename(out, probe)
        os.rename(probe, out)
        movable = "yes"
    except OSError as e:
        movable = f"NO ({e})"
    print(f"### [{step}] movable={movable} open={opened} mapped={mapped}", flush=True)


real_replace = os.replace


def replace_with_report(source, target):
    if Path(target) == out and out.exists():
        report("at the inplace replace")
        gc.collect()
        report("at the inplace replace, after gc.collect()")
    return real_replace(source, target)


dataframe_module.os.replace = replace_with_report
dataframe_module.REPLACE_RETRY_DELAYS = ()


def cli(*argv):
    sys.argv = ["damast", "-v", *map(str, argv)]
    try:
        run()
    except SystemExit as e:
        print(f"### damast {argv[0]} exited with {e.code}", flush=True)


print(f"### python {sys.version}", flush=True)
cli("convert", "-f", DATA / "test_ais.barentswatch.csv", "--output-file", out)
report("after convert")

if "--skip-test-adf" not in OPTIONS:
    # as the test does, and keeps alive while annotating
    adf = AnnotatedDataFrame.from_file(filename=out)
    adf.dtype("courseOverGround")
    report("after the test's own from_file")

last_holders = ([], [])
#: Only calls made from these packages are checked - checking after every call is far too slow
TRACED_PACKAGES = tuple(f"{os.sep}{name}{os.sep}" for name in ("damast", "polars", "pyarrow"))


def trace(frame, event, arg):
    """Print where the holders of `out` change, with the Python call stack at that point."""
    global last_holders
    if event not in ("return", "c_return") or not any(p in frame.f_code.co_filename for p in TRACED_PACKAGES):
        return
    current = holders()
    if current != last_holders:
        last_holders = current
        stack, f = [], frame
        while f and len(stack) < 8:
            stack.append(f"{Path(f.f_code.co_filename).name}:{f.f_lineno} {f.f_code.co_name}")
            f = f.f_back
        print(f"### trace open={current[0]} mapped={current[1]} | " + " <- ".join(stack), flush=True)


if "--trace" in OPTIONS:
    sys.setprofile(trace)
cli("annotate", "-f", out, "-m", DATA / "barentswatch.spec.yaml",
    "--apply", "--inplace", "--set-unit", "latitude:rad")
sys.setprofile(None)
report("after annotate")
