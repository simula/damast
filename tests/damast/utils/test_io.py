import bz2
import sys
import tempfile
import types
from pathlib import Path
from zipfile import ZipFile

import pyarrow
import pytest

from damast.core.dataframe import DAMAST_SPEC_SUFFIX, AnnotatedDataFrame
from damast.utils.io import Archive, ArchiveBackend, DataFile


@pytest.mark.skipif(sys.platform.startswith("win"), reason="ratarmount does not (easily) run on windows - zipfile backend should be used")
@pytest.mark.skipif(sys.platform.startswith("darwin"), reason="ratarmount does not run on macos - zipfile backend should be used")
@pytest.mark.parametrize("filename, spec_filename", [
    ["test_ais.csv", f"test_ais{DAMAST_SPEC_SUFFIX}"]
])
def test_archive_ratarmount(data_path, filename, spec_filename, tmp_path):
    output_zip = tmp_path / f"{Path(filename)}.zip"
    with ZipFile(output_zip, 'w') as f:
        f.write(str(data_path / filename), arcname=filename)
        f.write(str(data_path / spec_filename), arcname=spec_filename)


    assert Path(output_zip).exists()

    # default no filter
    with Archive(filenames=[output_zip], backend=ArchiveBackend.RATARMOUNT) as input_files:
        assert len(input_files) == 2

        filenames = [x.name for x in input_files]

        assert filename in filenames
        assert spec_filename in filenames

    # permit only supported files
    with Archive(filenames=[output_zip], filter_fn = lambda x : AnnotatedDataFrame.get_supported_format(Path(x).suffix) is None, backend=ArchiveBackend.RATARMOUNT) as input_files:
        assert len(input_files) == 1

        assert filename in [x.name for x in input_files]


def test_autoload_backend_ratarmount(monkeypatch):
    """Regression: autoload_backend always ended up with ZIPFILE, even when ratarmount was available."""
    formats = types.ModuleType("ratarmountcore.formats")
    formats.ARCHIVE_FORMATS = {}
    formats.COMPRESSION_FORMATS = {}
    monkeypatch.setitem(sys.modules, "ratarmountcore", types.ModuleType("ratarmountcore"))
    monkeypatch.setitem(sys.modules, "ratarmountcore.formats", formats)

    assert Archive(filenames=[])._backend == ArchiveBackend.RATARMOUNT


def test_autoload_backend_zipfile_fallback(monkeypatch):
    # None in sys.modules makes the import raise ImportError
    monkeypatch.setitem(sys.modules, "ratarmountcore.formats", None)

    with pytest.warns(UserWarning, match="ratarmount could not be loaded"):
        archive = Archive(filenames=[])
    assert archive._backend == ArchiveBackend.ZIPFILE


@pytest.mark.parametrize("filename, spec_filename", [
    ["test_ais.csv", f"test_ais{DAMAST_SPEC_SUFFIX}"]
])
def test_archive_zipfile(data_path, filename, spec_filename, tmp_path):
    output_zip = tmp_path / f"{Path(filename)}.zip"
    with ZipFile(output_zip, 'w') as f:
        f.write(str(data_path / filename), arcname=filename)
        f.write(str(data_path / spec_filename), arcname=spec_filename)


    assert Path(output_zip).exists()

    # default no filter
    with Archive(filenames=[output_zip], backend=ArchiveBackend.ZIPFILE) as input_files:
        assert len(input_files) == 2

        filenames = [x.name for x in input_files]

        assert filename in filenames
        assert spec_filename in filenames

    # permit only supported files
    with Archive(filenames=[output_zip], filter_fn = lambda x : AnnotatedDataFrame.get_supported_format(Path(x).suffix) is None, backend=ArchiveBackend.ZIPFILE) as input_files:
        assert len(input_files) == 1

        assert filename in [x.name for x in input_files]


requires_ratarmount = pytest.mark.skipif(sys.platform.startswith(("win", "darwin")),
                                         reason="ratarmount does not run on windows / macos")


@pytest.fixture
def mount_tmpdir(tmp_path, monkeypatch):
    """Let the archive create its mount root in a directory of its own, to check what is left behind."""
    mount_tmpdir = tmp_path / "mounts"
    mount_tmpdir.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(mount_tmpdir))
    return mount_tmpdir


@pytest.mark.parametrize("backend", [
    ArchiveBackend.ZIPFILE,
    pytest.param(ArchiveBackend.RATARMOUNT, marks=requires_ratarmount),
])
@pytest.mark.parametrize("zipped", [True, False])
def test_archive_umount_removes_mountpoint(data_path, tmp_path, mount_tmpdir, backend, zipped):
    input_file = data_path / "test_ais.csv"
    if zipped:
        input_file = tmp_path / "test_ais.csv.zip"
        with ZipFile(input_file, 'w') as f:
            f.write(str(data_path / "test_ais.csv"), arcname="test_ais.csv")

    archive = Archive(filenames=[str(input_file)], backend=backend)
    with archive:
        assert archive._mount_root.parent == mount_tmpdir

    assert list(mount_tmpdir.iterdir()) == []
    assert archive._mount_root is None
    assert archive._mounted_dirs == []

    # after cleanup the same archive can be mounted again
    with archive as input_files:
        assert len(input_files) == 1
    assert list(mount_tmpdir.iterdir()) == []


def test_archive_failed_mount_removes_mountpoint(tmp_path, mount_tmpdir, monkeypatch):
    def failing_mount(self, file, target):
        raise RuntimeError("mount failed")
    monkeypatch.setattr(Archive, "mount_zipfile", failing_mount)

    input_file = tmp_path / "test.zip"
    with ZipFile(input_file, 'w') as f:
        f.writestr("test.csv", "a\n1\n")

    with pytest.raises(RuntimeError, match="mount failed"):
        with Archive(filenames=[str(input_file)], backend=ArchiveBackend.ZIPFILE):
            pass

    assert list(mount_tmpdir.iterdir()) == []


def test_csv_compression_codecs_without_ratarmount(monkeypatch):
    monkeypatch.setitem(sys.modules, "ratarmountcore.formats", None)
    monkeypatch.setattr(DataFile, "_csv_compression_codecs", None)

    assert DataFile.csv_compression_codecs() == {".gz": "gzip", ".zst": "zstd"}


@pytest.mark.parametrize(
    ("filename", "suffix", "stem"),
    [
        ("dir/data.csv", ".csv", "data"),
        ("dir/data.csv.zst", ".csv.zst", "data"),
        ("dir/data.csv.gz", ".csv.gz", "data"),
        ("dir/ais-2024.01.01.csv.zst", ".csv.zst", "ais-2024.01.01"),
        ("dir/data.parquet", ".parquet", "data"),
        # not read directly by polars, so only the last suffix names the format
        ("dir/data.csv.bz2", ".bz2", "data.csv"),
        ("dir/data", "", "data"),
    ],
)
def test_data_file_format_suffix_and_stem(filename, suffix, stem):
    assert DataFile.format_suffix(filename) == suffix
    assert DataFile.data_file_stem(filename) == stem


def test_zipfile_archive_backend_passes_compressed_csv_to_from_files(data_path, tmp_path):
    """
    Without ratarmount (e.g. on macOS) Archive only extracts zip files: a csv.zst is passed on
    unchanged and read by polars directly
    """
    compressed = tmp_path / "test_dataframe.csv.zst"
    sink = pyarrow.BufferOutputStream()
    with pyarrow.CompressedOutputStream(sink, "zstd") as out:
        out.write((data_path / "test_dataframe.csv").read_bytes())
    compressed.write_bytes(sink.getvalue().to_pybytes())

    with Archive(filenames=[str(compressed)], backend=ArchiveBackend.ZIPFILE) as files:
        assert files == [str(compressed)]
        adf = AnnotatedDataFrame.from_files(files, metadata_required=False)
        assert adf.lazyframe.collect().shape == (5, 2)


@requires_ratarmount
def test_ratarmount_archive_mounts_compressed_csv_polars_cannot_read(data_path, tmp_path):
    compressed = tmp_path / "test_dataframe.csv.bz2"
    compressed.write_bytes(bz2.compress((data_path / "test_dataframe.csv").read_bytes()))

    with Archive(filenames=[str(compressed)], backend=ArchiveBackend.RATARMOUNT) as files:
        assert [Path(f).name for f in files] == ["test_dataframe.csv"]
        adf = AnnotatedDataFrame.from_files([str(f) for f in files], metadata_required=False)
        assert adf.lazyframe.collect().shape == (5, 2)
