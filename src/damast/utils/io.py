import io
import logging
import shutil
import subprocess
import tempfile
import time
import warnings
import zipfile
from collections.abc import Callable
from enum import Enum
from pathlib import Path

from damast.core.constants import DAMAST_MOUNT_PREFIX

logger = logging.getLogger(__name__)


class ArchiveBackend(str, Enum):
    ZIPFILE = 'zipfile'
    RATARMOUNT = 'ratarmount'


class Archive:
    """
    Class to wrap and extract archive objects using ratarmount
    """
    filenames: list[str]
    filter_fn: Callable[[str], bool]

    _extracted_files: list[str]
    _mounted_dirs: list[Path]
    #: Temporary directory holding all mountpoints of this archive
    _mount_root: Path | None = None

    _supported_suffixes: list[str] = None
    _backend: ArchiveBackend = None

    def autoload_backend(self):
        try:
            from ratarmountcore.formats import (  # noqa
                ARCHIVE_FORMATS,
                COMPRESSION_FORMATS,
            )
            self._backend = ArchiveBackend.RATARMOUNT
        except Exception:
            warnings.warn("ratarmount could not be loaded: falling back to zipfile-based support")
            self._backend = ArchiveBackend.ZIPFILE

    def supported_suffixes(self):
        """
        Get the list of suffixes for archives and compressed files which are supported
        """
        if not self._backend:
            raise RuntimeError("Archive.supported_suffixes: ensure that backend is set, e.g., call 'autoload_backend' first")

        fn_name = f"supported_suffixes_{self._backend.value}"
        if not hasattr(self, fn_name):
            raise RuntimeError(f"Missing implementation for {fn_name}")

        return getattr(self, fn_name)()

    def supported_suffixes_zipfile(self):
        if self._supported_suffixes is not None:
            return self._supported_suffixes

        self._supported_suffixes = ["zip"]
        return self._supported_suffixes

    def supported_suffixes_ratarmount(self):
        if self._supported_suffixes is not None:
            return self._supported_suffixes

        from ratarmountcore.formats import (
            ARCHIVE_FORMATS,
            COMPRESSION_FORMATS,
        )
        self._supported_suffixes = []
        for k, v in COMPRESSION_FORMATS.items():
            self._supported_suffixes += v.extensions
        for k, v in ARCHIVE_FORMATS.items():
            self._supported_suffixes += v.extensions
        return self._supported_suffixes

    def __enter__(self) -> list[str]:
        """
        Enter function for the contextmanager
        """
        extracted_files = self.mount()
        if extracted_files:
            return extracted_files
        return self.filenames

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Exit function for the contextmanager
        """
        self.umount()

    def __init__(self,
            filenames: list[str],
            filter_fn: Callable[[str], bool] | None = None,
            backend: ArchiveBackend = None
    ):
        if backend:
            self._backend = backend
        elif self._backend is None:
            self.autoload_backend()

        if type(filenames) is not list:
            raise TypeError(f"Archive.__init__: filenames must be 'list', but were '{type(filenames)}'")

        self.filenames = sorted(filenames)

        if filter_fn is None:
            self.filter_fn = lambda x: False
        else:
            self.filter_fn = filter_fn

        self._mounted_dirs = []
        self._extracted_files = []

    def mount_ratarmount(self, file, target):
        """
        Call ratarmount to mount and archive
        """
        if not self._backend == ArchiveBackend.RATARMOUNT:
            raise RuntimeError("damast.utils.io.Archive: "
                    "cannot load archive."
                    " 'ratarmount' support is not available."
                )

        response = subprocess.run(["ratarmount","--recursive",file, target])
        if response.returncode != 0:
            raise RuntimeError(f"Mounting archive failed with exitcode {response.returncode}")

        self._mounted_dirs.append(target)

    def mount_zipfile(self, file, target):
        """
        Use zipfile to mount and archive
        """
        with zipfile.ZipFile(file, "r") as f:
            for file_in_zip in f.namelist():
                if Path(file_in_zip).suffix != ".zip":
                    f.extract(file_in_zip, target)
                    continue

                dirname = Path(file_in_zip).parent
                extract_dir = target / dirname
                extract_dir.mkdir(parents=True, exist_ok=True)

                # read inner zip file into bytes buffer
                zip_content = io.BytesIO(f.read(file_in_zip))
                inner_zip_file = zipfile.ZipFile(zip_content)
                for i in inner_zip_file.namelist():
                    inner_zip_file.extract(i, extract_dir)

        self._mounted_dirs.append(target)

    def umount(self):
        """
        Umount the archive and remove all mountpoints, including the temporary mount root

        :raise RuntimeError: If a ratarmount mountpoint could not be unmounted - it is then kept,
            so that removing it cannot touch the mounted content
        """
        if self._backend == ArchiveBackend.RATARMOUNT:
            for mounted_dir in list(reversed(self._mounted_dirs)):
                for count in range(5):
                    time.sleep(0.5)

                    response = subprocess.run(["ratarmount", "-u", mounted_dir])

                    if response.returncode == 0:
                        break
                    else:
                        logger.debug(f"Retrying to unmount {mounted_dir}")
                else:
                    raise RuntimeError(f"Archive.umount: failed to unmount {mounted_dir}")

        if self._mount_root is not None and self._mount_root.exists():
            shutil.rmtree(self._mount_root)

        self._mount_root = None
        self._mounted_dirs = []
        self._extracted_files = []

    def mount(self) -> list[str]:
        """
        Mount the archive (and potentially) inner archives to make the files accessible
        """
        if self._mount_root is not None:
            raise RuntimeError("Archive.mount: looks like these files are already mounted. Call 'umount' first")

        extracted_files = []
        self._mount_root = Path(tempfile.mkdtemp(prefix=DAMAST_MOUNT_PREFIX))

        try:
            for file in self.filenames:
                if Path(file).suffix[1:] in self.supported_suffixes():
                    logger.info(f"Archive.mount: found archive: {file}")
                    target_mount = self._mount_root / Path(file).name
                    target_mount.mkdir(parents=True, exist_ok=True)

                    fn = getattr(self, f"mount_{self._backend.value}")
                    fn(file, target_mount)

                    decompressed_files = [x for x in Path(target_mount).glob("**/*") if Path(x).is_file()]
                    for idx, x in enumerate(decompressed_files):
                        if self.filter_fn(x):
                            logger.debug(f"Archive.mount: ignoring unsupported file={x}")
                        else:
                            extracted_files += [ x ]
                elif self.filter_fn(file):
                    logger.debug(f"Archive.mount: ignoring unsupported {file=}")
                else:
                    # no extraction needed, and file suffix is supported
                    extracted_files += [ file ]
        except Exception:
            # __exit__ is not called when __enter__ fails, so clean up what has been mounted so far
            self.umount()
            raise

        self._extracted_files = extracted_files
        return self._extracted_files

    def files(self):
        if not self._mounted_dirs and not self._extracted_files:
            raise RuntimeError("Archive.files: looks like nothing has been mounted. Call 'mount' first.")

        return self._extracted_files
