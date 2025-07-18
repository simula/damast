import logging
import shutil
import subprocess
import sys
import tempfile
import time
import warnings
from pathlib import Path
from typing import Callable, ClassVar

DAMAST_ARCHIVE_SUPPORT_AVAILABLE = False
try:
    from ratarmountcore.compressions import ARCHIVE_FORMATS, COMPRESSION_FORMATS
    DAMAST_ARCHIVE_SUPPORT_AVAILABLE = True
except Exception as e:
    warnings.warn("ratarmount could not be loaded: archive support is not available")


from damast.core.constants import DAMAST_MOUNT_PREFIX
from damast.core.dataframe import AnnotatedDataFrame

logger = logging.getLogger(__name__)

class Archive:
    """
    Class to wrap and extract archive objects using ratarmount
    """
    filenames: list[str]
    filter_fn: Callable[[str], bool]

    _extracted_files: list[str]
    _mounted_dirs: list[Path]

    _supported_suffixes: ClassVar[list[str]] = None

    @classmethod
    def supported_suffixes(cls):
        """
        Get the list of suffixes for archives and compressed files which are supported
        """
        if not DAMAST_ARCHIVE_SUPPORT_AVAILABLE:
            return []

        if cls._supported_suffixes is not None:
            return cls._supported_suffixes

        cls._supported_suffixes = []
        for k, v in COMPRESSION_FORMATS.items():
            cls._supported_suffixes += v.extensions
        for k, v in ARCHIVE_FORMATS.items():
            cls._supported_suffixes += v.extensions
        return cls._supported_suffixes

    def __enter__(self) -> list[str]:
        """
        Enter function for the contextmanager
        """
        extracted_files = self.mount()
        if extracted_files:
            return extracted_files
        return self.filenames

    def __exit__(self):
        """
        Exit function for the contextmanager
        """
        self.umount()

    def __init__(self,
            filenames: list[str],
            filter_fn: Callable[[str], bool] | None = None):
        self.filenames = sorted(filenames)

        if filter_fn is None:
            self.filter_fn = lambda x: False
        else:
            self.filter_fn = filter_fn

        self._mounted_dirs = []
        self._extracted_files = []

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.umount()

    def ratarmount(self, file, target):
        """
        Call ratarmount to mount and archive
        """
        if not DAMAST_ARCHIVE_SUPPORT_AVAILABLE:
            raise RuntimeError("damast.utils.io.Archive: "
                    "cannot load archive."
                    " 'ratarmount' support is not available."
                )

        response = subprocess.run(["ratarmount","--recursive",file, target])
        if response.returncode != 0:
            raise RuntimeError(f"Mounting archive failed with exitcode {response.returncode}")

        self._mounted_dirs.append(target)


    def umount(self):
        """
        Umount the archive
        """
        for mounted_dir in list(reversed(self._mounted_dirs)):
            for count in range(0,5):
                time.sleep(0.5)
                response = subprocess.run(["fusermount", "-u", mounted_dir])
                if response.returncode == 0:
                    break
                else:
                    logger.debug(f"Retrying to unmount {mounted_dir}")

        for mounted_dir in self._mounted_dirs:
            if Path(mounted_dir).exists():
                shutil.rmtree(mounted_dir)

    def mount(self) -> list[str]:
        """
        Mount the archive (and potentially) inner archives to make the files accessible
        """
        if self._mounted_dirs:
            raise RuntimeError("Archive.mount: looks like these files are already mounted. Call 'umount' first")

        extracted_files = []
        local_mount = tempfile.mkdtemp(prefix=DAMAST_MOUNT_PREFIX)

        for file in self.filenames:
            if Path(file).suffix[1:] in Archive.supported_suffixes():
                logger.info(f"Archive.mount: found archive: {file}")
                target_mount = Path(local_mount) / Path(file).name
                target_mount.mkdir(parents=True, exist_ok=True)

                self.ratarmount(file, target_mount)

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

        self._extracted_files = extracted_files
        return self._extracted_files

    def files(self):
        if not self._mounted_dirs and not self._extracted_files:
            raise RuntimeError("Archive.files: looks like nothing has been mounted. Call 'mount' first.")

        return self._extracted_files
