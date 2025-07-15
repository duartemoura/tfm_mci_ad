"""Convert a directory tree of DICOM files to NIfTI format.

Usage
-----
python convert_dicom_to_nifti.py <input_dir> <output_dir> [--no-compress]

Parameters
~~~~~~~~~~
input_dir
    Root directory containing DICOM files (can be nested arbitrarily).
output_dir
    Destination directory where NIfTI files will be written.
--no-compress
    Save NIfTI files as plain ``.nii`` instead of compressed ``.nii.gz``.

The script walks the entire *input_dir* recursively and uses
``dicom2nifti.convert_directory`` to detect individual DICOM series and convert
each one into a separate NIfTI file.  Converted files preserve the original
series description in their filenames so they remain interpretable.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Union

import dicom2nifti  # type: ignore

# -----------------------------------------------------------------------------
# Logging configuration
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s",
)
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Conversion helper
# -----------------------------------------------------------------------------

def convert_dicom_tree(
    input_dir: Union[str, Path],
    output_dir: Union[str, Path],
    *,
    compress: bool = True,
    reorient: bool = True,
) -> None:
    """Convert all DICOM series under *input_dir* into NIfTI files.

    The function is a thin wrapper around
    :pyfunc:`dicom2nifti.convert_directory` that ensures the output directory
    exists and adds a bit of logging.
    """

    input_dir = Path(input_dir).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Starting DICOM → NIfTI conversion")
    logger.info("Input directory : %s", input_dir)
    logger.info("Output directory: %s", output_dir)

    try:
        # dicom2nifti converts each DICOM *series* (not individual files) it
        # encounters.  It automatically determines an appropriate filename.
        dicom2nifti.convert_directory(
            str(input_dir),
            str(output_dir),
            compression=compress,
            reorient=reorient,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Conversion failed: %s", exc)
        raise

    logger.info("All series converted successfully.")

# -----------------------------------------------------------------------------
# CLI entry-point
# -----------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert a DICOM tree to NIfTI")
    parser.add_argument(
        "input_dir",
        type=Path,
        help="Root directory containing DICOM files (recursively walked).",
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Destination directory where NIfTI files will be saved.",
    )
    parser.add_argument(
        "--no-compress",
        action="store_true",
        help="Save output as .nii instead of compressed .nii.gz.",
    )
    return parser.parse_args()


def main() -> None:  # noqa: D401 – simple wrapper
    args = _parse_args()
    convert_dicom_tree(args.input_dir, args.output_dir, compress=not args.no_compress)


if __name__ == "__main__":
    main() 