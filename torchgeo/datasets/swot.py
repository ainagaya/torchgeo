# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""SWOT dataset."""

import os
import re
from collections.abc import Callable, Iterable, Sequence
from typing import ClassVar

import matplotlib.pyplot as plt
import torch
from matplotlib.figure import Figure
from pyproj import CRS

from .errors import RGBBandsMissingError
from .geo import RasterDataset
from .utils import Path, Sample


class SWOT(RasterDataset):
    """Sentinel-2 dataset.

    The `Copernicus Sentinel-2 mission
    <https://sentiwiki.copernicus.eu/web/s2-mission>`_ comprises a
    constellation of two polar-orbiting satellites placed in the same sun-synchronous
    orbit, phased at 180° to each other. It aims at monitoring variability in land
    surface conditions, and its wide swath width (290 km) and high revisit time (10 days
    at the equator with one satellite, and 5 days with 2 satellites under cloud-free
    conditions which results in 2-3 days at mid-latitudes) will support monitoring of
    Earth's surface changes.
    """

    # https://sentinels.copernicus.eu/web/sentinel/user-guides/sentinel-2-msi/naming-convention
    # https://sentinel.esa.int/documents/247904/685211/Sentinel-2-MSI-L2A-Product-Format-Specifications.pdf
    filename_glob = 'SWOT_L3_LR_SSH_Basic_*_*_*_v*.nc'
    filename_regex = (
        r"SWOT_L3_LR_SSH_Basic_"
        r"\d{3}_\d{3}_"
        r"(?P<date>\d{8}T\d{6})_"
        r"\d{8}T\d{6}_"
        r"v\d+\.\d+\.\d+\.nc"
    )
    date_format = "%Y%m%dT%H%M%S"

    # https://sentiwiki.copernicus.eu/web/s2-mission
    all_bands: tuple[str, ...] = (
        'B01',
        'B02',
        'B03',
        'B04',
        'B05',
        'B06',
        'B07',
        'B08',
        'B8A',
        'B09',
        'B10',
        'B11',
        'B12',
    )

    # Native resolutions of each band
    resolutions: ClassVar[dict[str, str]] = {
        'B01': '60m',
        'B02': '10m',
        'B03': '10m',
        'B04': '10m',
        'B05': '20m',
        'B06': '20m',
        'B07': '20m',
        'B08': '10m',
        'B8A': '20m',
        'B09': '60m',
        'B10': '60m',
        'B11': '20m',
        'B12': '20m',
    }

    rgb_bands = ('B04', 'B03', 'B02')

    separate_files = True

    # Central wavelength (μm)
    wavelengths: ClassVar[dict[str, float]] = {
        'B01': 0.4427,
        'B02': 0.4927,
        'B03': 0.5598,
        'B04': 0.6646,
        'B05': 0.7041,
        'B06': 0.7405,
        'B07': 0.7828,
        'B08': 0.8328,
        'B8A': 0.8647,
        'B09': 0.9451,
        'B10': 1.3735,
        'B11': 1.6137,
        'B12': 2.2024,
        # For compatibility with other dataset naming conventions
        'B1': 0.4427,
        'B2': 0.4927,
        'B3': 0.5598,
        'B4': 0.6646,
        'B5': 0.7041,
        'B6': 0.7405,
        'B7': 0.7828,
        'B8': 0.8328,
        'B9': 0.9451,
    }

    def __init__(
        self,
        paths: Path | Iterable[Path] = 'data',
        crs: CRS | None = None,
        res: float | tuple[float, float] = 10,
        bands: Sequence[str] | None = None,
        transforms: Callable[[Sample], Sample] | None = None,
        cache: bool = True,
    ) -> None:
        """Initialize a new Dataset instance.

        Args:
            paths: one or more root directories to search or files to load
            crs: :term:`coordinate reference system (CRS)` to warp to
                (defaults to the CRS of the first file found)
            res: resolution of the dataset in units of CRS in (xres, yres) format. If a
                single float is provided, it is used for both the x and y resolution.
                (defaults to the resolution of the first file found)
            bands: bands to return (defaults to all bands)
            transforms: a function/transform that takes an input sample
                and returns a transformed version
            cache: if True, cache file handle to speed up repeated sampling

        Raises:
            DatasetNotFoundError: If dataset is not found.

        .. versionchanged:: 0.5
            *root* was renamed to *paths*
        """
        bands = bands or self.all_bands
        self.filename_glob = self.filename_glob.format(bands[0])

        if isinstance(res, int | float):
            res = (res, res)

        self.filename_regex = self.filename_regex.format(self.resolutions[bands[0]])
        super().__init__(paths, crs, res, bands, transforms, cache)

    def _update_filepath(self, band: str, filepath: str) -> str:
        """Update `filepath` to point to `band`.

        Args:
            band: band to search for.
            filepath: base filepath to use for searching.

        Returns:
            updated filepath for `band`.
        """
        filepath = super()._update_filepath(band, filepath)

        # Sentinel-2 L2A includes resolution in directory and filename
        directory, filename = os.path.split(filepath)
        supdir, subdir = os.path.split(directory)

        match = re.match(self.filename_regex, filename, re.VERBOSE)
        if match and match.group('resolution'):
            start = match.start('resolution')
            end = match.end('resolution')
            filename = filename[:start] + self.resolutions[band] + filename[end:]
            subdir = 'R' + self.resolutions[band]

        return os.path.join(supdir, subdir, filename)

    def plot(
        self, sample: Sample, show_titles: bool = True, suptitle: str | None = None
    ) -> Figure:
        """Plot a sample from the dataset.

        Args:
            sample: a sample returned by :meth:`RasterDataset.__getitem__`
            show_titles: flag indicating whether to show titles above each panel
            suptitle: optional string to use as a suptitle

        Returns:
            a matplotlib Figure with the rendered sample

        Raises:
            RGBBandsMissingError: If *bands* does not include all RGB bands.

        .. versionchanged:: 0.3
           Method now takes a sample dict, not a Tensor. Additionally, possible to
           show subplot titles and/or use a custom suptitle.
        """
        rgb_indices = []
        for band in self.rgb_bands:
            if band in self.bands:
                rgb_indices.append(self.bands.index(band))
            else:
                raise RGBBandsMissingError()

        image = sample['image'][rgb_indices].permute(1, 2, 0)
        # DN = 10000 * REFLECTANCE
        # https://docs.sentinel-hub.com/api/latest/data/sentinel-2-l2a/
        image = torch.clamp(image / 10000, min=0, max=1)

        fig, ax = plt.subplots(1, 1, figsize=(4, 4))

        ax.imshow(image)
        ax.axis('off')

        if show_titles:
            ax.set_title('Image')

        if suptitle is not None:
            plt.suptitle(suptitle)

        return fig
