"""Datastructures used in bma_client_lib."""

import uuid
from dataclasses import dataclass
from typing import TypeAlias

from PIL import Image, ImageFile


@dataclass(kw_only=True)
class BaseJob:
    """Base class inherited by all other job classes."""

    job_type: str
    job_uuid: uuid.UUID
    basefile_uuid: uuid.UUID
    user_uuid: uuid.UUID
    client_uuid: uuid.UUID
    client_version: str
    finished: bool
    source_url: str
    schema_name: str


@dataclass(kw_only=True)
class ImageExifExtractionJob(BaseJob):
    """Represent an ImageExifExtractionJob and result."""

    exifdict: dict[str, dict[str, str]] | None = None


@dataclass(kw_only=True)
class BaseImageResultJob(BaseJob):
    """Base class for jobs with a result containing exif and a list of images."""

    exif: Image.Exif | None = None
    images: list[Image.Image | ImageFile.ImageFile] | None = None


@dataclass(kw_only=True)
class ThumbnailSourceJob(BaseImageResultJob):
    """Represent a ThumbnailSourceJob and result."""


@dataclass(kw_only=True)
class ImageConversionJob(BaseImageResultJob):
    """Represent an ImageConversionJob and result."""

    filetype: str
    width: int
    height: int
    mimetype: str
    custom_aspect_ratio: bool


@dataclass(kw_only=True)
class ThumbnailJob(ImageConversionJob):
    """Represent a ThumbnailJob and result."""


Job: TypeAlias = ImageConversionJob | ImageExifExtractionJob | ThumbnailSourceJob | ThumbnailJob
job_types = {
    "ImageConversionJob": ImageConversionJob,
    "ImageExifExtractionJob": ImageExifExtractionJob,
    "ThumbnailSourceJob": ThumbnailSourceJob,
    "ThumbnailJob": ThumbnailJob,
}


class JobNotSupportedError(Exception):
    """Exception raised when a job is not supported by bma_client_lib for some reason."""

    def __init__(self, job: Job) -> None:
        """Exception raised when a job is not supported by bma_client_lib for some reason."""
        super().__init__(f"{job.job_type} {job.job_uuid} for file {job.basefile_uuid} not supported by this client.")
