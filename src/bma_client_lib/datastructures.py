"""Datastructures used in bma_client_lib."""

import uuid
from dataclasses import dataclass
from typing import TypeAlias

from PIL import Image


@dataclass
class BaseJob:
    """Base class inherited by ImageConversionJob and ImageExifExtractionJob."""

    job_type: str
    job_uuid: uuid.UUID
    basefile_uuid: uuid.UUID
    user_uuid: uuid.UUID
    client_uuid: uuid.UUID
    client_version: str
    finished: bool
    source_url: str
    source_filename: str
    schema_name: str


@dataclass
class ImageConversionJob(BaseJob):
    """Represent an ImageConversionJob."""

    filetype: str
    width: int
    height: int
    mimetype: str
    custom_aspect_ratio: bool


class ImageExifExtractionJob(BaseJob):
    """Represent an ImageExifExtractionJob."""


class ThumbnailSourceJob(BaseJob):
    """Represent a ThumbnailSourceJob."""


class ThumbnailJob(ImageConversionJob):
    """Represent a ThumbnailJob."""


Job: TypeAlias = ImageConversionJob | ImageExifExtractionJob | ThumbnailSourceJob | ThumbnailJob
job_types = {
    "ImageConversionJob": ImageConversionJob,
    "ImageExifExtractionJob": ImageExifExtractionJob,
    "ThumbnailSourceJob": ThumbnailSourceJob,
    "ThumbnailJob": ThumbnailJob,
}

ImageConversionJobResult: TypeAlias = tuple[Image.Image, Image.Exif]
ThumbnailSourceJobResult: TypeAlias = ImageConversionJobResult
ExifExtractionJobResult: TypeAlias = dict[str, dict[str, str]]
JobResult: TypeAlias = ImageConversionJobResult | ExifExtractionJobResult | ThumbnailSourceJobResult


class JobNotSupportedError(Exception):
    """Exception raised when a job is not supported by bma_client_lib for some reason."""

    def __init__(self, job: Job) -> None:
        """Exception raised when a job is not supported by bma_client_lib for some reason."""
        super().__init__(f"{job.job_type} {job.job_uuid} for file {job.basefile_uuid} not supported by this client.")
