"""BMA client library."""

import json
import logging
import math
import time
import uuid
from fractions import Fraction
from http import HTTPStatus
from importlib.metadata import PackageNotFoundError, version
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING

import exifread
import httpx
import magic
from PIL import Image, ImageOps

from .datastructures import (
    ImageConversionJob,
    ImageExifExtractionJob,
    Job,
    JobNotSupportedError,
    ThumbnailJob,
    ThumbnailSourceJob,
)

logger = logging.getLogger("bma_client")

if TYPE_CHECKING:
    from django.http import HttpRequest

    from .datastructures import ExifExtractionJobResult, ImageConversionJobResult, JobResult, ThumbnailSourceJobResult

# maybe these should come from server settings
SKIP_EXIF_TAGS = ["JPEGThumbnail", "TIFFThumbnail", "Filename"]

# get version
try:
    __version__ = version("bma-client-lib")
except PackageNotFoundError:
    __version__ = "0.0.0"


class BmaBearerAuth(httpx.Auth):
    """An httpx.Auth subclass to add Bearer token to requests."""

    def __init__(self, token: str) -> None:
        """Just set the token."""
        self.token = token

    def auth_flow(self, request: "HttpRequest") -> "HttpRequest":
        """Add Bearer token to request headers."""
        request.headers["Authorization"] = f"Bearer {self.token}"
        yield request


class BmaClient:
    """The main BMA Client class."""

    def __init__(
        self,
        oauth_client_id: str,
        refresh_token: str,
        path: Path,
        base_url: str,
        client_uuid: uuid.UUID | None = None,
    ) -> None:
        """Save refresh token, get access token, get or set client uuid."""
        self.oauth_client_id = oauth_client_id
        self.refresh_token = refresh_token
        self.base_url = base_url
        logger.debug("Updating oauth token...")
        self.update_access_token()
        self.uuid = client_uuid if client_uuid else uuid.uuid4()
        self.path = path
        self.skip_exif_tags = SKIP_EXIF_TAGS
        self.get_server_settings()
        self.__version__ = __version__

    def update_access_token(self) -> None:
        """Set or update self.access_token using self.refresh_token."""
        r = httpx.post(
            self.base_url + "/o/token/",
            data={
                "client_id": self.oauth_client_id,
                "refresh_token": self.refresh_token,
                "grant_type": "refresh_token",
            },
        ).raise_for_status()
        data = r.json()
        self.refresh_token = data["refresh_token"]
        logger.warning(f"got new refresh_token: {self.refresh_token}")
        self.access_token = data["access_token"]
        logger.warning(f"got new access_token: {self.access_token}")
        self.auth = BmaBearerAuth(token=self.access_token)
        self.client = httpx.Client(auth=self.auth)

    def get_server_settings(self) -> dict[str, dict[str, dict[str, list[str]]]]:
        """Get BMA settings from server, return as dict."""
        r = self.client.get(
            self.base_url + "/api/v1/json/jobs/settings/",
        ).raise_for_status()
        self.settings = r.json()["bma_response"]
        return r.json()

    def get_jobs(self, job_filter: str = "?limit=0") -> list[Job]:
        """Get a filtered list of the jobs this user has access to."""
        r = self.client.get(self.base_url + f"/api/v1/json/jobs/{job_filter}").raise_for_status()
        response = r.json()["bma_response"]
        logger.debug(f"Returning {len(response)} jobs with filter {job_filter}")
        return response

    def get_file_info(self, file_uuid: uuid.UUID) -> dict[str, str]:
        """Get metadata for a file."""
        r = self.client.get(self.base_url + f"/api/v1/json/files/{file_uuid}/").raise_for_status()
        return r.json()["bma_response"]

    def download(self, url: str, path: Path) -> Path:
        """Download a file to a path."""
        r = self.client.get(url).raise_for_status()
        logger.debug(f"Done downloading {len(r.content)} bytes from {url}, saving to {path}")
        with path.open("wb") as f:
            f.write(r.content)
        return path

    def download_job_source(self, job: Job) -> Path:
        """Download the file needed to do a job."""
        return self.download(
            url=self.base_url + job.source_url,
            path=self.path / job.source_filename,
        )

    def get_job_assignment(self, job_filter: str = "") -> list[Job]:
        """Ask for new job(s) from the API."""
        url = self.base_url + "/api/v1/json/jobs/assign/"
        if job_filter:
            url += job_filter
        data = {
            "client_uuid": self.uuid,
            "client_version": f"bma-client-lib {__version__}",
        }
        try:
            r = self.client.post(url, json=data).raise_for_status()
            response = r.json()["bma_response"]
        except httpx.HTTPStatusError as e:
            if e.response.status_code == HTTPStatus.NOT_FOUND:
                response = []
            else:
                raise
        logger.debug(f"Returning {len(response)} assigned jobs")
        return response

    def unassign_job(self, job: Job) -> bool:
        """Unassign a job."""
        logger.debug(f"Unassigning job {job.job_uuid}")
        self.client.post(
            self.base_url + f"/api/v1/json/jobs/{job.job_uuid}/unassign/",
        ).raise_for_status()
        return True

    def upload_file(self, path: Path, attribution: str, file_license: str) -> dict[str, dict[str, str]]:
        """Upload a file."""
        # get mimetype
        with path.open("rb") as fh:
            mimetype = magic.from_buffer(fh.read(2048), mime=True)

        # find filetype (image, video, audio or document) from mimetype
        for filetype in self.settings["filetypes"]:
            if mimetype in self.settings["filetypes"][filetype]:
                break
        else:
            # unsupported mimetype
            logger.error(
                f"Mimetype {mimetype} is not supported by this BMA server. Supported types {self.settings['filetypes']}"
            )
            raise ValueError(mimetype)

        if filetype == "image":
            # get image dimensions
            with Image.open(path) as image:
                rotated = ImageOps.exif_transpose(image)  # creates a copy with rotation normalised
                if rotated is None:
                    raise ValueError("Rotation")
                logger.debug(
                    f"Image has exif rotation info, using post-rotate size {rotated.size} "
                    f"instead of raw size {image.size}"
                )
                width, height = rotated.size

        # open file
        with path.open("rb") as fh:
            files = {"f": (path.name, fh)}
            # build metadata
            data = {
                "attribution": attribution,
                "license": file_license,
                "mimetype": mimetype,
            }
            if filetype == "image":
                data.update(
                    {
                        "width": width,
                        "height": height,
                    }
                )
            # doit
            r = self.client.post(
                self.base_url + "/api/v1/json/files/upload/",
                data={"metadata": json.dumps(data)},
                files=files,
            )
            return r.json()

    def handle_job(self, job: Job) -> None:
        """Do the thing and upload the result."""
        # make sure the source file for the job is available
        source = self.download_job_source(job)
        # do it
        result: JobResult
        if isinstance(job, ImageConversionJob | ThumbnailJob):
            result = self.handle_image_conversion_job(job=job, orig=source)
            filename = f"{job.job_uuid}.{job.filetype.lower()}"

        elif isinstance(job, ImageExifExtractionJob):
            result = self.get_exif(fname=source)
            filename = "exif.json"

        elif isinstance(job, ThumbnailSourceJob):
            result = self.create_thumbnail_source(job=job)
            filename = job.source_filename

        else:
            raise JobNotSupportedError(job=job)

        self.write_and_upload_result(job=job, result=result, filename=filename)

    def write_and_upload_result(self, job: Job, result: "JobResult", filename: str) -> None:
        """Encode and write the job result to a buffer, then upload."""
        with BytesIO() as buf:
            metadata: dict[str, int | str] = {}
            if isinstance(job, ImageConversionJob | ThumbnailJob):
                image, exif = result
                if not isinstance(image, Image.Image) or not isinstance(exif, Image.Exif):
                    raise TypeError("Fuck")
                # apply format specific encoding options
                kwargs = {}
                if job.mimetype in self.settings["encoding"]["images"]:
                    # this format has custom encoding options, like quality/lossless, apply them
                    kwargs.update(self.settings["encoding"]["images"][job.mimetype])
                    logger.debug(f"Format {job.mimetype} has custom encoding settings, kwargs is now: {kwargs}")
                else:
                    logger.debug(f"No custom settings for format {job.mimetype}")
                image.save(buf, format=job.filetype, exif=exif, **kwargs)

            elif isinstance(job, ImageExifExtractionJob):
                logger.debug(f"Got exif data {result}")
                buf.write(json.dumps(result).encode())

            elif isinstance(job, ThumbnailSourceJob):
                image, exif = result
                if not isinstance(image, Image.Image) or not isinstance(exif, Image.Exif):
                    raise TypeError("Fuck")
                image.save(buf, format="WEBP", lossless=True, quality=1)
                metadata = {"width": 500, "height": image.height, "mimetype": "image/webp"}

            else:
                logger.error("Unsupported job type")
                raise JobNotSupportedError(job=job)

            self.upload_job_result(job=job, buf=buf, filename=filename, metadata=metadata)

    def handle_image_conversion_job(
        self, job: ImageConversionJob, orig: Path, crop_center: tuple[float, float] = (0.5, 0.5)
    ) -> "ImageConversionJobResult":
        """Handle image conversion job."""
        start = time.time()
        logger.debug(f"Opening original image {orig}...")
        image = Image.open(orig)
        logger.debug(
            f"Opening {orig.stat().st_size} bytes {image.size} source image took {time.time() - start} seconds"
        )

        logger.debug("Rotating image (if needed)...")
        start = time.time()
        ImageOps.exif_transpose(image, in_place=True)  # creates a copy with rotation normalised
        if image is None:
            raise ValueError("NoImage")
        orig_ar = Fraction(*image.size)
        logger.debug(
            f"Rotating image took {time.time() - start} seconds, image is now {image.size} original AR is {orig_ar}"
        )

        logger.debug("Getting exif metadata from image...")
        start = time.time()
        exif = image.getexif()
        logger.debug(f"Getting exif data took {time.time() - start} seconds")

        size = int(job.width), int(job.height)
        ratio = Fraction(*size)

        if job.custom_aspect_ratio:
            orig_str = "custom"
        else:
            orig_str = "original"
            if orig_ar != ratio:
                orig_str += "(ish)"

        logger.debug(f"Desired image size is {size}, aspect ratio: {ratio} ({orig_str}), converting image...")
        start = time.time()
        # custom AR or not?
        if job.custom_aspect_ratio:
            image = ImageOps.fit(image=image, size=size, method=Image.Resampling.LANCZOS, centering=crop_center)  # type: ignore[assignment]
        else:
            image.thumbnail(size=size, resample=Image.Resampling.LANCZOS)
        logger.debug(f"Converting image size and AR took {time.time() - start} seconds")

        logger.debug("Done, returning result...")
        return image, exif

    def upload_job_result(
        self,
        job: Job,
        buf: "BytesIO",
        filename: str,
        metadata: dict[str, str | int] | None = None,
    ) -> dict:
        """Upload the result of a job."""
        size = buf.getbuffer().nbytes
        logger.debug(f"Uploading {size} bytes result for job {job.job_uuid} with filename {filename}")
        start = time.time()
        files = {"f": (filename, buf)}
        # build client object
        client = {
            "client_uuid": self.uuid,
            "client_version": "bma-client-lib {__version__}",
        }
        data = {"client": json.dumps(client)}
        if isinstance(job, ThumbnailSourceJob):
            # ThumbnailSourceJob needs a metadata object as well
            data["metadata"] = json.dumps(metadata)
        # doit
        r = self.client.post(
            self.base_url + f"/api/v1/json/jobs/{job.job_uuid}/result/",
            data=data,
            files=files,
        ).raise_for_status()
        t = time.time() - start
        logger.debug(f"Done, it took {t} seconds to upload {size} bytes, speed {round(size/t)} bytes/sec")
        return r.json()

    def get_exif(self, fname: Path) -> "ExifExtractionJobResult":
        """Return a dict with exif data as read by exifread from the file.

        exifread returns a flat dict of key: value pairs where the key
        is a space seperated "IDF: Key" thing, split and group accordingly
        Key: "Image ExifOffset", len 3, value 266
        Key: "GPS GPSVersionID", len 12, value [2, 3, 0, 0]
        """
        with fname.open("rb") as f:
            tags = exifread.process_file(f, details=True)
        grouped: dict[str, dict[str, str]] = {}
        for tag, value in tags.items():
            if tag in SKIP_EXIF_TAGS:
                logger.debug(f"Skipping exif tag {tag}")
                continue
            # group by IDF
            group, *key = tag.split(" ")
            key = key[-1]
            logger.debug(f"Group: {group} Key: {key}, type {value.field_type}, len {len(str(value))}, value {value}")
            if group not in grouped:
                grouped[group] = {}
            grouped[group][key] = str(value)
        return grouped

    def create_album(self, file_uuids: list[uuid.UUID], title: str, description: str) -> dict[str, str]:
        """Create an album."""
        url = self.base_url + "/api/v1/json/albums/create/"
        data = {
            "files": file_uuids,
            "title": title,
            "description": description,
        }
        r = self.client.post(url, json=data).raise_for_status()
        return r.json()["bma_response"]

    def create_thumbnail_source(self, job: ThumbnailSourceJob) -> "ThumbnailSourceJobResult":
        """Create a thumbnail source for this file."""
        info = self.get_file_info(file_uuid=job.basefile_uuid)
        if info["filetype"] == "image":
            # use a max 500px wide version of the image as thumbnail source
            path = self.path / info["filename"]
            original_ratio = Fraction(int(info["width"]), int(info["height"]))
            height = math.floor(500 / original_ratio)
            # just call the regular image conversion method to make a thumbnail
            return self.handle_image_conversion_job(
                job=ImageConversionJob(
                    **job.__dict__,
                    width=500,
                    height=height,
                    custom_aspect_ratio=False,
                    filetype="WEBP",
                    mimetype="image/webp",
                ),
                orig=path,
            )
        # unsupported filetype
        raise JobNotSupportedError(job=job)
