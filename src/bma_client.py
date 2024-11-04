"""BMA client library."""

import json
import logging
import math
import time
import uuid
from fractions import Fraction
from http import HTTPStatus
from pathlib import Path
from typing import TYPE_CHECKING

import exifread
import httpx
from PIL import Image, ImageOps

logger = logging.getLogger("bma_client")

if TYPE_CHECKING:
    from io import BytesIO

    from django.http import HttpRequest

# maybe these should come from server settings
SKIP_EXIF_TAGS = ["JPEGThumbnail", "TIFFThumbnail", "Filename"]


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
        base_url: str = "https://media.bornhack.dk",
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
        self.settings = r.json()["bma_response"]["settings"]
        return r.json()

    def get_jobs(self, job_filter: str = "?limit=0") -> list[dict[str, str]]:
        """Get a filtered list of the jobs this user has access to."""
        r = self.client.get(self.base_url + f"/api/v1/json/jobs/{job_filter}").raise_for_status()
        response = r.json()["bma_response"]
        logger.debug(f"Returning {len(response)} jobs")
        return response

    def get_file_info(self, file_uuid: uuid.UUID) -> dict[str, str]:
        """Get metadata for a file."""
        r = self.client.get(self.base_url + f"/api/v1/json/files/{file_uuid}/").raise_for_status()
        return r.json()["bma_response"]

    def download(self, file_uuid: uuid.UUID) -> bytes:
        """Download a file from BMA."""
        info = self.get_file_info(file_uuid=file_uuid)
        path = self.path / info["filename"]
        if not path.exists():
            url = self.base_url + info["links"]["downloads"]["original"]
            logger.debug(f"Downloading file {url} ...")
            r = self.client.get(url).raise_for_status()
            logger.debug(f"Done downloading {len(r.content)} bytes, saving to {path}")
            with path.open("wb") as f:
                f.write(r.content)
        return info

    def get_job_assignment(self, file_uuid: uuid.UUID | None = None) -> list[dict[str, dict[str, str]]]:
        """Ask for new job(s) from the API."""
        url = self.base_url + "/api/v1/json/jobs/assign/"
        if file_uuid:
            url += f"?file_uuid={file_uuid}"
        data = {"client_uuid": self.uuid}
        try:
            r = self.client.post(url, data=json.dumps(data)).raise_for_status()
            response = r.json()["bma_response"]
        except httpx.HTTPStatusError as e:
            if e.response.status_code == HTTPStatus.NotFound:
                response = []
            else:
                raise
        logger.debug(f"Returning {len(response)} jobs")
        return response

    def upload_file(self, path: Path, attribution: str, file_license: str) -> dict[str, dict[str, str]]:
        """Upload a file."""
        # is this an image?
        extension = path.suffix[1:]
        for extensions in self.settings["filetypes"]["images"].values():
            if extension.lower() in extensions:
                # this file has the extension of a supported image
                logger.debug(f"Extension {extension} is supported...")
                break
        else:
            # file type not supported
            raise ValueError(f"{path.suffix}")

        # get image dimensions
        with Image.open(path) as image:
            rotated = ImageOps.exif_transpose(image)  # creates a copy with rotation normalised
            logger.debug(
                f"Image has exif rotation info, using post-rotate size {rotated.size} instead of raw size {image.size}"
            )
            width, height = rotated.size

        # open file
        with path.open("rb") as fh:
            files = {"f": (path.name, fh)}
            # build metadata
            data = {
                "attribution": attribution,
                "license": file_license,
                "width": width,
                "height": height,
            }
            # doit
            r = self.client.post(
                self.base_url + "/api/v1/json/files/upload/",
                data={"metadata": json.dumps(data)},
                files=files,
            )
            return r.json()

    def handle_job(self, job: dict[str, str], orig: Path) -> tuple[Image.Image, Image.Exif]:
        """Do the thing and return the result."""
        if job["job_type"] == "ImageConversionJob":
            return self.handle_image_conversion_job(job=job, orig=orig)
        if job["job_type"] == "ImageExifExtractionJob":
            return self.get_exif(orig)
        logger.error(f"Unsupported job type {job['job_type']}")
        return None

    def handle_image_conversion_job(self, job: dict[str, str], orig: Path) -> tuple[Image.Image, Image.Exif]:
        """Handle image conversion job."""
        # load original image
        start = time.time()
        logger.debug(f"Opening original image {orig}...")
        image = Image.open(orig)
        logger.debug(
            f"Opening {orig.stat().st_size} bytes {image.size} source image took {time.time() - start} seconds"
        )

        logger.debug("Rotating image (if needed)...")
        start = time.time()
        image = ImageOps.exif_transpose(image)  # creates a copy with rotation normalised
        logger.debug(f"Rotating image took {time.time() - start} seconds, image is now {image.size}")

        logger.debug("Getting exif metadata from image...")
        start = time.time()
        exif = image.getexif()
        logger.debug(f"Getting exif data took {time.time() - start} seconds")

        logger.debug("Calculating size and ratio...")
        start = time.time()
        if job["aspect_ratio_numerator"] and job["aspect_ratio_denominator"]:
            # height is calculated based on requested width and AR
            ratio = Fraction(job["aspect_ratio_numerator"], job["aspect_ratio_denominator"])
            height = math.floor(job["width"] / ratio)
        else:
            # height is a fraction of width, keeping AR the same
            ratio = None
            height = math.floor(job["width"] / Fraction(*image.size))
        size = math.floor(job["width"]), math.floor(height)
        logger.debug(f"Calculating size and AR took {time.time() - start} seconds")

        logger.debug(f"Desired image size is {size}, AR {ratio}, converting image...")
        start = time.time()
        # custom AR or not?
        if ratio:
            image = ImageOps.fit(image, size)
        else:
            image.thumbnail(size)
        logger.debug(f"Converting image size and AR took {time.time() - start} seconds")

        logger.debug("Done, returning result...")
        return image, exif

    def upload_job_result(self, job_uuid: uuid.UUID, buf: "BytesIO", filename: str) -> dict:
        """Upload the result of a job."""
        size = buf.getbuffer().nbytes
        logger.debug(f"Uploading {size} bytes result for job {job_uuid} with filename {filename}")
        start = time.time()
        files = {"f": (filename, buf)}
        # build metadata
        data = {
            "client_uuid": self.uuid,
        }
        # doit
        r = self.client.post(
            self.base_url + f"/api/v1/json/jobs/{job_uuid}/result/",
            data={"assign": json.dumps(data)},
            files=files,
        ).raise_for_status()
        t = time.time() - start
        logger.debug(f"Done, it took {t} seconds to upload {size} bytes, speed {round(size/t)} bytes/sec")
        return r.json()

    def get_exif(self, fname: Path) -> dict[str, dict[str, str]]:
        """Return a dict with exif data as read by exifread from the file.

        exifread returns a flat dict of key: value pairs where the key
        is a space seperated "IDF: Key" thing, split and group accordingly
        Key: "Image ExifOffset", len 3, value 266
        Key: "GPS GPSVersionID", len 12, value [2, 3, 0, 0]
        """
        with fname.open("rb") as f:
            tags = exifread.process_file(f, details=True)
        grouped = {}
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
