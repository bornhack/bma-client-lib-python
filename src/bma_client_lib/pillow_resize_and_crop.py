"""Pillow cropping with sequence (gif, webp) support.

Originally based on https://gist.github.com/muratgozel/ce1aa99f97fc1a99b3f3ec90cf77e5f5
"""

import logging
from fractions import Fraction
from math import ceil, floor

from PIL import Image, ImageFile, ImageSequence, JpegImagePlugin

logger = logging.getLogger("bma_client")

# some cameras save JPEGs as MPO with embedded thumbnails,
# can also be used for 3d images, handle MPO files as plain JPEGs for now
# https://github.com/python-pillow/Pillow/issues/1138
# https://github.com/python-pillow/Pillow/issues/4603
JpegImagePlugin._getmp = lambda _x: None  # type: ignore[assignment] # noqa: SLF001


def transform_image(
    original_img: Image.Image, crop_w: int, crop_h: int, center_point: tuple[float, float] = (0.5, 0.5)
) -> list[Image.Image | ImageFile.ImageFile]:
    """Shrinks and crops the image to the specified crop_w and crop_h if necessary.

    Works with multi frame gif and webp images.

    Args:
      original_img(Image.Image): is the image instance created by pillow ( Image.open(filepath) )
      crop_w(int): is the desired width in pixels
      crop_h(int): is the desired height in pixels
      center_point(tuple[float,float]): The center point of cropping as a percentage.

    returns:
      List of one or more Image instances
    """
    img_w, img_h = (original_img.size[0], original_img.size[1])
    # sequence?
    n_frames = getattr(original_img, "n_frames", 1)

    def transform_frame(frame: Image.Image) -> Image.Image | ImageFile.ImageFile:
        """Resizes and crops the individual frame in the image."""
        # return the original image if crop size is equal to img size
        if crop_w == img_w and crop_h == img_h:
            logger.debug(
                f"Image size and requested size are the same ({crop_w}*{crop_h}), returning image without resizing"
            )
            return frame

        # resizing is required before cropping only if both image dimensions are bigger than the crop size
        if crop_w < img_w and crop_h < img_h:
            # if calculated height is bigger than requested crop height
            if ceil(crop_w * img_h / img_w) > crop_h:
                # then resize image to requested crop width keeping proportional height
                new_w = crop_w
                new_h = ceil(crop_w * img_h / img_w)
            else:
                # else resize the image to requested crop height keeping proportional width
                new_w = ceil(crop_h * img_w / img_h)
                new_h = crop_h
        else:
            # keep size since one or both dimensions is <= crop size
            new_w = img_w
            new_h = img_h

        # get crop coordinates
        left = floor((new_w - crop_w) * center_point[0])
        top = floor((new_h - crop_h) * center_point[1])
        right = left + crop_w
        bottom = top + crop_h

        orig_ratio = Fraction(img_w, img_h)
        new_ratio = Fraction(new_w, new_h)
        logger.debug(
            f"Original size is {img_w}*{img_h} ({orig_ratio}), requested size is {crop_w}*{crop_h}, resizing "
            f"to {new_w}*{new_h} ({new_ratio}), then cropping - initial crop coord are {left, top, right, bottom}"
        )

        # make sure any space outside the image is transparent
        t_left = max(left, 0)
        t_top = max(top, 0)
        t_right = min(right, new_w)
        t_bottom = min(bottom, new_h)
        logger.debug(f"after transparency adjustments crop coords are {t_left, t_top, t_right, t_bottom}")

        # resize and crop the image
        frame = frame.resize((new_w, new_h), resample=Image.Resampling.LANCZOS).crop((t_left, t_top, t_right, t_bottom))
        logger.debug(f"Result frame size is {frame.width}*{frame.height}")

        logger.debug(f"new {crop_w}*{crop_h} orig {img_w}*{img_h}")
        if crop_w > img_w or crop_h > img_h or frame.width < crop_w:
            # original image has one or both dimensions smaller than the requested size,
            # paste the image onto a transparent canvas exactly as big as requested
            canvas = Image.new("RGBA", (crop_w, crop_h), (0, 0, 0, 0))
            logger.debug(f"Transparent canvas size is {canvas.width}*{canvas.height}")
            c_left = t_left - left
            c_top = t_top - top
            logger.debug(f"Positioning image on canvas at {c_left} {c_top}")
            canvas.paste(frame, (c_left, c_top))
            return canvas
        # original image larger than the requested size in both dimensions,
        # no transparent canvas needed, just return the resized frame as-is
        # frame might be 1px smaller than requested in one or both dimensions
        # due to rounding
        logger.debug(
            f"Image has been downsized from {img_w}*{img_h} to {frame.width}*{frame.height} - "
            f"requested size was {crop_w}*{crop_h}."
        )
        return frame

    # single frame image
    if n_frames == 1:
        return [transform_frame(original_img)]
    # in the case of a multiframe image
    return [transform_frame(frame) for frame in ImageSequence.Iterator(original_img)]
