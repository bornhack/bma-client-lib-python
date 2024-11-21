"""Pillow cropping with sequence (gif, webp) support.

Borrowed from https://gist.github.com/muratgozel/ce1aa99f97fc1a99b3f3ec90cf77e5f5
"""

from math import fabs, floor

from PIL import Image, ImageFile, ImageSequence


def transform_image(original_img: Image.Image, crop_w: int, crop_h: int) -> list[Image.Image | ImageFile.ImageFile]:
    """Resizes and crops the image to the specified crop_w and crop_h if necessary.

    Works with multi frame gif and webp images.

    Args:
      original_img(Image.Image): is the image instance created by pillow ( Image.open(filepath) )
      crop_w(int): is the width in pixels for the image that will be resized and cropped
      crop_h(int): is the height in pixels for the image that will be resized and cropped

    returns:
      Instance of an Image or list of frames which they are instances of an Image individually
    """
    img_w, img_h = (original_img.size[0], original_img.size[1])
    n_frames = getattr(original_img, "n_frames", 1)

    def transform_frame(frame: Image.Image) -> Image.Image | ImageFile.ImageFile:
        """Resizes and crops the individual frame in the image."""
        # resize the image to the specified height if crop_w is null in the recipe
        if crop_w is None:
            if crop_h == img_h:
                return frame
            new_w = floor(img_w * crop_h / img_h)
            new_h = crop_h
            return frame.resize((new_w, new_h), resample=Image.Resampling.LANCZOS)

        # return the original image if crop size is equal to img size
        if crop_w == img_w and crop_h == img_h:
            return frame

        # first resize to get most visible area of the image and then crop
        w_diff = fabs(crop_w - img_w)
        h_diff = fabs(crop_h - img_h)
        enlarge_image = bool(crop_w > img_w or crop_h > img_h)
        shrink_image = bool(crop_w < img_w or crop_h < img_h)

        if enlarge_image is True:
            new_w = floor(crop_h * img_w / img_h) if h_diff > w_diff else crop_w
            new_h = floor(crop_w * img_h / img_w) if h_diff < w_diff else crop_h

        if shrink_image is True:
            new_w = crop_w if h_diff > w_diff else floor(crop_h * img_w / img_h)
            new_h = crop_h if h_diff < w_diff else floor(crop_w * img_h / img_w)

        left = (new_w - crop_w) // 2
        right = left + crop_w
        top = (new_h - crop_h) // 2
        bottom = top + crop_h

        return frame.resize((new_w, new_h), resample=Image.Resampling.LANCZOS).crop((left, top, right, bottom))

    # single frame image
    if n_frames == 1:
        return [transform_frame(original_img)]
    # in the case of a multiframe image
    return [transform_frame(frame) for frame in ImageSequence.Iterator(original_img)]
