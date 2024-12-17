"""Unit tests for the transform_image function in pillow_resize_and_crop.py."""

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import PIL

from bma_client_lib.pillow_resize_and_crop import transform_image

if TYPE_CHECKING:
    import pytest
    import pytest_mock


@dataclass
class MockImage:
    """Fake PIL.Image class."""

    size: tuple[int, int]

    @property
    def width(self) -> int:
        """Return the width."""
        return self.size[0]

    @property
    def height(self) -> int:
        """Return the height."""
        return self.size[1]

    def resize(self, size: tuple[int, int], *args, **kwargs) -> "MockImage":
        """Noop resize method."""
        self.size = size
        return self

    def crop(self, crop_coords: tuple[int, int, int, int], *args, **kwargs) -> "MockImage":
        """Noop crop method."""
        self.size = (crop_coords[2] - crop_coords[0], crop_coords[3] - crop_coords[1])
        return self


def test_transform_image(
    mocker: "pytest_mock.plugin.MockerFixture", caplog: "pytest.logging.LogCaptureFixture"
) -> None:
    """Make sure the transform_image function crops correctly."""
    caplog.set_level(logging.DEBUG)
    data = {
        (749, 500): {
            # 1:1
            (100, 100): ((150, 100), (25, 0, 125, 100), None),
            (1000, 1000): ((749, 500), (0, 0, 749, 500), (126, 250)),
            # 4:3
            (300, 225): ((338, 225), (19, 0, 319, 225), None),
            (400, 300): ((450, 300), (25, 0, 425, 300), None),
            (4000, 3000): ((749, 500), (0, 0, 749, 500), (1626, 1250)),
            # 16:9
            (400, 225): ((400, 268), (0, 21, 400, 246), None),
        },
        (100, 3000): {
            # 1:1
            (100, 100): ((100, 3000), (0, 1450, 100, 1550), (450, 0)),
            (1000, 1000): ((100, 3000), (0, 1000, 100, 2000), (450, 0)),
            # 4:3
            (300, 225): ((100, 3000), (0, 1387, 100, 1612), (100, 0)),
            (400, 300): ((100, 3000), (0, 1350, 100, 1650), (150, 0)),
            (4000, 3000): ((100, 3000), (0, 0, 100, 3000), (1950, 0)),
            # 16:9
            (400, 225): ((100, 3000), (0, 1387, 100, 1612), (150, 0)),
        },
    }
    mocker.patch("PIL.Image.Image.paste", return_value=MockImage)
    for orig_size, tests in data.items():
        for cropsize, (resize_size, crop_coords, crop_pos) in tests.items():
            img = MockImage(size=orig_size)
            transform_image(img, *cropsize)
            if cropsize[0] > orig_size[0] or cropsize[1] > orig_size[1]:
                # asking for an image larger than the source in at least one dimension,
                # image will be pasted onto a transparent canvas of the requested size
                PIL.Image.Image.paste.assert_called_with(img, crop_pos)
            assert f"resizing to {resize_size[0]}*{resize_size[1]}" in caplog.text
            assert f"after transparency adjustments crop coords are {crop_coords}" in caplog.text
            caplog.clear()


def test_dont_transform_image(caplog: "pytest.logging.LogCaptureFixture") -> None:
    """Make sure the transform_image function returns image when resize/crop is not needed."""
    caplog.set_level(logging.DEBUG)
    img = MockImage(size=(749, 500))
    transform_image(original_img=img, crop_w=749, crop_h=500)
    assert "Image size and requested size are the same (749*500), returning image without resizing" in caplog.text
