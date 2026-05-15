"""(Virtual) Map representation."""

from PIL import Image


class Map:
    """
    This is a simple object representing a ground truth map.
    """

    def __init__(self, path_to_map: str, pixel_size: float) -> None:
        """
        params:
            path_to_map: path to the file containing the map
            pixel_size: how many meters is one pixel (in both directions)
        """
        self.pixel_size = pixel_size  # m / pixel
        self.map = Image.open(path_to_map).convert("L")  # we use grayscale measurements

    def read_value(self, x_m: float, y_m: float) -> int:
        """
        params:
            x_m: x coordinate in meters
            y_m: y coordinate in meters

        returns:
            map intensity at given position in [0, 255]
        """
        x_pixel = int(x_m / self.pixel_size)  # pixel
        y_pixel = int(y_m / self.pixel_size)  # pixel
        return self.map.getpixel((x_pixel, y_pixel))
