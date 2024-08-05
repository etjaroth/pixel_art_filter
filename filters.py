import cv2
import numpy as np

import palette
import helper_functions

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from converter_pipeline import ConverterPipeline

# -----------------------------------------------------------------------------

def simplify_img(pipeline: "ConverterPipeline", img: np.ndarray) -> np.ndarray:
    """
    Finds superpixels of the image and applies a median blur to remove 
    salt and pepper noise.
    """
    
    img = helper_functions.do_kmeans(pipeline.detail, img, use_position=True, use_gradient=True)
    img = cv2.medianBlur(img, 3)
    
    return img

def quantize_img(pipeline: "ConverterPipeline", img: np.ndarray) -> np.ndarray:
    """
    Reduces the number of colors in the image to the number of colors 
    in the target palette.
    """

    return helper_functions.do_kmeans(len(pipeline.target_palette), img)

def remap_img(pipeline: "ConverterPipeline", img: np.ndarray) -> np.ndarray:
    """
    Remaps the colors in an image to the colors in the target palette.
    """
    
    if pipeline.target_palette is None:
        return img
    
    # -------------------------------------------------------------------------
        
    original_palette: palette.Palette = palette.FromImage(img)
    
    # -------------------------------------------------------------------------
    # Find mapping
    
    mapping: dict[tuple[int, int, int], palette.PaletteColor] = {}
    for c_in in original_palette.palette.values():
        
        best_match = None
        closest_distance = float(np.inf)
        for c_out in pipeline.target_palette.palette.values():
            distance = np.linalg.norm(c_in.lab_color.astype(np.float32) - c_out.lab_color.astype(np.float32))
            print("distance", distance)
            if distance < closest_distance:
                closest_distance = distance
                best_match = c_out
                
        assert best_match is not None
        print(closest_distance, "-----------------------")
        mapping[c_in.tuple] = best_match
        
    # -------------------------------------------------------------------------
    # Apply mapping
    
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            img[y, x] = mapping[(img[y, x, 2],
                                 img[y, x, 1],
                                 img[y, x, 0])].color
    
    return img
