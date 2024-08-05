import copy
import cv2
import numpy as np

import palette

# -----------------------------------------------------------------------------    

def remap_image(img: np.ndarray, target_palette: palette.Palette):
    original_palette: palette.Palette = palette.FromImage(img)
    
    # -------------------------------------------------------------------------
    
    mapping: dict[tuple[int, int, int], palette.PaletteColor] = {}
    for c_in in original_palette.palette.values():
        
        best_match = None
        closest_distance = float(np.inf)
        for c_out in target_palette.palette.values():
            distance = np.linalg.norm(c_in.lab_color.astype(np.float32) - c_out.lab_color.astype(np.float32))
            print("distance", distance)
            if distance < closest_distance:
                closest_distance = distance
                best_match = c_out
                
        assert best_match is not None
        print(closest_distance, "-----------------------")
        mapping[c_in.tuple] = best_match
        
    # -------------------------------------------------------------------------
    
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            img[y, x] = mapping[(img[y, x, 2],
                                 img[y, x, 1],
                                 img[y, x, 0])].color
    
    return img
