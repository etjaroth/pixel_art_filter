import cv2
import numpy as np
import tkinter as tk

import app
import palette
import filters
import helper_functions
from converter_pipeline import ConverterPipeline

# -----------------------------------------------------------------------------

IMAGE_SIZE:   int = 128
p:            int =  1
detail: int = int((IMAGE_SIZE * IMAGE_SIZE) / (p * p))
display_size: int = 512

ADD_EDGES: bool = False

img_filters = [
    filters.simplify_img,
    filters.quantize_img,
    filters.remap_img
]

palette_name: palette.Palette|None = None
palette_name = "duel"
palette_name = "pico-8"
# palette_name = "resurrect-64"

inputImages = [
    "images/landscape1.jpg",
    "images/StarryNight.jpg",
    "images/The_million_march_man.jpg",
]

# -----------------------------------------------------------------------------

display_img_i = 1
def display_img(img: np.ndarray):
    global display_img_i
    
    img = img.copy()
    img = helper_functions.resize_no_interpolation(img, 512, 512)
    cv2.imshow("Image " + str(display_img_i), img)
    
    print(display_img_i)
    display_img_i += 1

# -----------------------------------------------------------------------------

def apply_pixel_filter():
    # Select Palette
    target_palette: palette.Palette = palette.FromFilepath(f"palettes/{palette_name}.gpl")
    
    pipeline: ConverterPipeline = ConverterPipeline(
        target_palette,
        IMAGE_SIZE,
        IMAGE_SIZE,
        p,
        ADD_EDGES
    )
    
    for f in img_filters:
        pipeline.append(f)
    
    for filepath in inputImages:
        img: np.ndarray = cv2.imread(cv2.samples.findFile(filepath))
        img = pipeline.apply(img)
            
        display_img(img)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    # apply_pixel_filter()
    
    root = tk.Tk()
    tkapp = app.ImageApp(root)
    root.mainloop()
