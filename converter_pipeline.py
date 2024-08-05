import cv2
import numpy as np

import filters
import helper_functions
import palette

from typing import Callable

display_img_i = 1
def display_img(img: np.ndarray):
    global display_img_i
    
    img = img.copy()
    img = helper_functions.resize_no_interpolation(img, 512, 512)
    cv2.imshow("Image " + str(display_img_i), img)
    
    print(display_img_i)
    display_img_i += 1

# -----------------------------------------------------------------------------

class ConverterPipeline:
    def __init__(self, 
                 target_palette: palette.Palette,
                 size_x: int, 
                 size_y: int,
                 p: int, 
                 add_edges: bool):
        
        self.target_palette: palette.Palette = target_palette
        self.size_x: int = size_x
        self.size_y: int = size_y
        self.p:      int = p
        self.add_edges: bool = add_edges
        
        self.detail: int = int((self.size_x * self.size_y) / (p * p))
        
        self.filters: list[Callable[["ConverterPipeline", np.ndarray], np.ndarray]] = []
        
    # -------------------------------------------------------------------------
        
    def append(self, f: Callable[["ConverterPipeline", np.ndarray], np.ndarray]) -> None:
        """
        Appends a filter to the filter pipeline.
        """
        self.filters.append(f)
    
    def apply(self, img: np.ndarray) -> np.ndarray:
        """
        Sends img through the pipeline.
        """
        
        size_y, size_x, _ = img.shape
        
        if self.add_edges:
            edges = extract_edges(img)
            edges = helper_functions.resize(edges, size_x, size_y)
        
        # Apply filters
        img = prep_img(img, self.size_x, self.size_y)
        for f in self.filters:
            img = f(self, img)
        
        if self.add_edges:
            img = apply_edges(img, edges)
            
        return img


# -----------------------------------------------------------------------------
# Pre-pipeline

def extract_edges(img: np.ndarray, size_x: int, size_y: int) -> np.ndarray:
    """
    Finds the edges of an image.
    """
    
    # Find Edges
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mean = gray.mean()
    std = 1 * gray.std()
    threshold_high = mean + std
    threshold_low = mean - std
    
    edges = cv2.Canny(gray, threshold_low, threshold_high)
    
    return edges

def extract_img_palette(img: np.ndarray, K: int) -> palette.Palette:
    img = helper_functions.do_kmeans(K, img)
    
    return palette.FromImage(img)

def prep_img(img: np.ndarray, size_x: int, size_y: int) -> np.ndarray:
    img = cv2.blur(img, (3, 3))
    
    return helper_functions.resize(img, size_x, size_y)

# -----------------------------------------------------------------------------
# Post-pipeline

def apply_edges(img: np.ndarray, edges: np.ndarray) -> np.ndarray:
    assert img.shape[0] == edges.shape[0]
    assert img.shape[1] == edges.shape[1]
    
    img[edges != 0] = [0, 0, 0]
    
    return img
