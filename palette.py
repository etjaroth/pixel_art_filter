import cv2
import numpy as np
import pathlib

# -----------------------------------------------------------------------------

class PaletteColor:
    def __init__(self, r: int, g: int, b: int):
        self.tuple = (r, g, b)
        self.name = "0x" + hex(r)[2:] + hex(g)[2:] + hex(b)[2:]
        self.color = np.asarray([[[b, g, r]]], dtype=np.uint8)
        self.lab_color = cv2.cvtColor(self.color, cv2.COLOR_BGR2LAB)
        
# -----------------------------------------------------------------------------

class Palette:
    def __init__(self, palette: dict[str, PaletteColor]):
        self.palette: dict[str, PaletteColor] = palette
        
    # -------------------------------------------------------------------------
        
    def add_color(self, color: PaletteColor) -> None:
        self.palette[color.name] = color
        
    def get_color(self, name: str) -> np.ndarray:
        return cv2.cvtColor(self.palette[name], cv2.COLOR_LAB2BGR)
    
    # -------------------------------------------------------------------------
    
    def __len__(self):
        return len(self.palette)
    
    def __contains__(self, color: PaletteColor) -> bool:
        return color.name in self.palette
    
# -----------------------------------------------------------------------------

def FromImage(img) -> "Palette":
    palette: Palette = Palette({})
    
    # Find colors in image
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            color = PaletteColor(img[y, x, 2],
                                    img[y, x, 1],
                                    img[y, x, 0])
            
            if color not in palette:
                palette.add_color(color)

    return palette

# -----------------------------------------------------------------------------
    
def FromFilepath(filepath: pathlib.Path) -> "Palette":
    palette: Palette = Palette({})
    
    with open(filepath, 'r') as file:
        lines = file.readlines()
    
    for line in lines:
        line = line.lstrip()
        
        if line[0] == '#' or line[0:4] == "GIMP":
            continue
        
        parts = line.split()
        assert len(parts) == 4
        
        r = int(parts[0])
        g = int(parts[1])
        b = int(parts[2])
        
        color: PaletteColor = PaletteColor(r, g, b)
        palette.add_color(color)
        
    return palette

# -----------------------------------------------------------------------------


