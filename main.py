import cv2
import numpy as np
import pathlib

class Palette:
    @staticmethod
    def FromImage(img) -> "Palette":
        palette = {}
        for y in range(img.shape[0]):
            for x in range(img.shape[1]):
                r = img[y, x, 2]
                g = img[y, x, 1]
                b = img[y, x, 0]
                
                name: str = hex(r)[2:] + hex(g)[2:] + hex(b)[2:]
                name = name.capitalize()
                
                if name not in palette:
                    a = np.asarray([[[img[y, x, 0], 
                                     img[y, x, 1],
                                     img[y, x, 2]]]],
                                   dtype=np.uint8)
                    palette[name] = cv2.cvtColor(a, cv2.COLOR_BGR2LAB)
                    
                
        return Palette(palette)
    
    @staticmethod
    def FromFilepath(filepath: pathlib.Path) -> "Palette":
        palette: dict[str, np.ndarray] = {}
        
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
            
            color_name = ' '.join(parts[3:]) if len(parts) == 3 else parts[3]
            a = np.asarray([[[b, g, r]]], dtype=np.uint8)
            palette[color_name] = cv2.cvtColor(a, cv2.COLOR_BGR2LAB)
            
        return Palette(palette)
            
    def __init__(self, palette: dict[str, np.ndarray]):
        self.palette: dict[str, np.ndarray] = palette
        
    def get_color(self, name: str) -> np.ndarray:
        return cv2.cvtColor(self.palette[name], cv2.COLOR_LAB2BGR)
    
    def __len__(self):
        return len(self.palette)

# -----------------------------------------------------------------------------

def resize(img, x, y):
    return cv2.resize(img, (x, y))
    
def resize_no_interpolation(img, x, y):
    return cv2.resize(img, (x, y), interpolation=cv2.INTER_NEAREST_EXACT)

def do_kmeans(K: int, inputImg):
    """
    Reduces the image to K colours.
    """
    f = inputImg.reshape((-1, 3)).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 10, 1.0)
    ret, label, center = cv2.kmeans(f, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    kmeansImg = center.astype(np.uint8)[label.flatten()].reshape((inputImg.shape))
    
    return kmeansImg

# -----------------------------------------------------------------------------

def color_distance(c1: np.ndarray, c2: np.ndarray):
    return np.linalg.norm(abs(c1 - c2))

def remap_palette(p_in: Palette, p_out: Palette):
    """
    Remaps p1 onto p2.
    """
    
    order: dict[str, list] = {}
    for name_in, color_in in p_in.palette.items():
        c_in = cv2.cvtColor(color_in, cv2.COLOR_LAB2BGR)
        c_in = tuple(c_in[0, 0].tolist())
        order[c_in] = []
        
        for name_out, color_out in p_out.palette.items():
            order[c_in].append([cv2.cvtColor(color_out, cv2.COLOR_LAB2BGR), 
                                color_distance(color_in, color_out)])
            
        order[c_in].sort(key=lambda x: x[1])
        
    mapping = {}
    for color_in in order:
        color_out = order[color_in][0]
        mapping[color_in] = color_out[0]
                
    return mapping
        
def recolor_image(img: np.ndarray, mapping: dict[np.ndarray, np.ndarray]) -> np.ndarray:
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            img[y, x] = mapping[tuple(img[y, x].tolist())]
            
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    palette: Palette = Palette.FromFilepath("palettes/pico-8.gpl")
    
    inputImg = cv2.imread(cv2.samples.findFile("images/StarryNight.jpg"))
    inputImg = cv2.imread(cv2.samples.findFile("images/The_million_march_man.jpg"))
    # inputImg = cv2.imread(cv2.samples.findFile("images/tiger.jpg"))
    
    size = 128
    smallImg = resize(inputImg, size, size)
    
    K = len(palette)
    display_size = 512
    
    kmeansImgs = []
    for i in range(K):
        kmeansImg = resize_no_interpolation(do_kmeans(i + 1, smallImg), display_size, display_size)
        kmeansImgs.append(kmeansImg)
        
    for i in range(len(kmeansImgs)):
        print(f"Image {i + 1} of {len(kmeansImgs)}")

        imgPalette = Palette.FromImage(kmeansImgs[i])
        print(len(imgPalette))
        
        newPalette = remap_palette(imgPalette, palette)
        imgOut = recolor_image(kmeansImgs[i], newPalette)

        cv2.imshow("Output Image", kmeansImgs[i])
        cv2.waitKey(0)
