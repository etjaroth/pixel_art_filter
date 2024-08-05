import cv2
import numpy as np

from palette import *
import remap_palette

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
    criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 100, 1.0)
    ret, label, center = cv2.kmeans(f, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    kmeansImg = center.astype(np.uint8)[label.flatten()].reshape((inputImg.shape))

    return kmeansImg

def do_kmeans_complex(K: int, image: np.ndarray):
    """
    Reduces the image to K colours.
    """
    # Get image dimensions
    height, width, _ = image.shape

    # Reshape the image to be a list of pixels
    pixel_values = image.reshape((-1, 3))
    pixel_values = np.float32(pixel_values)

    # Find pixel locations
    pixel_locations = np.indices((height, width)).reshape(2, -1).T
    pixel_locations = np.float32(pixel_locations)
    
    # Find gradient
    grad_x = cv2.Sobel(image, -1, 1, 0).reshape((-1, 3))
    grad_y = cv2.Sobel(image, -1, 0, 1).reshape((-1, 3))
    
    # Combine Data
    pixel_data = np.hstack((pixel_values, pixel_locations, grad_x, grad_y))

    # Define criteria, number of clusters (K) and apply K-means
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.2)
    _, labels, centers = cv2.kmeans(pixel_data, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # Discard non-color data
    centers_colors = centers[:, :3].astype(np.uint8)

    # Map the labels to the center points (quantized colors)
    quantized_image = centers_colors[labels.flatten()]
    quantized_image = quantized_image.reshape(image.shape)

    return quantized_image

# -----------------------------------------------------------------------------

def color_distance(c1: PaletteColor, c2: PaletteColor):
    return np.linalg.norm(abs(c1.lab_color - c2.lab_color))

# def remap_palette(p_in: Palette, p_out: Palette):
#     """
#     Remaps p1 onto p2.
#     """

#     order: dict[str, list] = {}
#     for name_in, color_in in p_in.palette.items():
#         c_in = color_in.tuple
#         order[c_in] = []

#         for name_out, color_out in p_out.palette.items():
#             order[c_in].append((color_out,
#                                 color_distance(color_in, color_out)))

#         order[c_in].sort(key=lambda x: x[1])

#     mapping: dict[str, PaletteColor] = {}
#     for color_in_tuple in order:
#         color_out: tuple[PaletteColor, float] = order[color_in_tuple][0]
#         mapping[color_in_tuple] = color_out[0]

#     return mapping

def recolor_image(img: np.ndarray, mapping: dict[str, PaletteColor]) -> np.ndarray:
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            img_color: PaletteColor = PaletteColor(
                img[y, x, 2],
                img[y, x, 1],
                img[y, x, 0])
            img[y, x] = mapping[img_color.tuple].color


# -----------------------------------------------------------------------------

size:         int = 128
p:            int =  16
detail: int = int((size * size) / (p * p))
display_size: int = 512

# -----------------------------------------------------------------------------

display_img_i = 1
def display_img(img: np.ndarray):
    global display_img_i
    
    img = img.copy()
    img = resize_no_interpolation(img, display_size, display_size)
    cv2.imshow("Image " + str(display_img_i), img)
    
    print(display_img_i)
    display_img_i += 1

# -----------------------------------------------------------------------------

if __name__ == "__main__":
    # Select Palette
    target_palette: Palette = FromFilepath("palettes/duel.gpl")
    target_palette: Palette = FromFilepath("palettes/pico-8.gpl")
    target_palette: Palette = FromFilepath("palettes/resurrect-64.gpl")

    # Select Input Image
    inputImg = cv2.imread(cv2.samples.findFile("images/landscape1.jpg"))
    inputImg = cv2.imread(cv2.samples.findFile("images/StarryNight.jpg"))
    inputImg = cv2.imread(cv2.samples.findFile("images/The_million_march_man.jpg"))
    # inputImg = cv2.imread(cv2.samples.findFile("images/tiger.jpg"))

    # Process Image
    inputImg = cv2.blur(inputImg, (3, 3))

    smallImg = resize(inputImg, size, size)
    display_img(smallImg)
    
    # Find Edges
    gray = cv2.cvtColor(smallImg, cv2.COLOR_BGR2GRAY)
    mean = gray.mean()
    std = 1 * gray.std()
    threshold_high = mean + std
    threshold_low = mean - std
    edges = cv2.Canny(smallImg, threshold_low, threshold_high)
    edges = resize(edges, size, size)
    display_img(edges)
    
    # Simplify image
    print("Detail:", detail)
    simpleImg = do_kmeans_complex(detail, smallImg)
    simpleImg = cv2.medianBlur(simpleImg, 3)
    display_img(simpleImg)

    # Quantize Colors
    quantizedImg = do_kmeans(len(target_palette), simpleImg)
    display_img(quantizedImg)

    remappedImg = remap_palette.remap_image(quantizedImg, target_palette)
    display_img(remappedImg)
    
    # Add edges
    outputImg = remappedImg.copy()
    outputImg[edges != 0] = [0, 0, 0]
    display_img(outputImg)

    cv2.waitKey(0)
