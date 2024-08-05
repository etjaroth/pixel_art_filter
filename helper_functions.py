import cv2
import numpy as np
# -----------------------------------------------------------------------------

def resize(img, x, y):
    """
    Resizes an image (with interpolation).
    """
    return cv2.resize(img, (x, y))

def resize_no_interpolation(img, x, y):
    """
    Resizes an image (without interpolation).
    """
    return cv2.resize(img, (y, x), interpolation=cv2.INTER_NEAREST_EXACT)

def do_kmeans(K: int, image: np.ndarray, use_position: bool = False, use_gradient = False):
    """
    Applies K-means to the image's colors. Optionally accounts for the 
    image's position and/or the orientation of the image's gradient.
    """
    
    # Get image dimensions
    height, width, _ = image.shape

    # Reshape the image to be a list of pixels
    pixel_values = image.reshape((-1, 3))
    pixel_values = np.float32(pixel_values)
    
    # Start hstack
    stack: list[np.ndarray] = [pixel_values]

    # Find pixel locations
    if use_position:
        pixel_locations = np.indices((height, width)).reshape(2, -1).T
        pixel_locations = np.float32(pixel_locations)
        
        stack.append(pixel_locations)
    
    # Find gradient
    if use_gradient:
        grad_x = cv2.Sobel(image, -1, 1, 0).reshape((-1, 3))
        grad_y = cv2.Sobel(image, -1, 0, 1).reshape((-1, 3))
        
        stack.append(grad_x)
        stack.append(grad_y)
    
    # Combine Data
    pixel_data = np.hstack(stack)

    # Define criteria, number of clusters (K) and apply K-means
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.2)
    _, labels, centers = cv2.kmeans(pixel_data, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # Discard non-color data
    centers_colors = centers[:, :3].astype(np.uint8)

    # Map the labels to the center points (quantized colors)
    quantized_image = centers_colors[labels.flatten()]
    quantized_image = quantized_image.reshape(image.shape)

    return quantized_image
