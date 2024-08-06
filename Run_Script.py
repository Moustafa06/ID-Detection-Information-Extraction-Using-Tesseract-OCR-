import os
import cv2
from ultralytics import YOLO
import numpy as np
from PIL import Image, ImageEnhance
from pytesseract import *
source_dir = 'runs/detect'

def resize_process_image(image, target_size=(300, 300)):
    # Load the image

    # Apply denoising filter
    # The first parameter is the source image
    # The second parameter is the destination image
    # The third parameter is the filter strength for luminance component
    # The fourth parameter is the filter strength for color component
    # The fifth parameter is the template window size
    # The sixth parameter is the search window size
    denoised_img = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)

    # Convert the image to PIL format
    pil_img = Image.fromarray(denoised_img)

    # Create enhancer objects
    #sharpness = ImageEnhance.Sharpness(pil_img)
    #contrast = ImageEnhance.Contrast(pil_img)
    brightness = ImageEnhance.Brightness(pil_img)

    # Apply enhancement factors
    # A factor of 1.0 gives the original image
    # A higher factor gives a sharper, more contrasted, or brighter image
    # A lower factor gives a smoother, less contrasted, or darker image
    #sharp_img = sharpness.enhance(2.0)
    #contrast_img = contrast.enhance(1.5)
    bright_img = brightness.enhance(1.2)

    # Convert the image back to OpenCV format
    enhanced_img = cv2.cvtColor(np.array(bright_img), cv2.COLOR_RGB2GRAY)

    # Resize the image
    #resized_image = cv2.resize(enhanced_img, target_size)

    return enhanced_img

def extract_ara_cropped_clusters(img, max_horizontal_gap=30):
    # Perform Arabic text extraction
    ocr_result = pytesseract.image_to_string(img, lang="ara", config='--psm 6')

    # Get bounding boxes for each detected word
    boxes = pytesseract.image_to_boxes(img, lang="ara", config='--psm 6')

    # Extract coordinates and convert them to integers
    coordinates = [list(map(int, box.split()[1:5])) for box in boxes.splitlines()]

    # Sort coordinates based on the x-coordinate (left to right)
    coordinates.sort(key=lambda x: x[0])
    #for i in range(len(coordinates) - 1):
        #print(f'coordinates {i}: {coordinates[i]}')
    # Initialize clusters list
    clusters = []

    # Iterate through coordinates and group based on horizontal proximity
    current_cluster = [coordinates[0]]
    for i in range(1, len(coordinates)):
        x1, _, _, _ = coordinates[i-1]
        x2, _, _, _ = coordinates[i]

        # Check if the horizontal gap is within the specified threshold
        if x2 - x1 <= max_horizontal_gap:
            current_cluster.append(coordinates[i])
        else:
            # Start a new cluster
            clusters.append(current_cluster)
            current_cluster = [coordinates[i]]

    # Add the last cluster
    clusters.append(current_cluster)

    # Iterate over clusters and extract individual images
    cropped_clusters = []
    for cluster in clusters:
        # Get the bounding box coordinates of the cluster
        cluster_coordinates = np.array(cluster)

        # Find the minimum and maximum coordinates for the cluster
        min_x = np.min(cluster_coordinates[:, 0])
        min_y = np.min(cluster_coordinates[:, 1])
        max_x = np.max(cluster_coordinates[:, 2])
        max_y = np.max(cluster_coordinates[:, 3])

        # Crop the region of interest (word cluster)
        cropped_cluster = img[min_y:max_y, min_x:max_x]
        cropped_clusters.append(cropped_cluster)

    return ocr_result, cropped_clusters

def extract_num_cropped_image(image, language="ara_t12"):
    # Read the preprocessed image
    img = resize_process_image(image)

    # Perform OCR on the image
    custom_config = f'--psm 6 -l {language}'
    ocr_result = pytesseract.image_to_string(img, config=custom_config)

    # Get bounding boxes for each detected word
    h, w, = img.shape
    boxes = pytesseract.image_to_boxes(img, config=custom_config)

    # Extract coordinates and convert them to integers
    coordinates = [list(map(int, box.split()[1:5])) for box in boxes.splitlines()]

    # Adjust coordinates to match the cropped region in the original image
    adjusted_coordinates = [(x + int(w / 3), y + int(h / 1.8), x2 + int(w / 3), y2 + int(h / 1.8)) for (x, y, x2, y2) in coordinates]

    # Find the minimum and maximum coordinates
    min_x = min(x for x, _, _, _ in adjusted_coordinates)
    min_y = min(y for _, y, _, _ in adjusted_coordinates)
    max_x = max(x2 for _, _, x2, _ in adjusted_coordinates)
    max_y = max(y2 for _, _, _, y2 in adjusted_coordinates)

    # Crop the region of interest
    cropped_image = img[min_y+15:max_y-15, min_x+10:max_x]

    # Create a mask to blank out the cropped region
    mask = np.ones_like(img) * 255
    mask[min_y+50:max_y-30, 0:max_x] = 0

    # Apply the mask to get the remaining image
    remaining_image = cv2.bitwise_and(img, mask)

    return ocr_result, cropped_image, remaining_image


def rotate_image(image, angle):
    # Get image dimensions
    height, width = image.shape[:2]

    # Calculate the rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)

    # Apply the rotation to get the bounding box of the rotated image
    cos = np.abs(rotation_matrix[0, 0])
    sin = np.abs(rotation_matrix[0, 1])

    new_width = int((height * sin) + (width * cos))
    new_height = int((height * cos) + (width * sin))

    rotation_matrix[0, 2] += (new_width / 2) - (width / 2)
    rotation_matrix[1, 2] += (new_height / 2) - (height / 2)

    # Apply the adjusted rotation to the image
    rotated_image = cv2.warpAffine(image, rotation_matrix, (new_width, new_height), flags=cv2.INTER_LINEAR)

    return rotated_image

def get_bounding_box_corners(x_center, y_center, width, height):
    # Calculate half-width and half-height
    half_width = width / 2
    half_height = height / 2

    # Calculate coordinates of each corner
    top_left = (x_center - half_width, y_center - half_height)
    top_right = (x_center + half_width, y_center - half_height)
    bottom_right = (x_center + half_width, y_center + half_height)
    bottom_left = (x_center - half_width, y_center + half_height)
    return [top_left, top_right, bottom_right, bottom_left]

def order_points(pts):
    # Convert the list to a NumPy array
    pts = np.array(pts)

    # Initialize a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype="float32")

    # The top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # Now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    # Return the ordered coordinates
    return rect


def four_point_transform(image, pts):
    # Obtain a consistent order of the points and unpack them individually

    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # Compute the width and height of the new image
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # Define destination points
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    # Compute the perspective transform matrix and apply it
    M = cv2.getPerspectiveTransform(rect, dst)

    # Convert the input image to a NumPy array if it's not already
    if not isinstance(image, np.ndarray):
        image = np.array(image)

    # Apply the perspective transform
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    # Return the warped image
    return warped


def save_image(image, filename):
    # Ensure the 'output' directory exists
    output_dir = 'output'
    os.makedirs(output_dir, exist_ok=True)

    # Save the image to the 'output' directory
    cv2.imwrite(os.path.join(output_dir, filename), image)

def predict(image_path, model=YOLO('best.pt')):
    results = model.predict(image_path, save=True)

    # this nested loop because the YOLO model creates a folder
    # For each prediction in runs/detect directory
    # So I am accessing each image in each folder to prevent error
    for item in os.listdir(source_dir):
        item_path = os.path.join(source_dir, item)
        if os.path.isdir(item_path):
            for filename in os.listdir(item_path):
                # Read the image
                image = cv2.imread(os.path.join(item_path, filename))

                for result in results:
                    boxes = result.boxes
                x_center, y_center, width, height = boxes.xywh[0].cpu().numpy()
                corners = get_bounding_box_corners(x_center, y_center, width, height)

                # Perform alignment
                warped = four_point_transform(image, corners)
    return image, warped


if __name__ == "__main__":
    test_image_path = '3.jpg'
    img = cv2.imread(test_image_path)
    rotated_img = rotate_image(img, 45)
    save_image(rotated_img, 'rotated_image.jpg')
    # Save the detected image
    test_image_path1 = 'rotated_1.jpg'

    detected_image, aligned_image = predict(test_image_path1)
    cv2.imshow('Detected', detected_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


    #save_image(Detected_image, 'detected_image.jpg')

    # Save the aligned image
    #save_image(Aligned_image, 'aligned_image.jpg')

    # Uncomment the following lines if you want to display the detected image
    # cv2.imshow('Detected', Detected_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # Uncomment the following lines if you want to display the aligned image
    # cv2.imshow('Aligned', Aligned_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # Rotate the aligned image
    #rotated = rotate_image(Aligned_image, 45)

    # Uncomment the following lines if you want to display the rotated image
    # cv2.imshow('rotated', rotated)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # Extract cropped number image and remaining image
   # ocr_result, cropped_num_image, remaining_image = extract_num_cropped_image(Aligned_image)

    #save_image(cropped_num_image, 'cropped_number_image.jpg')

    # Extract Arabic cropped word clusters
    #ocr_result2, cropped_word_clusters = extract_ara_cropped_clusters(remaining_image)

    # Uncomment the following lines if you want to display the national id number image
    # Display the cropped numbers image
    #cv2.imshow('numbers', cropped_num_image)
    #cv2.waitKey(0)

    # Uncomment the following lines if you want to display the information image
    # Display each word cluster image separately
    #for i, cluster_image in enumerate(cropped_word_clusters):
        #cv2.imshow(f'word_cluster_{i}', cluster_image)
        #cv2.waitKey(0)

    #cv2.destroyAllWindows()
    #for i, cluster_image in enumerate(cropped_word_clusters):
        #save_image(cluster_image, f'cropped_word_cluster_{i}.jpg')