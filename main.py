import cv2
import numpy as np
from skimage.feature import greycomatrix, greycoprops
import csv
import os


def palm_segmentation(img):
    thresholdedImg = np.zeros_like(img)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i, j] < 40:     # > 230 for 11k dataset, < 40 for second dataset
                thresholdedImg[i, j] = 0
            else:
                thresholdedImg[i, j] = img[i, j]

    kernel = np.ones((5, 5), np.uint8)
    erodedImg = cv2.erode(thresholdedImg, kernel, iterations=1)
    dilatedImg = cv2.dilate(erodedImg, kernel, iterations=1)

    contours, _ = cv2.findContours(dilatedImg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # take the largest contour (object) as palm
    filteredContour = []
    maxArea = 0
    for contour in contours:
        contour_area = cv2.contourArea(contour)
        if maxArea < contour_area:
            maxArea = contour_area
            filteredContour = [contour]

    # fill the background with black
    mask = np.zeros_like(thresholdedImg)
    cv2.drawContours(mask, filteredContour, -1, 255, thickness=cv2.FILLED)
    segmented = cv2.bitwise_and(img, img, mask=mask)

    return segmented


def get_direction_numbers(img, x, y):
    diffs = []

    # eight neighbours
    neighbour_codes = [(0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1)]

    for idx, (dx, dy) in enumerate(neighbour_codes):
        center = img[x, y]
        neighbor = img[x+dx, y+dy]
        diff = abs(int(center) - int(neighbor))
        diffs.append((diff, idx+1))

    diffs.sort(key=lambda x: x[0], reverse=True)

    m1 = diffs[0][1]
    m2 = diffs[1][1]

    return m1, m2


def extract_texture_features(img):
    texture_codes = np.zeros_like(img, dtype=np.uint8)
    height, width = img.shape

    for x in range(1, height-1):
        for y in range(1, width-1):
            m1, m2 = get_direction_numbers(img, x, y)
            texture_code = (m1 - 1) * 8 + (m2 - 1)
            texture_codes[x, y] = texture_code

    return texture_codes


def get_gradient_feature_code(img):
    kirsch_operators = [
        np.array([[5, 5, 5], [-3, 0, -3], [-3, -3, -3]]),  # north
        np.array([[5, 5, -3], [5, 0, -3], [-3, -3, -3]]),  # north-east
        np.array([[5, -3, -3], [5, 0, -3], [5, -3, -3]]),  # east
        np.array([[-3, -3, -3], [5, 0, -3], [5, 5, -3]]),  # south-east
        np.array([[-3, -3, -3], [-3, 0, -3], [5, 5, 5]]),  # south
        np.array([[-3, -3, -3], [-3, 0, 5], [-3, 5, 5]]),  # south-west
        np.array([[-3, -3, 5], [-3, 0, 5], [-3, -3, 5]]),  # west
        np.array([[-3, 5, 5], [-3, 0, 5], [-3, -3, -3]])   # north-west
    ]

    edge_responses = []
    for i in kirsch_operators:
        res = cv2.filter2D(img, -1, i)
        edge_responses.append(abs(res))

    combined_responses = np.stack(edge_responses, axis=-1)
    sorted_indices = np.argsort(-combined_responses, axis=-1)

    p1 = sorted_indices[..., 0] + 1
    p2 = sorted_indices[..., 1] + 1

    gradient_code = (p1 - 1) * 8 + (p2 - 1)

    return gradient_code


def glcm_features_extract(img):
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    glcm = greycomatrix(img, distances=[1], angles=angles, symmetric=True, normed=True)

    features = {
        'energy': greycoprops(glcm, 'energy').flatten(),
        'contrast': greycoprops(glcm, 'contrast').flatten(),
        'correlation': greycoprops(glcm, 'correlation').flatten(),
        'homogeneity': greycoprops(glcm, 'homogeneity').flatten()
    }

    return features


def calc_area(img):
    binaryImg = np.zeros_like(img)
    area = np.zeros((16, 16))

    # generate a binary image
    for i in range(256):
        for j in range(256):
            if img[i, j] == 0:
                binaryImg[i, j] = 0

            else:
                binaryImg[i, j] = 255

    # calculate the blockwise area by counting white pixels
    for i in range(0, 256, 16):
        for j in range(0, 256, 16):
            counter = 0
            for k in range(16):
                for l in range(16):
                    ii = i + k
                    jj = j + l
                    if binaryImg[ii, jj] == 255:
                        counter += 1

            ii = i // 16
            jj = j // 16
            area[ii, jj] = counter

    return area


def blockwise_histogram(img):
    height, width = 256, 256
    blockSize = 16
    hw = 256
    combined = []

    for i in range(0, height, blockSize):
        for j in range(0, width, blockSize):
            block = img[i:i+blockSize, j:j+blockSize]
            hist = np.zeros(16)
            for k in range(blockSize):
                for l in range(blockSize):
                    # consider histograms in batches of 16 intensity values
                    hist[block[k, l] // 16] += 1

            hist /= hw
            combined.extend(hist)

    return combined


textureHist_all = []
gradientHist_all = []
area_all = []
glcm_features_combined_all = []
name = []

folder_path = 'selected'
i = 0
for filename in os.listdir(folder_path):
    if filename.endswith(('.png', '.jpg', '.jpeg')):
        img_path = os.path.join(folder_path, filename)
        img = cv2.imread(img_path, 0)
        name.append(filename)

        # image after palm segmentation
        img = cv2.resize(img, (256, 256))
        segmentedImg = palm_segmentation(img)

        output_folder = 'segmented'
        img_name = f'name_{i}'
        output_path = os.path.join(output_folder, img_name + '.png')
        cv2.imwrite(output_path, segmentedImg)

        # Canny Edge detection
        edges = cv2.Canny(segmentedImg, 0, 50)

        output_folder = 'canny'
        img_name = f'name_{i}'
        output_path = os.path.join(output_folder, img_name + '.png')
        cv2.imwrite(output_path, edges)

        # texture features image
        texture_features = extract_texture_features(segmentedImg)
        output_folder = 'texture'
        img_name = f'name_{i}'
        output_path = os.path.join(output_folder, img_name + '.png')
        cv2.imwrite(output_path, texture_features)

        # gradient features image
        gradient_features = get_gradient_feature_code(segmentedImg)
        output_folder = 'gradient'
        img_name = f'name_{i}'
        output_path = os.path.join(output_folder, img_name + '.png')
        cv2.imwrite(output_path, gradient_features)

        # GLCM matrix features
        angles = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
        glcm_features = glcm_features_extract(edges)
        glcm_features_combined = []
        for feature_name, feature_values in glcm_features.items():
            for val in feature_values:
                glcm_features_combined.append(val)

        # calculate palm area into 1D array
        area = calc_area(segmentedImg)
        area = area.flatten()
        area = area.tolist()

        # calculate blockwise histograms for each feature image
        textureHist = blockwise_histogram(texture_features)
        gradientHist = blockwise_histogram(gradient_features)

        textureHist_all.append(textureHist)
        gradientHist_all.append(gradientHist)
        area_all.append(area)
        glcm_features_combined_all.append(glcm_features_combined)

        i += 1


# write extracted numerical features into a CSV file
output_file = 'data.csv'

with open(output_file, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["name", "texture", "gradient", "area", "glcm"])
    for i in range(len(textureHist_all)):
        row = [name[i], textureHist_all[i], gradientHist_all[i], area_all[i], glcm_features_combined_all[i]]
        writer.writerow(row)
