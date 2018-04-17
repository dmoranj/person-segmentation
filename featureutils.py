import numpy as np
from constants import *
import cv2


def average_neighbour(neighbour):
    return np.average(neighbour, (0, 1))/255


def create_histogram(neighbour):
    indexes = range(0, 64)
    content = [0] * 64
    histogram = dict(zip(indexes, content))
    total_values = (2*neighbour_radius + 1)**2

    for i in range(len(neighbour)):
        for j in range(len(neighbour[i])):
            value = int(np.floor(neighbour[i, j] / 4))
            histogram[value] = histogram[value] + 1

    return np.array(list(histogram.values()))/total_values


def create_features(hue_hist, satur_hist, rgb_avg):
    return list(hue_hist) + list(satur_hist) + list(rgb_avg)


def compute_features_for_neighbourhood(rgb_neighbour, hsv_neighbour):
    hue_histogram = create_histogram(hsv_neighbour[:, :, 0])
    saturation_histogram = create_histogram(hsv_neighbour[:, :, 1])
    rgb_avg = average_neighbour(rgb_neighbour)

    return create_features(hue_histogram, saturation_histogram, rgb_avg)


def extract_features(image):
    features = []
    rows, columns, channels = image.shape
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    for i in range(neighbour_radius, rows - neighbour_radius):
        if i%50 == 0:
            print('Iteration {}'.format(i))

        for j in range(neighbour_radius, columns - neighbour_radius):
            rgb_neighbour = image[(i - neighbour_radius):(i + neighbour_radius + 1), (j - neighbour_radius):(j + neighbour_radius + 1), :]
            hsv_neighbour = hsv_image[(i - neighbour_radius):(i + neighbour_radius + 1), (j - neighbour_radius):(j + neighbour_radius + 1), :]

            feature = compute_features_for_neighbourhood(rgb_neighbour, hsv_neighbour)
            features.append(feature)

    return features


def extract_filename(path):
    filename = path.split("/")[-1]
    name = filename.split('.')[0]

    return name
