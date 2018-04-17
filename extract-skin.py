import cv2
from constants import *
import numpy as np
from featureutils import compute_features_for_neighbourhood
from sklearn.externals import joblib


def predict_pixels(feature, model):
    print('Predicting pixels')
    X = np.array(feature)
    return model.predict(X)


def filter_with_prediction(image, prediction):
    height, width, _ = image.shape
    mask_h, mask_w, _ = image.shape

    mask_w -= 2*neighbour_radius
    mask_h -= 2*neighbour_radius

    pred_matrix = prediction.reshape((mask_h, mask_w))
    selector = pred_matrix == 'skin'
    selector = selector.astype(np.uint8)
    image_center = image[neighbour_radius:(height- neighbour_radius), neighbour_radius:(width - neighbour_radius), :]
    filtered = cv2.bitwise_and(image_center, image_center, mask=selector)

    return filtered


def extract(image, model):
    print('Extracting image')

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

    prediction = predict_pixels(features, model)

    return filter_with_prediction(image, prediction)


def load_image(image_path):
    img = cv2.imread(image_path)
    return img


def load_model(model_path):
    return joblib.load(model_path)


def show_image(skin):
    cv2.imshow('selection', skin)

    while(1):
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break

    cv2.destroyAllWindows()


def extract_skin(image_path, model_path):
    img = load_image(image_path)
    model = load_model(model_path)

    skin = extract(img, model)
    show_image(skin)


extract_skin('../examples/gente.jpg', './results/skin_model.pk1')
