import cv2
from constants import *
import numpy as np
from featureutils import compute_features_for_neighbourhood, extract_features
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

    features = extract_features(image)
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

def extract_skins(image_paths, model_path):
    for path in image_paths:
        extract_skin(path, model_path)

