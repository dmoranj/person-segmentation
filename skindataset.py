import cv2
import numpy as np
import pandas as pd
from constants import *
from featureutils import compute_features_for_neighbourhood, extract_filename
import os

drawing = False
ix, iy = -1, -1


def draw_roi(state):
    def handler(event, x, y, flags, param):
        global drawing, ix, iy

        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            ix, iy = x, y

        elif event == cv2.EVENT_MOUSEMOVE:
            if drawing:
                state['overlay'] = np.zeros(state['img'].shape)
                cv2.rectangle(state['overlay'], (ix, iy), (x, y), (0, 255, 0), 3)

        elif event == cv2.EVENT_LBUTTONUP:
            state['overlay'] = np.zeros(state['img'].shape)
            cv2.rectangle(state['target'], (ix, iy), (x, y), (0, 255, 0), -1)
            drawing = False

    return handler


def init_state(img, output_path):
    target = np.zeros(img.shape)
    overlay = np.zeros(img.shape)
    cv2.namedWindow('selection')

    state = {
        "img": img,
        "target": target,
        "overlay": overlay,
        "state": 'selection',
        "output_path": output_path,
        "type": 'skin'
    }

    cv2.setMouseCallback('selection', draw_roi(state))

    return state


def show_images(state):
    if state['state'] == 'selection':
        combined = (np.maximum(state['img'], state['overlay'])).astype(np.uint8)
        combined = (np.maximum(combined, state['target'])).astype(np.uint8)
        cv2.imshow('selection', combined)
    else:
        cv2.imshow('selection', state['capture'])


def capture(state):
    print('Capturing data...\n')
    mask = cv2.cvtColor(state['target'].astype(np.uint8), cv2.COLOR_BGR2GRAY)
    capture = cv2.bitwise_and(state['img'], state['img'], mask=mask)
    state['capture'] = capture.astype(np.uint8)
    state['state'] = 'save'
    return state


def extract_features(image, target):
    features = []

    rows, columns, channels = target.shape
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    target = cv2.cvtColor(target.astype(np.uint8), cv2.COLOR_BGR2GRAY)

    for i in range(neighbour_radius, rows - neighbour_radius):
        for j in range(neighbour_radius, columns - neighbour_radius):
            if target[i, j] > 0:
                rgb_neighbour = image[(i - neighbour_radius):(i + neighbour_radius + 1), (j - neighbour_radius):(j + neighbour_radius + 1), :]
                hsv_neighbour = hsv_image[(i - neighbour_radius):(i + neighbour_radius + 1), (j - neighbour_radius):(j + neighbour_radius + 1), :]

                feature = compute_features_for_neighbourhood(rgb_neighbour, hsv_neighbour)
                features.append(feature)

    return features


def save_features(path, features, type):
    cols = ["huehist_{}".format(i) for i in range(0,64)] + ["saturationhist_{}".format(i) for i in range(0,64)] + ['R_Avg', 'G_Avg', 'B_Avg']
    df = pd.DataFrame(features, columns = cols)
    df = df.assign(type=type)
    df.to_csv(path, mode='a', header=False)


def save_changes(state, type):
    print('Saving changes...\n')
    features = extract_features(state['img'], state['target'])
    save_features(state['output_path'], features, type)
    state['state'] = 'selection'
    state['target'] = np.zeros(state['img'].shape)
    return state


def wait_opencv(state):
    while(1):
        show_images(state)
        k = cv2.waitKey(1) & 0xFF

        if k == ord('c'):
            state = capture(state)
        elif k == ord('s'):
            state = save_changes(state, state['type'])

            if state['type'] == 'skin':
                state['type'] = 'other'
            else:
                break

        elif k == 27:
            break

    cv2.destroyAllWindows()


def create_target_path(output, path):
    filename = extract_filename(path)
    name = filename + ".csv"

    return os.path.join(output, name)


def process_image(path, output):
    img = cv2.imread(path)
    target_path = create_target_path(output, path)
    state = init_state(img, target_path)
    wait_opencv(state)


def process_images(path_list, output):
    for path in path_list:
        process_image(path, output)
