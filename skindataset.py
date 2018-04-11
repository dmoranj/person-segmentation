import cv2
import numpy as np

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


def init_state(img):
    target = np.zeros(img.shape)
    overlay = np.zeros(img.shape)
    cv2.namedWindow('selection')

    state = {
        "img": img,
        "target": target,
        "overlay": overlay,
        "state": 'selection'
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
    mask = cv2.cvtColor(state['target'].astype(np.uint8), cv2.COLOR_BGR2GRAY)
    capture = cv2.bitwise_and(state['img'], state['img'], mask=mask)
    state['capture'] = capture.astype(np.uint8)
    state['state'] = 'save'
    return state


def save_changes(state):
    print('Saving')


def wait_opencv(state):

    while(1):
        show_images(state)
        k = cv2.waitKey(1) & 0xFF

        if k == ord('c'):
            state = capture(state)
        elif k == ord('s'):
            save_changes(state)
            break
        elif k == 27:
            break

    cv2.destroyAllWindows()


def process_image(path):
    img = cv2.imread(path)
    state = init_state(img)
    wait_opencv(state)


process_image('../examples/gente.jpg')