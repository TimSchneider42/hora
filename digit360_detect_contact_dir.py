import numpy as np

import digit360.interface.proto as d360frame
from digit360.interface.control.led import led_set_channel
from digit360.interface.digit360 import Digit360 as Digit360Device
from digit360.interface.usb.usb import get_digit360_devices

import cv2
import tqdm


def get_digit360s():
    return [Digit360(d) for d in get_digit360_devices()]


class Digit360:
    def __init__(self, device_descriptor):
        self.device_descriptor = device_descriptor
        COLORS = [
            (64, 0, 0),
            (0, 64, 0),
            (0, 0, 64),
        ]
        for i in range(9):
            led_set_channel(
                Digit360Device(device_descriptor.data),
                d360frame.LightingChannel(i),
                COLORS[i % 3],
            )
        self.dev = cv2.VideoCapture(device_descriptor.ics)
        # self.dev.set(cv2.CAP_PROP_EXPOSURE, 0)
        # self.dev.set(cv2.CAP_PROP_FOCUS, 100)

    def read_img(self):
        ret, frame = self.dev.read()
        if not ret:
            raise RuntimeError("Failed to read image from Digit 360 device.")
        return (
            cv2.cvtColor(
                cv2.resize(frame, (frame.shape[1] // 4, frame.shape[0] // 4)),
                cv2.COLOR_BGR2RGB,
            ).astype(np.float32)
            / 255.0
        )

    def close(self):
        self.dev.release()


connected_devices = get_digit360s()
try:
    shape = connected_devices[0].read_img().shape
    imgs = np.zeros((1000,) + shape, dtype=np.float32)
    for i in tqdm.tqdm(range(100)):
        imgs[i] = connected_devices[0].read_img()

    i = 0
    while True:
        i += 1
        m = 1
        i = i % (imgs.shape[0] * m)
        img = connected_devices[0].read_img()
        if i % m == 0:
            imgs[i // m] = img
        img_norm = (img - np.mean(imgs, axis=0)) / np.maximum(
            np.std(imgs, axis=0), 0.01
        )
        img_norm = np.maximum(np.abs(img_norm).max(axis=-1), 1.5) - 1.5
        img_sum = img_norm.sum()
        coords = np.meshgrid(np.arange(img_norm.shape[1]), np.arange(img_norm.shape[0]))
        coords = np.stack(coords, axis=-1)
        coord_weight = img_norm / img_sum
        change_coord = np.round(
            (coords * coord_weight[..., None]).sum(axis=(0, 1))
        ).astype(np.int32)
        img_show = img_norm.copy() * 0.1
        img_show[
            change_coord[1] - 5 : change_coord[1] + 5,
            change_coord[0] - 5 : change_coord[0] + 5,
        ] = 1.0
        cv2.imshow("Digit 360", cv2.cvtColor(img_show, cv2.COLOR_RGB2BGR))
        if cv2.waitKey(1) == ord("q"):
            break


finally:
    for dev in connected_devices:
        dev.close()
