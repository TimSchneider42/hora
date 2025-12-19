import threading
import time
from collections import deque
from pathlib import Path

import betterproto
import numpy as np

import digit360.interface.proto as d360frame
from digit360.interface.control.led import led_set_channel
from digit360.interface.digit360 import Digit360
from digit360.interface.usb.usb import get_digit360_devices

import cv2
import tqdm
from pynput.keyboard import Key, Listener


def get_digit360s():
    return [Digit360Wrapper(d) for d in get_digit360_devices()]


class Digit360Wrapper:
    def __init__(self, device_descriptor):
        self.device_descriptor = device_descriptor
        self.dev = Digit360(device_descriptor.data)
        self.dev.tlc = 1
        COLORS = [
            (64, 0, 0),
            (0, 64, 0),
            (0, 0, 64),
        ]
        for i in range(9):
            led_set_channel(self.dev, d360frame.LightingChannel(i), COLORS[i % 3])
        self.cap = cv2.VideoCapture(device_descriptor.ics)
        self.lock = threading.Lock()
        self._img = None
        self._acc_dir = None
        self.terminate = False

        self.img_thread = threading.Thread(target=self.read_img_thread)
        self.acc_thread = threading.Thread(target=self.read_acc_thread)
        self.img_thread.start()
        self.acc_thread.start()
        self.exc = None

    def read_img_thread(self):
        try:
            while not self.terminate:
                self._img = self.read_img()
        except Exception as e:
            self.exc = e

    def read_acc_thread(self):
        try:
            while not self.terminate:
                self._acc_dir = self.read_acc_dir()
        except Exception as e:
            self.exc = e

    def read_img(self):
        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError("Failed to read image from Digit 360 device.")
        return (
            cv2.cvtColor(
                cv2.resize(frame, (frame.shape[1] // 4, frame.shape[0] // 4)),
                cv2.COLOR_BGR2RGB,
            ).astype(np.float32)
            / 255.0
        )

    def read_acc(self):
        while True:
            data = self.dev.read()
            frame_name, frame_type = betterproto.which_one_of(data, "type")
            # various streams can be access through d360frame
            if isinstance(frame_type, d360frame.ImuData):
                sensor_name, sensor_type = betterproto.which_one_of(
                    data.imu_data, "imu_type"
                )
                r = data.imu_data.raw
                if isinstance(sensor_type, d360frame.RawImuData) and r.sensor == 1:
                    return np.array([r.x, r.y, r.z], dtype=np.float32)

    def read_acc_dir(self):
        acc = self.read_acc()
        return acc / np.linalg.norm(acc)

    def close(self):
        self.terminate = True
        self.img_thread.join()
        self.acc_thread.join()
        self.cap.release()

    @property
    def img(self):
        if self.exc is not None:
            raise self.exc
        assert self._img is not None
        return self._img

    @property
    def acc_dir(self):
        if self.exc is not None:
            raise self.exc
        assert self._acc_dir is not None
        return self._acc_dir


def on_press(key):
    global state
    prev_state = state
    if not hasattr(key, "char"):
        return
    if key.char == "a":
        state = "up"
    elif key.char == "d":
        state = "down"
    if prev_state != state:
        print(f"State changed to: {state}")


def on_release(key):
    global state
    prev_state = state
    if not hasattr(key, "char"):
        return
    if key.char == "a" and state == "up":
        state = "neutral"
    elif key.char == "d" and state == "down":
        state = "neutral"
    if prev_state != state:
        print(f"State changed to: {state}")


state = "neutral"

# Collect events until released
with Listener(on_press=on_press, on_release=on_release) as listener:
    connected_devices = get_digit360s()
    try:
        data = deque()
        time.sleep(1.0)
        while len(data) < 1000:
            time.sleep(0.1)
            img = connected_devices[0].img
            acc = connected_devices[0].acc_dir
            if state != "neutral":
                if state == "up":
                    acc = np.zeros_like(acc)
                data.append((img, acc))
                if len(data) % 10 == 0:
                    print(len(data))

        imgs = np.array([i for i, _ in data])
        directions = np.array([d for _, d in data])

        output_dir = Path(__file__).parent / "digit360_data"
        output_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output_dir / f"{connected_devices[0].device_descriptor.serial}.npz",
            imgs=imgs,
            directions=directions,
        )

    finally:
        for dev in connected_devices:
            dev.close()
