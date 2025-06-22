import pyautogui
import datetime
import os

from utils import memory_dir

def take_screenshot(save_path=None):
    global memory_dir
    if save_path is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(memory_dir, "images", f"{timestamp}.png")
    screenshot = pyautogui.screenshot()
    screenshot.save(save_path)
    return save_path