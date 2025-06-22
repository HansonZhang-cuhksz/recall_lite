import pyautogui
import datetime
import time
import os

from utils import memory_dir, max_file
import shared

def take_screenshot(save_path=None):
    global memory_dir
    if save_path is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(memory_dir, "images", f"{timestamp}.png")
    screenshot = pyautogui.screenshot()
    screenshot.save(save_path)
    return save_path

def capture_task():
    global max_file

    paths = []
    
    while True:
        start_time = time.time()

        shared.pth = take_screenshot()
        paths.append(shared.pth)

        if len(paths) > max_file:
            oldest_file = paths.pop(0)
            try:
                os.remove(oldest_file)
            except Exception as e:
                print(f"Error removing file {oldest_file}: {e}")
    
        time.sleep(max(0, shared.period - (time.time() - start_time)))
        print(shared.period)