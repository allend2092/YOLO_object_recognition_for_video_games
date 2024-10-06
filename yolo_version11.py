import cv2
import numpy as np
import mss
from screeninfo import get_monitors
from ultralytics import YOLO
import pyautogui
import time
import threading
import logging
import queue  # For thread-safe communication

# Configure logging
logging.basicConfig(level=logging.INFO, filename='game_ai.log', filemode='w',
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Load YOLO11 model
model = YOLO('yolo11l.pt')  # Ensure you have the correct YOLO11 model file

# Create a thread-safe queue for sharing detection results
detection_queue = queue.Queue()

def capture_screen(region=None):
    with mss.mss() as sct:
        monitor = {"top": region[1], "left": region[0], "width": region[2], "height": region[3]}
        img = np.array(sct.grab(monitor))
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

def move_toward_person(box):
    x1, y1, x2, y2 = box
    center_x = (x1 + x2) // 2
    center_y = (y1 + y2) // 2
    screen_center_x = 640 // 2  # Adjusted to match the capture region width
    screen_center_y = 450 // 2  # Adjusted to match the capture region height

    # Calculate the distance to move the mouse
    move_x = center_x - screen_center_x
    move_y = center_y - screen_center_y

    # Adjust sensitivity
    sensitivity = 0.5
    move_x *= sensitivity
    move_y *= sensitivity

    # Move the mouse
    pyautogui.moveRel(move_x, move_y, duration=0.1)

def press_key(key, duration=0.2):
    pyautogui.keyDown(key)
    time.sleep(duration)
    pyautogui.keyUp(key)

def main_loop():
    region = (0, 30, 640, 450)  # Adjust the region as per your screen resolution

    # Find the primary monitor or adjust as necessary
    monitors = get_monitors()
    primary_monitor = monitors[0]

    cv2.namedWindow('YOLO11 Object Detection', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('YOLO11 Object Detection', 640, 450)
    cv2.moveWindow('YOLO11 Object Detection', primary_monitor.x, primary_monitor.y)

    try:
        while True:
            start_time = time.time()
            screenshot = capture_screen(region)
            results = model(screenshot, imgsz=640)

            # Put the detection results into the queue for the control loop
            detection_queue.put(results)

            # Visualization
            for result in results:
                if result.boxes is not None and len(result.boxes) > 0:
                    for box in result.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                        conf = box.conf[0].item()
                        cls = int(box.cls[0].item())
                        label = f'{model.names[cls]} {conf:.2f}'

                        cv2.rectangle(screenshot, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(screenshot, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.6, (0, 255, 0), 2)

            cv2.imshow('YOLO11 Object Detection', screenshot)

            if cv2.waitKey(1) == ord('q'):
                # Signal the control loop to exit
                detection_queue.put(None)
                break

            # Calculate FPS
            end_time = time.time()
            fps = 1 / (end_time - start_time)
            logging.info(f'FPS: {fps:.2f}')
    except Exception as e:
        logging.error(f"An error occurred in main_loop: {e}")
    finally:
        cv2.destroyAllWindows()

def control_loop():
    try:
        while True:
            # Wait for detection results from the main loop
            results = detection_queue.get()

            # If None is received, it's time to exit the loop
            if results is None:
                break

            for result in results:
                if result.boxes is not None and len(result.boxes) > 0:
                    for box in result.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                        conf = box.conf[0].item()
                        cls = int(box.cls[0].item())
                        label = f'{model.names[cls]} {conf:.2f}'

                        # if model.names[cls] == 'person' and conf >= 0.90:
                        #     logging.info(f"Detected person at coordinates: x1={x1}, y1={y1}, x2={x2}, y2={y2}, confidence={conf:.2f}")
                        #     move_toward_person((x1, y1, x2, y2))
                        #     press_key('w')  # Move forward

            # Indicate that the task is done
            detection_queue.task_done()
    except Exception as e:
        logging.error(f"An error occurred in control_loop: {e}")

if __name__ == "__main__":
    # Start the main_loop and control_loop in separate threads
    main_thread = threading.Thread(target=main_loop)
    control_thread = threading.Thread(target=control_loop)

    main_thread.start()
    control_thread.start()

    # Wait for both threads to complete
    main_thread.join()
    control_thread.join()
