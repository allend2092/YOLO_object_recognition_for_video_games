import cv2
import numpy as np
from PIL import ImageGrab
from screeninfo import get_monitors
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO('yolov8x.pt')  # Ensure the 'yolov8x.pt' file is in the same directory as this script

def capture_screen(region=None):
    screenshot = ImageGrab.grab(bbox=region)
    screenshot_np_array = np.array(screenshot)
    return cv2.cvtColor(screenshot_np_array, cv2.COLOR_RGB2BGR)

def main():
    region = (0, 0, 1920, 1080)  # Set the region of the screen to capture

    # Find the second monitor
    monitors = get_monitors()
    if len(monitors) < 2:
        print("Second monitor not found!")
        return

    second_monitor = monitors[1]

    cv2.namedWindow('YOLOv8 Object Detection', cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty('YOLOv8 Object Detection', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    # Move the window to the second monitor
    cv2.moveWindow('YOLOv8 Object Detection', second_monitor.x, second_monitor.y)

    try:
        while True:
            screenshot = capture_screen(region)
            results = model(screenshot)

            for result in results:
                if result.boxes is not None and len(result.boxes) > 0:
                    for box in result.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                        conf = box.conf[0].item()
                        cls = int(box.cls[0].item())
                        label = f'{model.names[cls]} {conf:.2f}'
                        cv2.rectangle(screenshot, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(screenshot, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            cv2.imshow('YOLOv8 Object Detection', screenshot)

            if cv2.waitKey(1) == ord('q'):
                break
    except KeyboardInterrupt:
        print("Interrupted by user.")
    finally:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
