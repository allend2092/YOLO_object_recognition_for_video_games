import cv2
import torch
import numpy as np
from PIL import ImageGrab

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

def capture_screen(region=None):
    screenshot = ImageGrab.grab(bbox=region)
    screenshot_np_array = np.array(screenshot)
    return cv2.cvtColor(screenshot_np_array, cv2.COLOR_RGB2BGR)

def main():
    region = (0, 0, 1920, 1080)  # Set the region of the screen to capture

    cv2.namedWindow('YOLOv5 Object Detection', cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty('YOLOv5 Object Detection', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    # Initialize the VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Codec
    output_path = r"C:\Users\daryl\PycharmProjects\CUDA_Programming_Examples\training_data\output.avi"
    out = cv2.VideoWriter(output_path, fourcc, 20.0, (region[2] - region[0], region[3] - region[1]))

    try:
        while True:
            screenshot = capture_screen(region)
            results = model(screenshot)

            for result in results.xyxy[0]:  # xyxy: (x1, y1, x2, y2)
                x1, y1, x2, y2, conf, cls = result
                label = f'{model.names[int(cls)]} {conf:.2f}'
                cv2.rectangle(screenshot, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(screenshot, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            cv2.imshow('YOLOv5 Object Detection', screenshot)

            # Write the frame to the video file
            out.write(screenshot)

            if cv2.waitKey(1) == ord('q'):
                break
    except KeyboardInterrupt:
        print("Interrupted by user.")
    finally:
        out.release()  # Release the video writer
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
