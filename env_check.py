import sys
print(f"Python version: {sys.version}")
try:
    import numpy
    print(f"Numpy version: {numpy.__version__}")
except Exception as e:
    print(f"Numpy import failed: {e}")

try:
    import cv2
    print(f"OpenCV version: {cv2.__version__}")
except Exception as e:
    print(f"OpenCV import failed: {e}")

try:
    from ultralytics import YOLO
    print("YOLO imported")
except Exception as e:
    print(f"YOLO import failed: {e}")
