import tkinter as tk
import video_app_ui as mainui
import image_processor as mainproc
import os

def main():
    # Initialize the Tkinter root window
    root = tk.Tk()

    # Create the ImageProcessor with model files and parameters
    mainprocessor = mainproc.ImageProcessor(
        object_detection_model_file="C:/Users/kondr/Downloads/Object detection project/weights/YOLOv5s.onnx",  # Object detection model
        labels_file="C:/Users/kondr/Downloads/Object detection project/weights/coco.names",  # Class labels for detection
        textdetection_model_file="C:/Users/kondr/Downloads/Object detection project/weights/text_detection_DB_TD500_resnet18_2021sep.onnx",  # Text detection model
        textrecognition_model_file="C:/Users/kondr/Downloads/Object detection project/weights/text_recognition_CRNN_EN_2021sep.onnx",  # Text recognition model
        confidence_threshold=0.8  # Confidence threshold for detection
    )



    # Set up the VideoAppUI with the root window and ImageProcessor instance
    app = mainui.VideoAppUI(root, mainprocessor)

    # Launch the Tkinter main loop
    root.mainloop()

if __name__ == "__main__":
    main()
