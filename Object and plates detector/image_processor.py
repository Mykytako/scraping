import cv2
import object_detector as detector
import numberplate_recognizor as numplaterecog

class ImageProcessor:
    def __init__(self, object_detection_model_file, labels_file, textdetection_model_file, textrecognition_model_file, confidence_threshold):
        """
        Initialize the ImageProcessor object.

        Args:
            object_detection_model_file (str): Path to the object detection model file.
            labels_file (str): Path to the labels file.
            textdetection_model_file (str): Path to the text detection model file.
            textrecognition_model_file (str): Path to the text recognition model file.
            confidence_threshold (float): Confidence threshold for object detection.

        Raises:
            FileNotFoundError: If any of the provided file paths do not exist.
        """
        # Initialize object detection model
        self.__object_detection_model = detector.ObjectDetector(
            object_detection_model_file=object_detection_model_file,
            class_labels_file=labels_file,
            confidence_threshold=confidence_threshold
        )

        # Initialize number plate recognition model
        self.__numberplate_detection_model = numplaterecog.NumberPlateRecognizor(
            textdetection_model_file=textdetection_model_file,
            textrecognition_model_file=textrecognition_model_file
        )

    def detect_objects(self, image):
        """
        Detect objects in the input image.

        Args:
            image (numpy.ndarray): The image in which objects need to be detected.

        Returns:
            numpy.ndarray: The image with detected objects marked.

        Raises:
            Exception: If an error occurs during object detection.
        """
        try:
            # Perform object detection
            result_image = self.__object_detection_model.detect_objects(image)
        except Exception as e:
            print(f"Error in detect_objects: {str(e)}")
            result_image = image  # Return original image in case of error

        return result_image

    def detect_numberplate(self, image):
        """
        Detect number plates in the input image.

        Args:
            image (numpy.ndarray): The image in which number plates need to be detected.

        Returns:
            str: The recognized number plate text, or None if an error occurs.

        Raises:
            Exception: If an error occurs during number plate detection.
        """
        try:
            # Perform number plate detection
            return self.__numberplate_detection_model.detect_numberplate(image)
        except Exception as e:
            print(f"Error in detect_numberplate: {str(e)}")
            return None  # Return None in case of error
