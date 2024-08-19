import cv2
import numpy as np

class ObjectDetector:
    def __init__(self, object_detection_model_file, class_labels_file, confidence_threshold):
        """
        Initialize the ObjectDetector object.

        Args:
            object_detection_model_file (str): Path to the object detection model.
            class_labels_file (str): Path to the class labels file.
            confidence_threshold (float): Confidence threshold for object detection.

        Raises:
            FileNotFoundError: If the provided file paths do not exist.
        """
        # Check if the provided files exist
        if not all(map(lambda f: cv2.os.path.exists(f), [object_detection_model_file, class_labels_file])):
            raise FileNotFoundError("One or more provided file paths do not exist for the object detection model.")

        # Initialize model and load class labels
        self.__model = cv2.dnn.readNet(object_detection_model_file)
        self.__classes = self.__load_labels(class_labels_file)
        self.__confidence_threshold = confidence_threshold

        # Get all the layer names of the model and filter output layers
        self.__layer_names = self.__model.getLayerNames()
        self.__output_layers = [self.__layer_names[i - 1] for i in self.__model.getUnconnectedOutLayers()]

    def __load_labels(self, labels_file):
        """
        Load class labels from the provided labels file.

        Args:
            labels_file (str): Path to the labels file.

        Returns:
            list: List of class labels.

        Raises:
            FileNotFoundError: If the labels file does not exist.
        """
        try:
            if not cv2.os.path.exists(labels_file):
                raise FileNotFoundError(f"Labels file '{labels_file}' does not exist.")

            with open(labels_file, 'r') as file:
                return [line.strip() for line in file.readlines()]
        except Exception as e:
            print(f"Error in load_labels: {str(e)}")

    def __process_detection(self, image, yolo_shape, detection):
        """
        Process a single object detection and return the bounding box.

        Args:
            image (numpy.ndarray): The input image.
            yolo_shape (tuple): The shape of the YOLO model input.
            detection (numpy.ndarray): Detection data.

        Returns:
            tuple: The top-left corner coordinates, width, and height of the bounding box.
        """
        height, width, _ = image.shape
        x_scaling = width / yolo_shape[0]
        y_scaling = height / yolo_shape[1]

        center_x, center_y, obj_width, obj_height = detection[:4]

        # Calculate rectangle coordinates
        topleft_x = int(x_scaling * (center_x - obj_width / 2))
        topleft_y = int(y_scaling * (center_y - obj_height / 2))

        obj_width = int(x_scaling * obj_width)
        obj_height = int(y_scaling * obj_height)

        return topleft_x, topleft_y, obj_width, obj_height

    def detect_objects(self, original_image):
        """
        Detect objects in an input image.

        Args:
            original_image (numpy.ndarray): An OpenCV image object.

        Returns:
            numpy.ndarray: The image with detected objects marked, or None if an error occurs.

        Raises:
            ValueError: If the provided image is not a valid numpy.ndarray.
        """
        try:
            # Ensure the image is a valid numpy.ndarray
            if not isinstance(original_image, np.ndarray):
                raise ValueError("Input image is not a valid numpy.ndarray.")

            # Prepare the image for object detection
            image = np.copy(original_image)
            blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (640, 640), (0, 0, 0), swapRB=True, crop=False)

            # Set the blob as input and perform forward pass
            self.__model.setInput(blob)
            results = self.__model.forward(self.__output_layers)

            # Initialize lists for object data
            object_classes = []
            object_confidences = []
            object_coordinates = []

            # Process each detection
            for detection in results[0][0]:
                confidence_scores = detection[5:]
                class_id = np.argmax(confidence_scores)
                confidence = confidence_scores[class_id]

                if confidence > self.__confidence_threshold:
                    bbox = self.__process_detection(image, (640, 640), detection)
                    object_coordinates.append(bbox)
                    object_confidences.append(float(confidence))
                    object_classes.append(self.__classes[class_id])

            # Apply Non-Maximum Suppression (NMS) to remove overlapping boxes
            indexes = cv2.dnn.NMSBoxes(object_coordinates, object_confidences, self.__confidence_threshold, 0.1)

            # Draw bounding boxes and labels on the original image
            for i in indexes.flatten():
                x, y, w, h = object_coordinates[i]
                label = object_classes[i]
                confidence = object_confidences[i]

                cv2.rectangle(original_image, (x, y), (x + w, y + h), (255, 255, 255), 3)
                cv2.putText(original_image, f"{label}: {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)

            return original_image
        except Exception as e:
            print(f"Error in detect_objects: {str(e)}")
            return None
