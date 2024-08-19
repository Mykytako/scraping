import cv2
import numpy as np

class NumberPlateRecognizor:
    # Constants for the model and image processing
    __MODEL_SHAPE = (736, 736)  # Model input shape (width, height)
    __BACKEND_ID = cv2.dnn.DNN_BACKEND_OPENCV
    __TARGET_ID = cv2.dnn.DNN_TARGET_CPU

    def __init__(self, textdetection_model_file, textrecognition_model_file):
        """
        Initialize the NumberPlateRecognizor object.

        Args:
            textdetection_model_file (str): Path to the text detection model.
            textrecognition_model_file (str): Path to the text recognition model.

        Raises:
            FileNotFoundError: If any of the provided file paths do not exist.
        """
        # Initialize the text detection model
        self.__detector_model = self.__initialize_textdetector_model(textdetection_model_file)

        # Initialize CRNN for text recognition
        self.__recognizer, self.__character_set, self.__character_size, self.__vertex_coordinates = self.__initialize_textrecognition_model(textrecognition_model_file)

    def __initialize_textdetector_model(self, model_path):
        # Constants for text detection parameters
        binary_threshold = 0.3
        polygon_threshold = 0.5
        max_candidates = 200
        unclip_ratio = 2.0

        # Create a text detection model
        model = cv2.dnn_TextDetectionModel_DB(cv2.dnn.readNet(model_path))
        model.setPreferableBackend(self.__BACKEND_ID)
        model.setPreferableTarget(self.__TARGET_ID)
        model.setBinaryThreshold(binary_threshold)
        model.setPolygonThreshold(polygon_threshold)
        model.setUnclipRatio(unclip_ratio)
        model.setMaxCandidates(max_candidates)
        model.setInputParams(1.0 / 255.0, self.__MODEL_SHAPE, (122.67891434, 116.66876762, 104.00698793))
        
        return model

    def __initialize_textrecognition_model(self, model_path):
        # Create a text recognition model
        model = cv2.dnn.readNet(model_path)
        model.setPreferableBackend(self.__BACKEND_ID)
        model.setPreferableTarget(self.__TARGET_ID)

        # Define character set and size
        character_set = '0123456789abcdefghijklmnopqrstuvwxyz'
        character_size = (100, 32)  # Must be in sync with the next line
        vertex_coordinates = np.array([[0, 31], [0, 0], [99, 0], [99, 31]], dtype=np.float32)

        return model, character_set, character_size, vertex_coordinates

    def __recognize_text(self, image, boxshape):
        # Preprocess the image
        vertices = boxshape.reshape((4, 2)).astype(np.float32)
        rotation_matrix = cv2.getPerspectiveTransform(vertices, self.__vertex_coordinates)
        cropped_image = cv2.warpPerspective(image, rotation_matrix, self.__character_size)
        cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
        text_blob = cv2.dnn.blobFromImage(cropped_image, size=self.__character_size, mean=127.5, scalefactor=1 / 127.5)

        # Forward pass for text recognition
        self.__recognizer.setInput(text_blob)
        output_blob = self.__recognizer.forward()

        # Postprocess the recognized text
        text = ''.join(self.__character_set[np.argmax(char)] if np.argmax(char) != 0 else '-' for char in output_blob)
        char_list = [text[i] for i in range(len(text)) if text[i] != '-' and (i == 0 or text[i] != text[i - 1])]
        
        return ''.join(char_list)

    def __visualize(self, image, boxes, texts):
        # Visualize recognized text on the image
        color = (255, 255, 255)
        isClosed = True
        thickness = 2
        output = cv2.polylines(image, np.array(boxes[0]), isClosed, color, thickness)
        
        for box, text in zip(boxes[0], texts):
            cv2.putText(output, text, tuple(box[1].astype(np.int32)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color)
        
        return output

    def detect_numberplate(self, original_image):
        """
        Detect number plates in an input image.

        Args:
            original_image (numpy.ndarray): An OpenCV image object.

        Returns:
            numpy.ndarray: The image with number plates marked.

        Raises:
            ValueError: If the provided image is not a valid numpy.ndarray.
        """
        try:
            # Ensure the input is a valid numpy.ndarray
            if not isinstance(original_image, np.ndarray):
                raise ValueError("Input image is not a valid numpy.ndarray.")

            # Resize the image to the model's input shape
            original_h, original_w, _ = original_image.shape
            scale_height = original_h / self.__MODEL_SHAPE[1]
            scale_width = original_w / self.__MODEL_SHAPE[0]
            image = cv2.resize(original_image, self.__MODEL_SHAPE)

            # Detect text locations in the resized image
            results = self.__detector_model.detect(image)

            # Recognize text in the detected locations
            texts = [self.__recognize_text(image, box.reshape(8)) for box, score in zip(results[0], results[1])]

            # Scale the result bounding boxes back to the original image dimensions
            for i, box in enumerate(results[0]):
                for j in range(4):
                    results[0][i][j][0] *= scale_width
                    results[0][i][j][1] *= scale_height

            # Draw results on the original input image
            return self.__visualize(original_image, results, texts)
        
        except Exception as e:
            print(f"Error in detect_numberplate: {str(e)}")
            return original_image  # Return the original image in case of an error
