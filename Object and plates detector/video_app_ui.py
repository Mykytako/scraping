import tkinter as tk
from tkinter import messagebox
import tkinter.font as tkFont
import cv2
import numpy as np
import threading

class VideoAppUI:
    def __init__(self, root, improcessor):
        self.__initialize_widgets(root)
        self.__initialize_variables(improcessor)
        self.__lock = threading.Lock()  # Thread lock for synchronization
        self.__stop_event = threading.Event()  # Event to signal thread to stop


    def __initialize_widgets(self, root):
        self.__root = root

        # Set window title and size
        root.title("OpenCV DNN End-to-end Demo")
        window_width = 320
        window_height = 240
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        position = f'{window_width}x{window_height}+{(screen_width - window_width) // 2}+{(screen_height - window_height) // 2}'
        root.geometry(position)
        root.resizable(width=False, height=False)

        # Create Start Video button
        self.__btn_start_video = tk.Button(root, text="Start Video", bg="#f0f0f0", fg="#000000", font=tkFont.Font(family='Times', size=10), command=self.__start_video)
        self.__btn_start_video.place(x=30, y=50, width=70, height=25)

        # Create Stop Video button
        self.__btn_stop_video = tk.Button(root, text="Stop Video", bg="#f0f0f0", fg="#000000", font=tkFont.Font(family='Times', size=10), command=self.__stop_video, state=tk.DISABLED)
        self.__btn_stop_video.place(x=30, y=100, width=70, height=25)

        # Create Process Video checkbox
        self.__process_video_var = tk.BooleanVar()
        self.__chk_process_video = tk.Checkbutton(root, text="Process Video", variable=self.__process_video_var, font=tkFont.Font(family='Times', size=10), fg="#333333", command=self.__toggle_process_video)
        self.__chk_process_video.place(x=0, y=170, width=150, height=30)

        # Create Detect Objects checkbox
        self.__detect_objects_var = tk.BooleanVar()
        self.__chk_detect_objects = tk.Checkbutton(root, text="Detect Objects", variable=self.__detect_objects_var, font=tkFont.Font(family='Times', size=10), fg="#333333", state=tk.DISABLED)
        self.__chk_detect_objects.place(x=10, y=210, width=135, height=30)

        # Create Detect Number Plate checkbox
        self.__detect_number_plates_var = tk.BooleanVar()
        self.__chk_detect_numberplate = tk.Checkbutton(root, text="Detect Number Plate", variable=self.__detect_number_plates_var, font=tkFont.Font(family='Times', size=10), fg="#333333", state=tk.DISABLED)
        self.__chk_detect_numberplate.place(x=10, y=250, width=175, height=30)

    def __initialize_variables(self, improcessor):
        self.__improcessor = improcessor
        self.__video_capture = None
        self.__video_thread = None

    def __start_video(self):
        with self.__lock:  # Ensure thread-safe access
            self.__video_capture = cv2.VideoCapture(0)
            self.__btn_start_video["state"] = tk.DISABLED
            self.__btn_stop_video["state"] = tk.NORMAL
            self.__stop_event.clear()  # Clear the stop event
            self.__video_thread = threading.Thread(target=self.__process_video)
            self.__video_thread.daemon = True
            self.__video_thread.start()


    def __stop_video(self):
        with self.__lock:
            if self.__video_capture:
                self.__stop_event.set()  # Signal the thread to stop
                self.__video_thread.join()  # Wait for the thread to finish
                self.__video_capture.release()
                self.__video_capture = None

            self.__btn_start_video["state"] = tk.NORMAL
            self.__btn_stop_video["state"] = tk.DISABLED


    def __toggle_process_video(self):
        process_video = self.__process_video_var.get()
        self.__chk_detect_objects["state"] = tk.NORMAL if process_video else tk.DISABLED
        self.__chk_detect_numberplate["state"] = tk.NORMAL if process_video else tk.DISABLED

    def __process_video(self):
        if not self.__video_capture:
            messagebox.showerror("Error", "Video is not running. Start the video first.")
            return

        cv2.namedWindow("Video", cv2.WINDOW_FULLSCREEN)
        while not self.__stop_event.is_set():  # Continue until stop event is set
            ret, frame = self.__video_capture.read()
            if not ret:
                break

            process_video = self.__process_video_var.get()
            detect_objects = self.__detect_objects_var.get()
            detect_number_plates = self.__detect_number_plates_var.get()

            if process_video:
                if detect_objects:
                    frame = self.__detect_objects(frame)
                if detect_number_plates:
                    frame = self.__detect_number_plate(frame)

            cv2.imshow("Video", frame)
            cv2.waitKey(1)

        cv2.destroyAllWindows()  # Close the window once the loop exits


        cv2.namedWindow("Video", cv2.WINDOW_FULLSCREEN)
        while True:
            if self.__video_capture:
                ret, frame = self.__video_capture.read()
                if not ret:
                    break

                process_video = self.__process_video_var.get()
                detect_objects = self.__detect_objects_var.get()
                detect_number_plates = self.__detect_number_plates_var.get()

                if process_video:
                    if detect_objects:
                        frame = self.__detect_objects(frame)
                    if detect_number_plates:
                        frame = self.__detect_number_plate(frame)

                cv2.imshow("Video", frame)
                cv2.waitKey(1)

        cv2.destroyAllWindows()

    def __detect_objects(self, frame):
        return self.__improcessor.detect_objects(frame)

    def __detect_number_plate(self, frame):
        return self.__improcessor.detect_number_plate(frame)

    def run(self):
        self.__root.mainloop()
