import cv2
import tkinter as tk
from PIL import Image, ImageTk
from tkinter import messagebox, ttk
import time
from predictor import Predictor
from send_message import Database
from person_detect import Person
import threading


class SurveillanceSystemGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Intelligent Surveillance System 🎦")
        self.root.geometry("1366x768")
        self.root.configure(bg="#2c3e50")

        self.cap = None
        self.predictor = None

        # Initialize the person object
        self.person = Person(None)

        self.db_entry = Database()

        #Tracking personv presence duration
        self.person_detected_time = None  # Start time when a person is detected

        # Load the placeholder image
        self.load_placeholder_image()

        # Set up the layout of the GUI
        self.create_widgets()

    def create_widgets(self):
        # Main Frame
        main_frame = tk.Frame(self.root, bg="#34495e", padx=10, pady=10)
        main_frame.pack(expand=True, fill="both", padx=20, pady=20)

        # Camera Display
        self.camera_frame = tk.Label(main_frame, bg="black", bd=2, relief="groove")
        self.camera_frame.grid(row=0, column=0, columnspan=2, padx=10, pady=10)

        # Info Text Box
        self.info_text_box = tk.Text(main_frame, height=15, width=50, font=("Arial", 12), fg="white", bg="#34495e", state=tk.DISABLED)
        # Insert the initial text message "Information Box"
        self.info_text_box.config(state=tk.NORMAL)
        self.info_text_box.insert(tk.END, "Information Box: ")
        self.info_text_box.config(state=tk.DISABLED)
        self.info_text_box.grid(row=0, column=2, padx=10, pady=10, sticky="w")

        # Control Buttons
        button_frame = tk.Frame(main_frame, bg="#34495e")
        button_frame.grid(row=1, column=0, columnspan=3, pady=20)

        self.start_button = ttk.Button(button_frame, text="Start Camera", command=self.start_camera, style="TButton")
        self.start_button.grid(row=0, column=0, padx=10)

        self.stop_button = ttk.Button(button_frame, text="Stop Camera", command=self.stop_camera, style="TButton")
        self.stop_button.grid(row=0, column=1, padx=10)

        # Configure styles
        style = ttk.Style()
        style.configure("TButton", font=("Arial", 12), padding=10, background="#1abc9c")
        style.map("TButton", background=[('active', '#16a085')])

        # Add the class legend
        self.create_class_legend(main_frame)

        self.show_placeholder()

    def load_placeholder_image(self):
        # Load the placeholder image
        placeholder_image_path = (
        "D:\\ObjectTracking\\pythonProject1\\.venv\\images\\placeholder.webp")
        self.placeholder_image = Image.open(placeholder_image_path)
        self.placeholder_image = self.placeholder_image.resize((600, 400), Image.Resampling.LANCZOS)
        return self.placeholder_image

    def create_class_legend(self, parent_frame):
        # Define class-color mapping for the legend using CITYSCAPES_COLOR_MAP
        class_color_mapping = {
            "Road": "#804080",  # (128, 64, 128)
            "Sidewalk": "#f423e8",  # (244, 35, 232)
            "Building": "#464646",  # (70, 70, 70)
            "Wall": "#66669c",  # (102, 102, 156)
            "Fence": "#be9999",  # (190, 153, 153)
            "Pole": "#999999",  # (153, 153, 153)
            "Traffic Light": "#faa61e",  # (250, 170, 30)
            "Traffic Sign": "#dcdc00",  # (220, 220, 0)
            "Vegetation": "#6b8e23",  # (107, 142, 35)
            "Terrain": "#98fb98",  # (152, 251, 152)
            "Sky": "#4682b4",  # (70, 130, 180)
            "Person": "#dc143c",  # (220, 20, 60)
            "Rider": "#ff0000",  # (255, 0, 0)
            "Car": "#00008e",  # (0, 0, 142)
            "Truck": "#000046",  # (0, 0, 70)
            "Bus": "#003c64",  # (0, 60, 100)
            "Train": "#005064",  # (0, 80, 100)
            "Motorcycle": "#0000e6",  # (0, 0, 230)
            "Bicycle": "#770b20"  # (119, 11, 32)
        }

        # Create a frame for the legend
        legend_frame = tk.Frame(parent_frame, bg="#34495e")
        legend_frame.grid(row=3, column=0, columnspan=3, pady=10)

        legend_label = tk.Label(legend_frame, text="Class Legend:", font=("Arial", 14), fg="white", bg="#34495e")
        legend_label.pack(side=tk.LEFT, padx=10)

        # Add the class labels and colors
        for class_name, color in class_color_mapping.items():
            color_label = tk.Label(legend_frame, text=class_name, font=("Arial", 12), fg="white", bg=color, width=7, padx=5, pady=5)
            color_label.pack(side=tk.LEFT, padx=5)


    def show_placeholder(self):
        imgtk = ImageTk.PhotoImage(image=self.placeholder_image)
        self.camera_frame.imgtk = imgtk
        self.camera_frame.configure(image=imgtk)

    def start_camera(self):
        # Initialize the camera
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("Error", "Cannot open camera")
            return

        # Initialize the predictor
        args = {
            "cfg": "D:\\ObjectTracking\\pythonProject1\\.venv\\model\\deploy.yaml",
            "batch_size": 1,
            "device": "cpu",
            "use_trt": False,
            "precision": "uint8",
            "min_subgraph_size": 3,
            "cpu_threads": 10,
            "enable_mkldnn": False,
            "with_argmax": False,
            "print_detail": True
        }
        self.predictor = Predictor(args)
        self.show_feed()



    def show_feed(self):

        self.lock = threading.Lock()

        with self.lock:
            ret, frame = self.cap.read()

        if ret:
            frame = cv2.flip(frame, 1)

            # Run inference on each frame
            result_frame, results = self.predictor.run(frame)

            self.person.update(results)

            if self.person.is_person_detected():
                if self.person_detected_time is None:
                    self.person_detected_time = time.strftime("%m/%d/%Y %I:%M:%S %p") # Record the detection start time
                bbox = self.person.get_bounding_box()
                if bbox:
                    x, y, w, h = bbox
                    print(bbox)
                    # Calculate center_x when a bounding box is detected
                    center_x = x + w // 2
                    print("Center_x: ", center_x)
                    cv2.rectangle(result_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                else:
                    # Set center_x to None when no bounding box is detected
                    center_x = None
            else:
                # No person detected, set center_x to None
                center_x = None

            # Entry/Exit detection based on center_x
            entry_exit_status = self.person.detect_entry_exit(center_x)
            if entry_exit_status is not None:
                self.update_info_box(f"\nPerson detected. {entry_exit_status}")
                self.db_entry.add_message(self.person_detected_time, entry_exit_status)
                self.person_detected_time = None


            # Convert frame for Tkinter display
            cv2image = cv2.cvtColor(result_frame, cv2.COLOR_BGR2RGBA)
            img = Image.fromarray(cv2image)
            imgtk = ImageTk.PhotoImage(image=img)

            # Update GUI frame
            self.camera_frame.imgtk = imgtk
            self.camera_frame.configure(image=imgtk)

        # Update feed every 5ms
        if self.cap.isOpened():
            self.root.after(5, self.show_feed)

    def stop_camera(self):
        if self.cap:
            self.cap.release()
            self.cap = None
            self.show_placeholder()  # Show the placeholder image when the camera is stopped

            # Close the OpenCV windows
            cv2.destroyAllWindows()

    def update_info_box(self, message):
        # Enable the text box so we can update it
        self.info_text_box.config(state=tk.NORMAL)

        # Insert the new message
        self.info_text_box.insert(tk.END, message)

        # Disable the text box to prevent user interaction
        self.info_text_box.config(state=tk.DISABLED)


if __name__ == "__main__":
    root = tk.Tk()
    app = SurveillanceSystemGUI(root)
    root.mainloop()
