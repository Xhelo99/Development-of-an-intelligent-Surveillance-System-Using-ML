import numpy as np
import cv2


class Person:
    def __init__(self, results):
        """
        Initialize the Person class with results. Check detect_entry_exit method.
        :param results: A 2D NumPy array representing pixel-wise segmentation results for a single frame.
        """
        self.results = results  # NumPy array (height x width)
        self.edge_threshold = 64  # Threshold for detecting entry/exit
        self.entry_side = None
        self.exit_side = None
        self.last_center_x = None
        self.enter = False

    def is_person_detected(self):
        # Define the class ID for "person"
        person_class_id = 11

        # Count the number of pixels labeled as "person" in the results
        person_pixel_count = (self.results == person_class_id).sum()

        # Check if there are more than 80 pixels labeled as "person"
        return person_pixel_count > 80

    def get_bounding_box(self):
        """
        Get the bounding box around the person if detected.
        :return: A tuple (x, y, w, h) representing the bounding box, or None if no person is detected.
        """
        if not self.is_person_detected():
            return None

        # Create a binary mask where pixels corresponding to the "person" class are 1, others are 0
        mask = (self.results == 11).astype(np.uint8)

        # Find contours in the binary mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            # Find the largest contour (in case there are multiple)
            largest_contour = max(contours, key=cv2.contourArea)
            # Get bounding box for the largest contour
            x, y, w, h = cv2.boundingRect(largest_contour)
            return (x, y, w, h)
        else:
            return None

    def detect_entry_exit(self, center_x):
        if center_x is not None:
            if not self.enter:
                # Detect entry based on the initial position of center_x
                if center_x < self.edge_threshold:
                    self.entry_side = 'Enter left'
                    self.enter = True
                    return self.entry_side
                elif center_x > 640 - self.edge_threshold:
                    self.entry_side = 'Enter right'
                    self.enter = True
                    return self.entry_side
            # Update last detected center_x for tracking the exit

        elif self.enter:
            # When center_x is None and `enter` is True, it indicates an exit
            if self.last_center_x is not None:
                if self.last_center_x < self.edge_threshold:
                    self.exit_side = 'Exit left'
                    self.enter = False
                    return self.exit_side
                elif self.last_center_x > 640 - self.edge_threshold:
                    self.exit_side = 'Exit right'
                    self.enter = False
                    return self.exit_side



        self.last_center_x = center_x
        # Return the most recent detected entry or exit side
        return None

    def update(self, results):
        self.results = results


