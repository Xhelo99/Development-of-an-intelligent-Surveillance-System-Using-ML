import numpy as np
from paddle.inference import create_predictor, PrecisionType
from paddle.inference import Config as PredictConfig
from paddleseg.deploy.infer import DeployConfig
from paddleseg.utils import logger
from paddleseg.utils.visualize import get_pseudo_color_map
import tkinter as tk
from PIL import Image, ImageTk
from tkinter import messagebox, ttk
import cv2

class Predictor:
    def __init__(self, args):
        self.args = args
        self.cfg = DeployConfig(args['cfg'])
        self._init_base_config()

        # Define label colors
        self.CITYSCAPES_COLOR_MAP = [
            (128, 64, 128),  # road
            (244, 35, 232),  # sidewalk
            (70, 70, 70),  # building
            (102, 102, 156),  # wall
            (190, 153, 153),  # fence
            (153, 153, 153),  # pole
            (250, 170, 30),  # traffic light
            (220, 220, 0),  # traffic sign
            (107, 142, 35),  # vegetation
            (152, 251, 152),  # terrain
            (70, 130, 180),  # sky
            (220, 20, 60),  # person
            (255, 0, 0),  # rider
            (0, 0, 142),  # car
            (0, 0, 70),  # truck
            (0, 60, 100),  # bus
            (0, 80, 100),  # train
            (0, 0, 230),  # motorcycle
            (119, 11, 32)  # bicycle
        ]

        if args['device'] == 'cpu':
            self._init_cpu_config()
        try:
            self.predictor = create_predictor(self.pred_cfg)
        except Exception as e:
            logger.info(str(e))
            logger.info(
                "If the above error is '(InvalidArgument) some trt inputs dynamic shape info not set, "
                "..., Expected all_dynamic_shape_set == true, ...', "
                "please set enable_auto_tune=True to use auto_tune. \n")
            exit()

    def _init_base_config(self):
        self.pred_cfg = PredictConfig(self.cfg.model, self.cfg.params)
        if not self.args['print_detail']:
            self.pred_cfg.disable_glog_info()
        self.pred_cfg.enable_memory_optim()
        self.pred_cfg.switch_ir_optim(True)

    def _init_cpu_config(self):
        logger.info("Use CPU")
        self.pred_cfg.disable_gpu()
        if self.args['enable_mkldnn']:
            logger.info("Use MKLDNN")
            self.pred_cfg.set_mkldnn_cache_capacity(10)
            self.pred_cfg.enable_mkldnn()
        self.pred_cfg.set_cpu_math_library_num_threads(self.args['cpu_threads'])


    def run(self, frame):
        input_names = self.predictor.get_input_names()
        input_handle = self.predictor.get_input_handle(input_names[0])
        output_names = self.predictor.get_output_names()
        output_handle = self.predictor.get_output_handle(output_names[0])

        # Preprocess the frame
        data = np.array([self._preprocess(frame)])

        # Run inference
        input_handle.reshape(data.shape)
        input_handle.copy_from_cpu(data)
        self.predictor.run()
        results = output_handle.copy_to_cpu()


        # Post-process the results
        results = self._postprocess(results)

        # Convert the result to a pseudo-color image
        FLATTENED_CITYSCAPES_COLOR_MAP = [color for rgb in self.CITYSCAPES_COLOR_MAP for color in rgb]
        result_image = get_pseudo_color_map(results[0], color_map=FLATTENED_CITYSCAPES_COLOR_MAP)  # Returns a PIL Image with a palette

        # Convert the PIL image to RGB mode to apply the palette correctly
        result_image = result_image.convert("RGB")

        # Convert the RGB PIL image to an OpenCV-compatible format (BGR)
        result_image = np.array(result_image)
        result_image = cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR)

        return result_image, results[0]  # Return both the image and the raw results for further processing

    def _preprocess(self, img):
        data = {}
        data['img'] = img
        return self.cfg.transforms(data)['img']

    def _postprocess(self, results):
        if self.args['with_argmax']:
            results = np.argmax(results, axis=1)
        return results
