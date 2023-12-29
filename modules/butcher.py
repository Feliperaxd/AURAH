# -----------------------------------------------------------------------------
# AURAH - Analysis and Understanding of Human Attributes
# Copyright (c) 2023 Felipe Amaral dos Santos
# Licensed under the MIT License (see LICENSE file)
# -----------------------------------------------------------------------------

import cv2
import numpy as np
from pathlib import Path
from resources.ids import IDs
from resources.custom_typing import DataPack, ZoneInt
from typing import Any, Dict, List, Optional, Tuple, Union


class Butcher:
    

    def __init__(
        self: 'Butcher',
        net_config_path: Union[Path, str],
        net_weights_path: Union[Path, str]
    ) -> None:
        """
            Initializes the Butcher class with the provided neural network configuration
            and weights.

        Parameters:
            - net_config_path (Union[Path, str]): Path to the neural network
            configuration file.
            - net_weights_path (Union[Path, str]): Path to the neural network weights file.

        Attributes:
            - net_config_path (str): Path to the neural network configuration file.
            - net_weights_path (str): Path to the neural network weights file.
            - net (cv2.dnn_Net): YOLO neural network object initialized with the provided
            configuration and weights.
        """

        if isinstance(net_config_path, Path):
            self.net_config_path = str(net_config_path)
        else:
            self.net_config_path = net_config_path

        if isinstance(net_weights_path, Path):
            self.net_weights_path = str(net_weights_path)
        else:
            self.net_weights_path = net_weights_path

        self.net = cv2.dnn.readNet(
            model=self.net_weights_path, 
            config=self.net_config_path
        )
        
    def detect_in(
        self: 'Butcher',
        blob: np.ndarray
    ) -> Tuple[np.ndarray]:
        """
            Detects objects in an image represented by a NumPy array.

        Parameters:
            - self (Butcher): Instance of the 'Butcher' class.
            - blob (np.ndarray): A NumPy array representing the blob to input.

        Returns:
            A tuple containing the blob (input image representation) and outputs 
            (detection results) from the neural network, both represented by NumPy arrays.
        """

        self.net.setInput(blob)
        layer_names = self.net.getUnconnectedOutLayersNames()
        output_data = self.net.forward(layer_names)

        return output_data

    def remove_overlapping_items(
        self: 'Butcher',
        data_pack: DataPack[IDs, Tuple[ZoneInt, float]],
        nms_threshold: Optional[float] = 0.3,
        score_threshold: Optional[float] = 0.5
    ) -> DataPack[IDs, Tuple[ZoneInt, float]]:    
        """
            Filters and cleans up the contents of a 'DataPack' by removing overlapping 
            items!

        Parameters:
            - self (Butcher): Instance of the 'Butcher' class.
            - data_pack (ct.DataPack[IDs, Tuple[ZoneInt, float]]): contains tuples with 
            ZoneInt and confidence score as float.
            - nms_threshold (Optional[float]): Non-maximum suppression threshold for 
            removing overlapping items (default is 0.3).
            - score_threshold (Optional[float] = 0.5): Minimum score threshold for 
            retaining items (default is 0.5).

        Returns:
            'DataPack' with 'IDs' as key and contents as tuples with zones and confidence 
            scores as float.
        """
        
        for row in data_pack:
            all_data = data_pack[row]
            coordinates = [x[0] for x in all_data]
            scores = [x[1] for x in all_data]    

            indexes = cv2.dnn.NMSBoxes(
                bboxes=coordinates, 
                scores=scores, 
                score_threshold=score_threshold, 
                nms_threshold=nms_threshold
            )
            data_pack[row] = [all_data[x] for x in indexes]
        
        return data_pack

    def get_coordinates(
        self: 'Butcher',
        output_data: Tuple[np.ndarray],
        keys: Union[Any, List[Any]]
    ) -> DataPack[IDs, Tuple[ZoneInt, float]]:        
        """
            Extracts coordinates from detection outputs.

        Parameters:
            - self (Butcher): Instance of the 'Butcher' class.
            - output_data (Tuple[np.ndarray]): Tuple containing 
            the input image representation (blob) and detection results as NumPy arrays.
            - keys (Union[Any, List[Any]]): Either a single key or a list of 
            keys to extract data for.

        Returns:
            'DataPack' with 'IDs' as key and contents as tuples with zones and confidence 
            scores as float.
        """
        
        fridge = {}
        img_height, img_width = ()

        if not isinstance(keys, list):
            keys = [keys]

        for detections in output_data:
            for detection in detections:
                scores = detection[5:]
                single_key = np.argmax(scores)
                score = scores[single_key]

                if single_key in keys:
                    if single_key not in fridge:
                        fridge[single_key] = []

                    w = detection[2] * img_width
                    h = detection[3] * img_height
                    x = (detection[0] * img_width) - w / 2
                    y = (detection[1] * img_height) - h / 2
                    
                    data = (
                        int(x), int(x + w), int(y), int(y + h),
                        score
                    )
                    fridge[single_key].append(data)

        return fridge

    def extract_image_segments(
        self: 'Butcher',
        image: np.ndarray,
        data_pack: DataPack[IDs, Tuple[ZoneInt, float]]
    ) -> DataPack[IDs, np.ndarray]:       
        """
            Extracts image segments from the original image based on the coordinates 
            provided in the 'data_pack'.

        Parameters:
            - self (Butcher): An instance of the 'Butcher' class.
            - image (np.ndarray): The original image from which segments are to 
            be extracted.
            - data_pack (DataPack[IDs, Tuple[ZoneInt, float]]): 'DataPack' with 'IDs' 
            as key and contents as tuples with zones and confidence scores as float.

        Returns:
            'data_pack' with 'IDs' as key containing images in array format
        """
        
        new_data_pack = {}
        for key in data_pack:
            if key not in new_data_pack:
                new_data_pack[key] = []

            for data in data_pack[key]:
                zone = data[0]
                new_data_pack[key].append(
                    cv2.cvtColor(
                        src=image[
                            zone[2]:zone[3], #YCoordinate, Height
                            zone[0]:zone[1]  #XCoordinate, Width
                        ],
                        code=cv2.COLOR_BGR2RGB
                    )
                )
        
        return new_data_pack
                