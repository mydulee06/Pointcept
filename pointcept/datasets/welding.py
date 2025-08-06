import numpy as np

from .builder import DATASETS
from .defaults import DefaultDataset


def segment2onehot(segment, num_classes):
    mask = segment != -1
    segment_onehot = np.zeros((len(segment), num_classes), dtype=np.float32)
    segment_onehot[mask, segment[mask]] = 1
    return segment_onehot


@DATASETS.register_module()
class WeldingDataset(DefaultDataset):
    VALID_ASSETS = [
        "coord",
        "color",
        "segment",
        "segment_onehot",
        "edge",
        "spline_t",
        "spline_c",
        "spline_k",
    ]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_data(self, idx):
        data_dict = super().get_data(idx)

        if "segment_onehot" not in data_dict:
            data_dict["segment_onehot"] = segment2onehot(data_dict["segment"], 2)

        data_dict["index_valid_keys"] = [
            "coord",
            "color",
            "segment",
            "segment_onehot",
        ]

        return data_dict