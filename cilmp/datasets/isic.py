import os
import pickle

from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
from dassl.utils import mkdir_if_missing

from .oxford_pets import OxfordPets
from .dtd import DescribableTextures as DTD

IGNORED = ["BACKGROUND_Google", "Faces_easy"]
NEW_CNAMES = {
    "airplanes": "airplane",
    "Faces": "face",
    "Leopards": "leopard",
    "Motorbikes": "motorbike",
}


@DATASET_REGISTRY.register()
class ISIC(DatasetBase):

    dataset_dir = "isic"
    all_class_names = ['Melanoma',
                        'Melanocytic nevus',
                        'Basal cell carcinoma',
                        'Actinic keratosis',
                        'Benign keratosis',
                        'Dermatofibroma',
                        'Vascular lesion']

    def __init__(self, cfg):
        self.all_class_names = ['Melanoma',
                                'Melanocytic nevus',
                                'Basal cell carcinoma',
                                'Actinic keratosis',
                                'Benign keratosis',
                                'Dermatofibroma',
                                'Vascular lesion']
        
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir)
        self.split_path = os.path.join(self.dataset_dir, "split_isic.json")
        # self.split_fewshot_dir = os.path.join(self.dataset_dir, "split_fewshot")
        # mkdir_if_missing(self.split_fewshot_dir)

        train, val, test = OxfordPets.read_split(self.split_path, self.image_dir)
        super().__init__(train_x=train, val=val, test=test)
