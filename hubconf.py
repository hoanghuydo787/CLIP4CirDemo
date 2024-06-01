dependencies = ["torch"]

import torch
from torch import nn

from model import Combiner

#cirr combiner url
# CIRR_URL = "https://www.dropbox.com/s/cdesqz7yincaq8g/cirr_combiner.pth?dl=1"
CIRR_URL = "./models/cirr_combiner.pth"
#fashionIQ combiner url
# FASHIONIQ_URL = "https://www.dropbox.com/s/tra1no8ionus3lk/fashionIQ_combiner.pth?dl=1"
# FASHIONIQ_URL = "./models/fashionIQ_combiner.pth"
FASHIONIQ_URL = "./models/combiner.pt"


if torch.cuda.is_available():
    torch.cuda.set_device(0)
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


def combiner(dataset: str) -> nn.Module:
    """
    Get the Combiner model trained on CIRR or FashionIQ datasets
    :param dataset: dataset on which the Combiner has been trained
    :return: Combiner model
    """
    model = Combiner(640, 640 * 4, 640 * 8)
    if dataset == 'cirr':
        # state_dict = torch.hub.load_state_dict_from_url(CIRR_URL, progress=True, file_name='cirr_combiner',
                                                        # map_location=device)
        state_dict = torch.load(CIRR_URL, map_location=device)
    elif dataset == 'fashionIQ':
        # state_dict = torch.hub.load_state_dict_from_url(FASHIONIQ_URL, progress=True, file_name='fashionIQ_combiner',
                                                        # map_location=device)
        state_dict = torch.load(FASHIONIQ_URL, map_location=device)
        print(FASHIONIQ_URL)
    else:
        raise ValueError("Dataset should be in ['cirr', 'fashionIQ'] ")
    model.load_state_dict(state_dict['Combiner'])
    return model
