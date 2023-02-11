import os
import random
import logging
import datetime
from pytz import timezone

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.dateministic = True
    torch.backends.cudnn.benchmark = False


def customtime(*args):
    return datetime.now(timezone('Asia/Tokyo')).timetuple()


def get_logger(file_name, is_filehandler=True):
    logger = logging.getLogger(__name__)
    log_format = '%(asctime)s:%(lineno)d:%(levelname)s:%(message)s'
    formatter = logging.Formatter(log_format)
    formatter.converter = customtime
    for h in logger.handlers[:]:
        logger.removeHandler(h)
        h.close()
    logger.setLevel(logging.INFO)
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    if is_filehandler:
        fh = logging.FileHandler(filename=file_name)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    return logger
