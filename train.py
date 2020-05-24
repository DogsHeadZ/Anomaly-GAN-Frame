"""
TRAIN SKIP/GANOMALY

. Example: Run the following command from the terminal.
    run train.py                                    \
        --model <skipganomaly, ganomaly>            \
        --dataset cifar10                           \
        --abnormal_class airplane                   \
        --display                                   \
"""

##
# LIBRARIES

from options import Options
from dataloader import load_data
from utils import load_model

##
def main():
    """ Training
    """
    opt = Options().parse()
    dataloader = load_data(opt)
    model = load_model(opt, dataloader)
    model.train()


if __name__ == '__main__':
    main()
