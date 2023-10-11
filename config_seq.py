import os

class GlobalConfig:
    """ base architecture configurations """
	# Data
    seq_len = 5 # input timesteps
    pred_len = 1 # future waypoints predicted (ORIGINAL WAS 4)

    data_root = './Scenario36/' #path to the dataset

    n_views = 2 # no. of camera views
    n_gps = 2 # no. of gps views
    input_resolution = 256

    scale = 1 # image pre-processing
    crop = 256 # image pre-processing

    lr = 1e-4 # learning rate

    # Conv Encoder
    vert_anchors = 8
    horz_anchors = 8
    anchors = vert_anchors * horz_anchors

	# GPT Encoder
    n_embd = 192
    block_exp = 4
    n_layer = 8
    n_head = 4
    n_scale = 4
    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1

    def __init__(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)
