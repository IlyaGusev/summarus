#!/usr/bin/env python3

"""
This is code to take a trained Fairseq model and discard the ADAM optimizer state,
which is not needed at test time. It can reduce a model size by ~70%.

Original author: Brian Thompson
"""

from fairseq import checkpoint_utils
import torch

import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Strip ADAM optimizer \
        states out of a fairseq checkpoint to make it smaller for release.')
    parser.add_argument('fin', type=str, help='input checkpoint file')
    parser.add_argument('fout', type=str, help='output checkpoint file')
    args = parser.parse_args()

    assert args.fin != args.fout  # do not allow overwrite input

    model = checkpoint_utils.load_checkpoint_to_cpu(args.fin)
    for key in model['last_optimizer_state']['state']:
        del model['last_optimizer_state']['state'][key]['exp_avg_sq']
        del model['last_optimizer_state']['state'][key]['exp_avg']

    torch.save(model, f=open(args.fout, 'wb'))
