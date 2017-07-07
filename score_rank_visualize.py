import math
import sys
import os.path
import zipfile
import random
from collections import Counter
import pickle
import re

import numpy as np
from scipy.ndimage.filters import gaussian_filter
import matplotlib.pyplot as plt

import chainer
from chainer import cuda, Variable
from chainer import optimizers, serializers
import chainer.functions as F


from rank import TextEncoder

def main():
    import time
    import argparse

    parser = argparse.ArgumentParser(
            description='Visualize text quality')
    parser.add_argument('--model', type=str, required=True,
                        metavar='FILE',
                        help='Prefix of model files')
    parser.add_argument('--gpu', type=int,
                        metavar='N', default=0,
                        help='GPU to use (-1 for CPU)')
    parser.add_argument('--latex', action='store_true')
    parser.add_argument('input', metavar='FILE')
    args = parser.parse_args()

    model_prefix = args.model

    filename_metadata = model_prefix + '.metadata'
    filename_model = model_prefix + '.model'
    filename_optimizer = model_prefix + '.optimizer'

    try:
        with open(filename_metadata, 'rb') as f:
            print('Loading old model hyperparameters...',
                  file=sys.stderr, flush=True)
            args2, alphabet = pickle.load(f)
            args.cnn_size = args2.cnn_size
            args.n_resnet_blocks = args2.n_resnet_blocks
    except FileNotFoundError:
        print('Unable to load model!', file=sys.stderr, flush=True)
        sys.exit(1)

    with open(args.input, 'r', encoding='utf-8') as f:
        data = [line.strip() for line in f]

    #alphabet = '\x00' + ALPHABET
    alphabet_idx = {c:i for i,c in enumerate(alphabet)}

    print('Alphabet size: %d' % len(alphabet), file=sys.stderr, flush=True)

    gpu = args.gpu

    model = TextEncoder(len(alphabet), args.n_resnet_blocks, args.cnn_size)

    if os.path.exists(filename_model):
        print('Loading model parameters...', file=sys.stderr, flush=True)
        serializers.load_npz(filename_model, model)

    if gpu >= 0: model.to_gpu(gpu)

    xp = model.xp

    def encode_batch(chunks):
        batch_int = [[alphabet_idx.get(c, 0) for c in chunk]
                     for chunk in chunks]
        return xp.array(batch_int, dtype=np.int32)

    cmap = plt.get_cmap('seismic')
    if not args.latex:
        print('<meta http-equiv="Content-Type" content="text/html; charset=utf-8">')
        print('<html><body>')
    for line in data:
        scores = cuda.to_cpu(
                model.sequence_scores(encode_batch([line])).data)[0]
        mean_score, median_score, std_score = \
                np.mean(scores), np.median(scores), np.std(scores)
        scores = gaussian_filter(scores, 3)
        #scores = (scores - scores.min()) / (scores.max() - scores.min())
        #scores = -0.5 *  (scores - np.median(scores)) / scores.std()
        scores = -0.4 * scores
        colors = cmap(1 / (1 + np.exp(-scores)))[:,:3]
        if args.latex:
            colors = 0.75 * colors
            colors = ['%g,%g,%g' % tuple(color.tolist())
                      for color in colors]
            text = [' ' if char.isspace() else r'\textcolor[rgb]{%s}{%s}' % (color, char)
                    for char, color in zip(line, colors)]
            print(r'%.3f & %s \\' % (mean_score, ''.join(text)))
        else:
            colors = (colors * 255).astype(np.uint8)
            colors = ['#%02x%02x%02x' % tuple(color.tolist())
                      for color in colors]
            #int_scores = [int(255*(0.75*float(x) + 0.25)) for x in scores]
            #colors = ['#%02x%02x%02x' % (x,x,x) for x in int_scores]
            print('Mean %.3f +/- %.3f, median %.3f: ' % (
                mean_score, std_score, median_score))
            text = ['<font color="%s">%s</font>' % (color, char)
                    for char, color in zip(line, colors)]
            print(''.join(text))
            print('<br />')
    if not args.latex:
        print('</body></html>')

if __name__ == '__main__': main()

