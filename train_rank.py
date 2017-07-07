import math
import sys
import os.path
import zipfile
import random
from collections import Counter
import pickle
import re

import numpy as np

import chainer
from chainer import cuda, Variable
from chainer import optimizers, serializers
import chainer.functions as F


from rank import TextEncoder

ALPHABET = r''' !"&\'()*+,-./0123456789:;<>?ABCDEFGHIJKLMNOPQRSTUVWXYZ_''' + \
           r'''abcdefghijklmnopqrstuvwxyzÄÅÖäåö–“”…'''

class AuthorCorpus:
    def __init__(self, path):
        self.zipfile = zipfile.ZipFile(path)
        self.files = [info.filename for info in self.zipfile.infolist()
                      if info.file_size >= 4096 and
                        not info.filename.endswith('/')]
        self.heldout = set()

    def sample_chunk(self, length):
        while True:
            filename = random.choice(self.files)
            if filename in self.heldout: continue
            with self.zipfile.open(filename) as f:
                text = str(f.read(), 'utf-8')
                text = re.sub(r'\s+', ' ', text)
            if len(text) < length: continue
            pos = random.randrange(len(text) - length)
            return text[pos:pos+length]

    def sample_pair(self, length1, length2, replace=True):
        while True:
            filename = random.choice(self.files)
            if filename in self.heldout: continue
            with self.zipfile.open(filename) as f:
                text = str(f.read(), 'utf-8')
                text = re.sub(r'\s+', ' ', text)
            if len(text) < length1 + length2: continue
            pos1 = random.randrange(len(text) - length1)
            # Number of characters available before first sample
            before = max(0, pos1 - length2)
            # Number of characters available after first sample
            after = max(0, len(text) - (pos1 + length1) - length2)
            if not (before or after): continue
            if random.random() < before / (before+after):
                pos2 = random.randrange(before)
            else:
                pos2 = pos1 + length1 + random.randrange(after)
            if not replace: self.heldout.add(filename)
            return text[pos1:pos1+length1], text[pos2:pos2+length2]

    def sample_batch(self, batch_size, length1, length2, replace=True):
        return [self.sample_pair(length1, length2, replace=replace)
                for _ in range(batch_size)]


def main():
    import time
    import argparse

    parser = argparse.ArgumentParser(
            description='Text ranking training')
    parser.add_argument('--blog-authors', type=str, required=True,
                        metavar='FILE',
                        help='Training corpus (.zip file with text files)')
    parser.add_argument('--model', type=str, required=True,
                        metavar='FILE',
                        help='Prefix of model files')
    parser.add_argument('--batch-size', type=int,
                        metavar='N', default=64,
                        help='Batch size')
    parser.add_argument('--sample-size', type=int,
                        metavar='N', default=1024,
                        help='Size of text samples in characters')
    parser.add_argument('--gpu', type=int,
                        metavar='N', default=0,
                        help='GPU to use (-1 for CPU)')
    parser.add_argument('--n-test-batches', type=int,
                        metavar='N', default=4,
                        help='Number of batches for validation')
    parser.add_argument('--n-resnet-blocks', type=int,
                        metavar='N', default=10,
                        help='Number of residual blocks')
    parser.add_argument('--cnn-size', type=int,
                        metavar='N', default=512,
                        help='Size of encoder convolutional layers')
    parser.add_argument('--semi-supervised', action='store_true',
                        help='Use semi-supervised training using blog authors')
    parser.add_argument('texts', nargs='+', metavar='FILE')
    args = parser.parse_args()

    model_prefix = args.model

    alphabet = '\x00' + ALPHABET
    alphabet_idx = {c:i for i,c in enumerate(alphabet)}

    print('Alphabet size: %d' % len(alphabet), file=sys.stderr, flush=True)

    filename_metadata = model_prefix + '.metadata'
    filename_model = model_prefix + '.model'
    filename_optimizer = model_prefix + '.optimizer'

    try:
        with open(filename_metadata, 'rb') as f:
            print('Loading old model hyperparameters...',
                  file=sys.stderr, flush=True)
            args2, _ = pickle.load(f)
            args.cnn_size = args2.cnn_size
            args.n_resnet_blocks = args2.n_resnet_blocks
    except FileNotFoundError:
        print('Creating new model...', file=sys.stderr, flush=True)
        with open(filename_metadata, 'wb') as f:
            pickle.dump((args, alphabet), f)

    good_texts = []
    for filename in args.texts:
        with open(filename, 'r', encoding='utf-8') as f:
            good_texts.append(f.read())

    print('Counting corpus files...', file=sys.stderr, flush=True)
    corpus = AuthorCorpus(args.blog_authors)
    print('Corpus contains %d files' % len(corpus.files), file=sys.stderr,
            flush=True)


    batch_size = args.batch_size
    text_length = args.sample_size
    n_test_batches = args.n_test_batches
    gpu = args.gpu

    model = TextEncoder(len(alphabet), args.n_resnet_blocks, args.cnn_size)

    if os.path.exists(filename_model):
        print('Loading model parameters...', file=sys.stderr, flush=True)
        serializers.load_npz(filename_model, model)

    optimizer = optimizers.Adam()
    optimizer.use_cleargrads()
    optimizer.setup(model)

    if os.path.exists(filename_optimizer):
        print('Loading optimizer state...', file=sys.stderr, flush=True)
        serializers.load_npz(filename_optimizer, optimizer)

    if gpu >= 0: model.to_gpu(gpu)

    xp = model.xp

    def encode_batch(chunks):
        batch_int = [[alphabet_idx.get(c, 0) for c in chunk]
                     for chunk in chunks]
        return xp.array(batch_int, dtype=np.int32)

    def get_snippet(corpus, n):
        i = random.randint(0, len(corpus)-n)
        return corpus[i:i+n]

    def get_batch(semi_supervised=False):
        texts_a, texts_b, is_better = [], [], []

        # Pairs of chunks from the good domain, equal quality
        for i in range(batch_size // (4 if semi_supervised else 2)):
            source_a, source_b = random.sample(good_texts, 2)
            texts_a.append(get_snippet(source_a, text_length))
            texts_b.append(get_snippet(source_b, text_length))
            is_better.append(0.5)

        # Pairs of chunks from the same blog authors, equal quality
        for i in range((batch_size // 4) if semi_supervised else 0):
            text_a, text_b = corpus.sample_pair(text_length, text_length)
            texts_a.append(text_a)
            texts_b.append(text_b)
            is_better.append(0.5)

        # Pairs of chunks from different blog authors, trained on predicted
        # quality difference on separate chunks from the same authors
        if semi_supervised:
            test_texts_a, test_texts_b = [], []
            for i in range(batch_size // 4):
                text_a1, text_a2 = corpus.sample_pair(text_length, text_length)
                text_b1, text_b2 = corpus.sample_pair(text_length, text_length)
                test_texts_a.append(text_a1)
                test_texts_b.append(text_b1)
                texts_a.append(text_a2)
                texts_b.append(text_b2)
            with chainer.using_config('train', False):
                score1 = model(encode_batch(test_texts_a))
                score2 = model(encode_batch(test_texts_b))
                pred_is_better = F.sigmoid(score1 - score2)
                is_better.extend(list(cuda.to_cpu(pred_is_better.data)))
            del score1
            del score2
            del pred_is_better

        # Pairs of blog/published chunks, blog is always worse
        for i in range(batch_size // (4 if semi_supervised else 2)):
            source = random.choice(good_texts)
            texts_a.append(corpus.sample_chunk(text_length))
            texts_b.append(get_snippet(source, text_length))
            is_better.append(0.0)

        return (encode_batch(texts_a),
                encode_batch(texts_b),
                xp.array([[x] for x in is_better], dtype=xp.float32))

    def compute_loss(batch_a, batch_b, is_better):
        score_a = model(batch_a)
        score_b = model(batch_b)
        return F.bernoulli_nll(is_better, score_a - score_b, reduce='no')

    #test_batches = [get_batch(replace=False) for _ in range(n_test_batches)]

    n_batches = 0

    while True:
        t0 = time.time()
        train_batch = get_batch(args.semi_supervised and n_batches >= 100)
        t1 = time.time()
        loss_vector = compute_loss(*train_batch)
        loss = F.sum(loss_vector)
        model.cleargrads()
        loss.backward()
        optimizer.update()
        t2 = time.time()

        print('TRAIN %.3f %.3f %.4f' % (
                t1-t0, t2-t1, cuda.to_cpu(loss.data)/math.log(2)),
              flush=True)
        target = list(cuda.to_cpu(F.flatten(train_batch[-1]).data))
        print(' '.join('%.3f'%x for x in target))
        print(' '.join('%.3f'%x for x in cuda.to_cpu(loss_vector.data)))

        del loss
        del train_batch

        if n_batches % 1000 == 0:
            serializers.save_npz(filename_model, model)
            serializers.save_npz(filename_optimizer, optimizer)

        #if n_batches % (16*n_test_batches) == 0:
        #    test_loss = 0.0
        #    distorted_loss = 0.0
        #    for test, distorted in zip(test_batches, distorted_test_batches):
        #        test_loss += compute_loss(
        #                *batch_to_gpu(*test, volatile='ON'), train=False)
        #        distorted_loss += compute_loss(
        #                *batch_to_gpu(*distorted, volatile='ON'), train=False)

        #    loss_scale = 1 / (math.log(2) * text_length * n_test_batches)
        #    test_loss = test_loss.data.get()*loss_scale
        #    distorted_loss = distorted_loss.data.get()*loss_scale

        #    print('TEST %.4f %.4f %g' % (
        #            test_loss, distorted_loss, distorted_loss-test_loss),
        #        flush=True)

        #    serializers.save_npz(filename_model, model)
        #    serializers.save_npz(filename_optimizer, optimizer)

        n_batches += 1


if __name__ == '__main__': main()

