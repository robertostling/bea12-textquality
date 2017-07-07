from chainer import Chain
import chainer.functions as F
import chainer.links as L


class MultiBNConvolution1D(Chain):
    def __init__(self, ch, kernel_sizes=(3,)):
        super().__init__()

        self.n_convs = len(kernel_sizes)
        ch_out = ch // self.n_convs
        assert ch % self.n_convs == 0
        #links = []
        for i, kernel_size in enumerate(kernel_sizes):
            assert kernel_size%2 == 1
            pad = (kernel_size-1) // 2
            self.add_link(
                    'conv%d'%i, 
                    L.ConvolutionND(1, ch, ch_out, kernel_size, pad=pad))
            self.add_link(
                    'bn%d'%i,
                    L.BatchNormalization(ch_out))
            #links.extend([
            #    ('conv%d'%i,
            #     L.ConvolutionND(1, ch, ch, kernel_size, pad=pad)),
            #    ('bn%d'%i,
            #     L.BatchNormalization(ch))])
        #self.forward = links
        #for link in links: self.add_link(*link)

    def __call__(self, x):
        ys = [getattr(self, 'bn%d'%i)(getattr(self, 'conv%d'%i)(x))
              for i in range(self.n_convs)]
        if self.n_convs == 1: return ys[0]
        else: return F.concat(ys, axis=1)


class ResNet1D(Chain):
    def __init__(self, ch, kernel_sizes=(3,)):
        super().__init__(
                conv1=MultiBNConvolution1D(ch, kernel_sizes),
                conv2=MultiBNConvolution1D(ch, kernel_sizes))

    def __call__(self, x):
        h = self.conv2(F.relu(self.conv1(x)))
        return F.relu(h + x)


class TextEncoder(Chain):
    def __init__(self, alphabet_size, n_layers, conv_size):
        super().__init__(
                embeddings=L.EmbedID(alphabet_size, conv_size),
                output=L.Linear(conv_size, 1))
        links = [('res%d'%i, ResNet1D(conv_size,
                    kernel_sizes=(3,5,7,9) if i == 0 else (3,)))
                 for i in range(n_layers)]
        self.forward = links
        for link in links: self.add_link(*link)

    def resnet(self, x):
        x = self.embeddings(x)
        x = F.transpose(x, axes=(0,2,1))
        for name, _ in self.forward:
            f = getattr(self, name)
            x = f(x)
        return x

    def sequence_scores(self, x):
        x = self.resnet(x)
        # x: (batch_size, ch, length)
        batch_size, ch, length = x.data.shape
        x = F.reshape(F.swapaxes(x, 1, 2), (batch_size*length, ch))
        # x: (batch_size*length, ch)
        scores = self.output(x)
        # scores: (batch_size*length, 1)
        return F.reshape(scores, (batch_size, length))

    def __call__(self, x):
        return self.output(F.mean(self.resnet(x), 2))

