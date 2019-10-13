import pytest
import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 3)
        self.fc1 = nn.Linear(16 * 6 * 6, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


class MultiInputNet(nn.Module):
    def __init__(self):
        super(MultiInputNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1), nn.ReLU()
        )

    def forward(self, x, y):
        x1 = self.features(x)
        x2 = self.features(y)
        return x1, x2


class SimpleBatchNormModel(nn.Module):
    def __init__(self):
        super(SimpleBatchNormModel, self).__init__()
        hidden = 15
        input_size = 20
        self.main = nn.Sequential(
            nn.Linear(input_size, hidden, bias=False),
            nn.BatchNorm1d(hidden),
            nn.Linear(hidden, hidden, bias=False),
            nn.BatchNorm1d(hidden),
            nn.Linear(hidden, 1),
        )

    def forward(self, input):
        output = self.main(input)
        return output


class Generator(nn.Module):
    def __init__(self, ngpu=1):
        super(Generator, self).__init__()
        nz = 100
        ngf = 64
        nc = 3
        self.ngpu = ngpu
        self.main = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(
                self.main, input, range(self.ngpu)
            )
        else:
            output = self.main(input)
        return output


class LSTMTagger(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores


class SimpleConv(nn.Module):
    def __init__(self):
        super(SimpleConv, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1), nn.ReLU()
        )

    def forward(self, x, y):
        x1 = self.features(x)
        x2 = self.features(y)
        return x1, x2


class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=5),
            nn.ReLU(True),
            nn.Conv2d(6, 16, kernel_size=5),
            nn.ReLU(True),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(16, 6, kernel_size=5),
            nn.ReLU(True),
            nn.ConvTranspose2d(6, 3, kernel_size=5),
            nn.ReLU(True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


@pytest.fixture(scope='session')
def simple_model():
    net = SimpleNet()
    return net


@pytest.fixture(scope='session')
def mobilenet():
    model = torchvision.models.mobilenet_v2()
    return model


@pytest.fixture(scope='session')
def multi_input_net():
    model = MultiInputNet()
    return model


@pytest.fixture(scope='session')
def lstm_tagger():
    model = LSTMTagger(6, 6, 5, 3)
    return model


@pytest.fixture(scope='session')
def netgenerator():
    model = Generator()
    return model


@pytest.fixture(scope='session')
def netbatchnorm():
    model = SimpleBatchNormModel()
    return model


@pytest.fixture(scope='session')
def simpleconv():
    model = SimpleConv()
    return model


@pytest.fixture(scope='session')
def autoencoder():
    net = Autoencoder()
    return net


class RNNModel(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim, n_layers):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.rnn = nn.RNN(input_size, hidden_dim, n_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_size)

    def forward(self, x):
        batch_size = x.size(0)
        hidden = self.init_hidden(batch_size)
        out, hidden = self.rnn(x, hidden)
        out = out.contiguous().view(-1, self.hidden_dim)
        out = self.fc(out)
        return out, hidden

    def init_hidden(self, batch_size):
        hidden = torch.zeros(self.n_layers, batch_size, self.hidden_dim)
        return hidden


@pytest.fixture(scope='session')
def rnn():
    net = RNNModel(3, 1, 5, 3)
    return net


class MultiInputNet2(nn.Module):
    def __init__(self):
        super(MultiInputNet2, self).__init__()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 96, 3, padding=1)
        self.conv3 = nn.Conv2d(96, 128, 3, padding=1)
        self.conv4 = nn.Conv2d(128, 192, 3, padding=1)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear1 = nn.Linear(192, 64)
        self.linear2 = nn.Linear(64, 4)

    def forward(self, image, unrefined_bounding_box):
        x = self.pool(F.relu(self.conv1(image)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = F.relu(self.conv4(x))

        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = x + unrefined_bounding_box
        return x


@pytest.fixture(scope='session')
def multi_input_net2():
    net = MultiInputNet2()
    return net


class LSTMModel(nn.Module):
    # https://www.deeplearningwizard.com/deep_learning/practical_pytorch/
    # pytorch_lstm_neuralnetwork/
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(
            self.layer_dim, x.size(0), self.hidden_dim
        ).requires_grad_()
        c0 = torch.zeros(
            self.layer_dim, x.size(0), self.hidden_dim
        ).requires_grad_()
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        out = self.fc(out[:, -1, :])
        return out


@pytest.fixture(scope='session')
def lstm_model():
    input_dim = 28
    hidden_dim = 100
    layer_dim = 1
    output_dim = 10
    model = LSTMModel(input_dim, hidden_dim, layer_dim, output_dim)
    return model
