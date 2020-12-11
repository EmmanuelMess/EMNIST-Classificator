import numpy as np
from matplotlib.pyplot import plot
from tqdm import trange
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import emnist
from PIL import Image


def load_image(infilename) :
    img = Image.open(infilename)
    img.load()
    data = np.zeros((28, 28))

    for i in range(img.width):
        for j in range(img.height):
            if (255, 255, 255, 255) == img.getpixel((i, j)):
                data[i][j] = 1
    return data


class BobNet(torch.nn.Module):
    def __init__(self):
        super(BobNet, self).__init__()
        self.l1 = nn.Linear(784, 128, bias=False)
        self.l2 = nn.Linear(128, 27, bias=False)
        self.sm = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = self.l2(x)
        x = self.sm(x)
        return x


# numpy forward pass
def forward(x):
    x = x.dot(l1)
    x = np.maximum(x, 0)
    x = x.dot(l2)
    return x


def numpy_eval():
    Y_test_preds_out = forward(X_test.reshape((-1, 28 * 28)))
    Y_test_preds = np.argmax(Y_test_preds_out, axis=1)
    return (Y_test == Y_test_preds).mean()


if __name__ == '__main__':
    np.set_printoptions(suppress=True)
    torch.set_printoptions(sci_mode=False)
    #torch.manual_seed(43)
    #np.random.seed(43)

    X_train, Y_train = emnist.extract_training_samples('letters')
    X_test, Y_test = emnist.extract_test_samples('letters')

    model = BobNet()

    loss_function = nn.NLLLoss()
    optim = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.005)
    BS = 512
    losses, accuracies = [], []
    for i in (t := trange(20000)):
        samp = np.random.randint(0, X_train.shape[0], size=(BS))
        X = torch.tensor(X_train[samp].reshape((-1, 28 * 28))).float()
        Y = torch.tensor(Y_train[samp]).long()
        model.zero_grad()
        out = model(X)
        cat = torch.argmax(out, dim=1)
        accuracy = (cat == Y).float().mean()
        loss = loss_function(out, Y)
        loss = loss.mean()
        loss.backward()
        optim.step()
        loss, accuracy = loss.item(), accuracy.item()
        losses.append(loss)
        accuracies.append(accuracy)
        t.set_description("loss %.2f accuracy %.2f" % (loss, accuracy))
    plt.ylim(-0.1, 1.1)
    plot(losses)
    plot(accuracies)
    plt.show()

    # evaluation
    Y_test_preds = torch.argmax(model(torch.tensor(X_test.reshape((-1, 28 * 28))).float()), dim=1).numpy()
    (Y_test == Y_test_preds).mean()

    # copy weights from pytorch
    l1 = model.l1.weight.detach().numpy().T
    l2 = model.l2.weight.detach().numpy().T

    print("acc : {}".format(numpy_eval()))

    petru = load_image("data/img.png")
    #TODO preprocess the image
    print(np.argmax(forward(petru.reshape((-1, 28 * 28)))))
    print(forward(petru.reshape((-1, 28 * 28))))

