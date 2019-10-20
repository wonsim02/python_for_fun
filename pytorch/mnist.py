import torch
import torch.nn as nn
import torch.optim as optim

import torchvision.transforms as transforms
import torchvision.datasets as datasets
import time

# training을 위한 변수
BATCH_SIZE = 50             # loader가 데이터를 불러오는 단위
NUM_EPOCH = 10              # 반복학습 횟수
VALID_MODE = False          # True이면 validation 단계를 거침
VALID_RATIO = 0.1           # training data 중에서 validation data의 비율
HYPERPARAM_FINAL = 60       # 후보들에 대한 validation 결과 최종 결정된 hyperparameter
HYPERPARAM_CANDIDATES = [40, 50, 60, 70] # hyperparameter 후보 list

train_data, test_data, splitted_train_data = None, None, None


class Model(nn.Module):
    def __init__(self, hidden_size):
        super(Model, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 6, 5, 1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(6, 16, 3, 1), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.fc = nn.Sequential(
            nn.Linear(16*5*5, hidden_size), nn.ReLU(), 
            nn.Linear(hidden_size, 10)
        )
    
    def forward(self, x):
        x = self.conv(x)    # convolution
        x = x.view(-1, 400) # reshape
        x = self.fc(x)      # fully-connected
        return x


# MNIST data 로드
def load_data(root = './data', download = True):
    _trd = datasets.MNIST(root = root, train = True, download = download, transform = transforms.ToTensor())
    _ted = datasets.MNIST(root = root, train = False, download = download, transform = transforms.ToTensor())
    
    valid_size = int(VALID_RATIO * len(_trd))
    _spl = torch.utils.data.random_split(_trd, [len(_trd)-valid_size, valid_size])

    return (_trd, _ted, _spl)


# candidates에 validation을 거칠 hyperparameter 값의 후보 list를 받음
# return value는 각 hyperparameter 값에 대한 accuracy
def validate(candidates):
    train_portion, valid_portion = splitted_train_data

    train_loader = torch.utils.data.DataLoader(train_portion, batch_size=BATCH_SIZE)
    valid_loader = torch.utils.data.DataLoader(valid_portion, batch_size=BATCH_SIZE)

    log = []
    print("training with %d data & validating with %d data..." % (len(train_portion), len(valid_portion)))

    for cand in candidates:
        print("...as the size of the hidden layer = %d." % cand)
        net = Model(cand)
        optimizer = optim.Adam(net.parameters())
        criterion = nn.CrossEntropyLoss()

        st = time.time()
        for epoch in range(NUM_EPOCH):
            net.train()
            for xs, ys in train_loader:
                ys_pred = net(xs)
                loss = criterion(ys_pred, ys)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            net.eval()
            tot_loss, tot_acc = 0, 0
            for xs, ys in valid_loader:
                ys_pred = net(xs)
                loss = criterion(ys_pred, ys)
                
                tot_loss += loss
                tot_acc += (ys==ys_pred.argmax(1)).sum()

            tot_acc = 100 * tot_acc.double() / len(valid_portion)
            print(" - Epoch %d/%d. validation loss: %.2f, \t validation acc: %.2f%%....(elapsed %.2fs)" % (
                epoch+1, NUM_EPOCH, tot_loss, tot_acc, time.time()-st))
        
        log.append((tot_acc, cand))
    
    print()
    return log


# 실제로 training을 진행한 후 test set으로 성능 검사
def train_and_test():
    print("training with %d data... " % len(train_data))

    train_loader = torch.utils.data.DataLoader(train_data, batch_size = BATCH_SIZE)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size = BATCH_SIZE)
    
    net = Model(HYPERPARAM_FINAL)
    optimizer = optim.Adam(net.parameters())
    criterion = nn.CrossEntropyLoss()

    net.train()
    st = time.time()
    for epoch in range(NUM_EPOCH):
        tot_loss = 0
        for xs,ys in train_loader:
            ys_pred = net(xs)
            loss = criterion(ys_pred, ys)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            tot_loss += loss

        print(" - Epoch %d/%d. train loss: %f....(elapsed %.2fs)" % (
            epoch+1, NUM_EPOCH, tot_loss, time.time()-st))
    print()

    print("final test with %d data..." % len(test_data))
    net.eval()
    acc, tot = 0, 0
    res = [[0 for j in range(10)] for i in range(10)]
    for x, y in test_loader:
        y_pred = net(x).argmax(1).squeeze()
        for i in range(len(y)):
            res[y[i]][y_pred[i]] += 1
        acc, tot = acc+(y==y_pred).sum(), tot+len(y)
    print("test accuracy : %d/%d (%.2f%%)" % (acc, tot, 100*acc.double()/tot))
    print("          pred.0 pred.1 pred.2 pred.3 pred.4 pred.5 pred.6 pred.7 pred.8 pred.9 recall")
    
    num_pred, num_match = torch.zeros(10), torch.zeros(10)
    for actual in range(10):
        print(" actual.%d" % actual, " ".join(map("%6d".__mod__, res[actual])), "%.2f%%" % (
            100*res[actual][actual]/sum(res[actual])))
        num_pred.add_(torch.tensor(res[actual], dtype = torch.float32))
        num_match[actual] = res[actual][actual]
    print("precision", " ".join(map("%.2f%%".__mod__, 100*num_match/num_pred)))


if __name__ == '__main__':
    train_data, test_data, splitted_train_data = load_data()

    if VALID_MODE:
        valid_log = validate(HYPERPARAM_CANDIDATES)
        print("=============== VALIDATION COMPLETE ===============")
        for acc, cand in valid_log:
            print("Accuracy with the size of the hidden layer = %02d : %.2f%%" % (cand, acc))
        valid_log.sort()
        HYPERPARAM_FINAL = valid_log[-1][1]
        print("...Selected size of the hidden layer = %d." % HYPERPARAM_FINAL)
        print()

    train_and_test()
