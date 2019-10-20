import mnist as mine
import torch, torchvision
import time, sys, os

FILENAME = './model'
LOG_PATH = './log.txt'
CREATE_MSG = "The new classifier has been created.\n"
TRAIN = True


current_time = lambda: "%04d/%02d/%02d %02d:%02d:%02d" % ((
    lambda t: tuple(
        map(t.__getattribute__, ['tm_year', 'tm_mon', 'tm_mday', 'tm_hour', 'tm_min', 'tm_sec'])
    ))(time.localtime()))


def train(net, num_epoch):
    train_data = torchvision.datasets.MNIST(
        root = './data', train = True, transform = torchvision.transforms.ToTensor())
    train_loader = torch.utils.data.DataLoader(train_data, batch_size = mine.BATCH_SIZE)

    optimizer = torch.optim.Adam(net.parameters())
    criterion = torch.nn.CrossEntropyLoss()

    print("training with %d data... " % len(train_data))

    net.train()
    st = time.time()
    for epoch in range(num_epoch):
        tot_loss = 0
        for xs,ys in train_loader:
            ys_pred = net(xs)
            loss = criterion(ys_pred, ys)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            tot_loss += loss

        print(" - Epoch %d/%d. train loss: %f....(elapsed %.2fs)" % (
            epoch+1, num_epoch, tot_loss, time.time()-st))
    print()


def test(net):
    test_data = torchvision.datasets.MNIST(
        root = './data', train = False, transform = torchvision.transforms.ToTensor())
    test_loader = torch.utils.data.DataLoader(test_data, batch_size = mine.BATCH_SIZE)
    test_data_raw = torchvision.datasets.MNIST(root = './data', train = False)

    DIRNAME = "./"+current_time().replace(" ", "_").replace("/", "_").replace(":", "_")+"/"
    SUBDIRNAME = [("pred_%d_wrong/" % num, "pred_%d_right/" % num) for num in range(10)]
    os.mkdir(DIRNAME)
    for wrong, right in SUBDIRNAME:
        os.mkdir(DIRNAME+wrong)
        os.mkdir(DIRNAME+right)

    print("testing the model with %d data..." % len(test_data))

    net.eval()
    acc, tot, idx = 0, 0, 0
    res = [[0 for j in range(10)] for i in range(10)]

    for x, y in test_loader:
        y_pred = net(x).argmax(1).squeeze()
        for i in range(len(y)):
            test_data_raw[idx][0].save(DIRNAME+SUBDIRNAME[y_pred[i]][y[i]==y_pred[i]]+"%04d_%d.png"%(idx, y[i]))
            res[y[i]][y_pred[i]], idx = res[y[i]][y_pred[i]]+1, idx+1
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
    print()


if __name__ == '__main__':
    net = mine.Model(mine.HYPERPARAM_FINAL)
    log_wr = open(LOG_PATH, "at")
    
    with open(LOG_PATH) as log_rd:
        if log_rd.readline() == CREATE_MSG:
            net.load_state_dict(torch.load(FILENAME))
        else:
            log_wr.write(CREATE_MSG)
    
    def print(*args):
        ln = " ".join(map(str, args))+"\n"
        sys.stdout.write(ln)
        log_wr.write(ln)
    
    print("========== EXECUTED AT %s ==========" % current_time())
    print("Train mode:", TRAIN)
    print()

    if TRAIN:
        train(net, 1)
        torch.save(net.state_dict(), FILENAME)
    else:
        test(net)
    
    log_wr.close()