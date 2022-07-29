import torch
import wandb as wb
import torch.nn as nn
import torch.optim as optim
from model import BPSDataset, CNN
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split

DEVICE = torch.device('cuda:0')

def load_data(root, new=False):
    transform = transforms.Normalize((196.0, 216.0, 216.0), (86.0, 57.0, 56.0))
    data = BPSDataset(root, transform)
    total = len(data)
    mid_idx = int(0.8 * total)
    traindata, valdata = random_split(data, [mid_idx, total - mid_idx])
    # print(len(traindata))
    # temploader = DataLoader(traindata, 2016)
    # for batch, y in temploader:
    #     batch = batch.float()
    #     batch1 = batch[0, :, :]
    #     batch2 = batch[1, :, :]
    #     batch3 = batch[2, :, :]
    #     print(batch1.mean(), batch1.std())
    #     print(batch2.mean(), batch2.std())
    #     print(batch3.mean(), batch3.std())

    return traindata, valdata

def accuracy(pred_val, true_val):
    pred_val = torch.argmax(pred_val, -1)
    true_val = torch.argmax(true_val, -1)
    correct = (pred_val == true_val).sum()

    return correct.item() / pred_val.shape[0]

    

def train_1_epoch(model, loader, crit, opt, acc=False):
    model.train()
    tloss = 0
    corr_frac = None

    for b, (inp, y) in enumerate(loader):
        inp = inp.to(DEVICE)
        y = y.to(DEVICE)
        opt.zero_grad()
        out = model(inp)
        loss = crit(out, y)
        loss.backward()
        opt.step()
        tloss += loss.item()
        if acc:
            corr_frac = accuracy(out, y)

    return tloss / (b + 1), corr_frac


def evaluate(model, loader, crit, acc=False):
    model.eval()
    vloss = 0
    corr_frac = None

    with torch.no_grad():
        for b, (inp, y) in enumerate(loader):
            inp = inp.to(DEVICE)
            y = y.to(DEVICE)
            out = model(inp)
            loss = crit(out, y)
            vloss += loss.item()
            if acc:
                corr_frac = accuracy(out, y)

    return vloss / (b + 1), corr_frac

def run(setting, save=False):

    root = '/home/minion/Documents/boulder_parchment_shears/rps'
    traindata, valdata = load_data(root)

    trainloader = DataLoader(traindata, batch_size=setting['batch'],
                             shuffle=True)
    valloader = DataLoader(valdata, batch_size=setting['batch'],
                           shuffle=True)
    # raise Exception
    net = CNN(
            setting['ks'],
            setting['channels'],
            setting['linear']
        ).to(DEVICE)
    # net = CNN(
    #         [(5, 1), (5, 1)],
    #         [6, 16],
    #         [24 * 24, 12 * 12, 3]
    # ).to(DEVICE)

    criteria = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=setting['lr'])

    def weights_init(m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
    net.apply(weights_init)

    for e in range(setting['epoch']):
        tloss, tacc = train_1_epoch(net, trainloader, criteria, optimizer, True)
        vloss, vacc = evaluate(net, valloader, criteria, True)
        # print(tacc, vacc)
        # if e + 1)

        wb.log({'Train Loss': tloss,
                'Train Accuracy': tacc,
                'Validation Loss': vloss,
                'Valication Accuracy': vacc
            }
        )    
    if save:
        save_model = net.state_dict()
        torch.save(save_model, 'current.pt')

if __name__ == '__main__':
    config = {
        'epoch': 30,
        'batch': 24,
        'lr': 0.001,
        'ks': [(10, 2), (10, 1)],
        'channels': [6, 16],
        'linear': [24 * 24, 12, 3]
    }

    wb.init('boulder_parchment_shears', entity='sh-divya')
    wb.config = config
    name = [key + '-' + str(config[key]) for key in ['lr', 'batch']]
    wb.run.name = '2layers' + '_'.join(name)
    run(config, True)
