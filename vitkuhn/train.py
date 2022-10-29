import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from vit_qiutian import vit_base_patch16_224
from data import transform
from vitkuhn.utils import AverageMeter


def train(train_loader, model, criterion, optimizer):
    model.train()
    losses = AverageMeter()
    train_acc = AverageMeter()

    for i, (inputs, labels) in enumerate(train_loader): # batch_size * 3 * 512 * 512 ,batch_szie * 1
        inputs = inputs.cuda()
        labels = labels.cuda()
        # half precision: float32 -> float16
        # inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.update(loss.item(), inputs.size(0))
        train_acc.update(torch.sum(torch.argmax(outputs, dim=1) == labels).item() / inputs.size(0))
        # if i % 10 == 0:
        #     logger.info("Epoch: [{}][{}/{}]\t Loss: {:.4f} Acc: {:.4f}".format(eopoch, i, len(train_loader), losses.avg, train_acc.avg))
    return losses.avg, train_acc.avg

if __name__ == '__main__':
    dataset = torchvision.datasets.ImageFolder(root=r'd:\ANewspace\code\vit-pytorch\datasets\num345', transform=transform)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=2)

    model = vit_base_patch16_224(num_classes=3)
    model.cuda()
    model.train()


    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # optimizer = torch.optim.Adam(parameters, args.lr, betas=(0.9, 0.999), weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=0, last_epoch=-1)
    criterion = nn.CrossEntropyLoss()


    for i, (images, labels) in enumerate(dataloader):
        train_loss, train_acc = train(dataloader, model, criterion, optimizer)
