# coding:utf8
import torch as t
import models
from torch.utils.data import DataLoader
from torchnet import meter
import torchvision
from utils.visualize import Visualizer
from tqdm import tqdm
from config import MnistConfig

opt = MnistConfig()

def train(**kwargs):
    opt.parse(kwargs)
    vis = Visualizer(opt.env)

    # step1: configure model
    model = getattr(models, opt.model)()
    if opt.load_model_path:
        model.load(opt.load_model_path)
    if opt.use_gpu: model.cuda()

    # step2: data
    data_root = "data/MNIST"
    transfroms = torchvision.transforms.ToTensor()
    train_set = torchvision.datasets.MNIST(data_root, train=True, transform=transfroms, target_transform=None, download=True)
    val_set = torchvision.datasets.MNIST(data_root, train=False, transform=transfroms, target_transform=None, download=True)
    train_dataloader = DataLoader(train_set, opt.batch_size,
                                  shuffle=True, num_workers=opt.num_workers)
    val_dataloader = DataLoader(val_set, opt.batch_size,
                                shuffle=False, num_workers=opt.num_workers)

    # step3: criterion and optimizer
    criterion = t.nn.CrossEntropyLoss()
    lr = opt.lr
    optimizer = t.optim.SGD(model.parameters(), lr=lr)

    # step4: meters
    loss_meter = meter.AverageValueMeter()
    val_loss_meter = meter.AverageValueMeter()
    train_time_meter = meter.TimeMeter(unit=False)
    train_time_meter.reset()

    # train
    for epoch in range(opt.max_epoch):
        loss_meter.reset()
        val_loss_meter.reset()
        model.train()
        train_correct_cnt = 0.0
        for batch_idx, (input, target) in tqdm(enumerate(train_dataloader)):
            # train model
            if opt.use_gpu:
                input = input.cuda()
                target = target.cuda()

            optimizer.zero_grad()
            output = model(input)
            loss = criterion(output, target)
            pred = t.argmax(output, 1, False)
            train_correct_cnt += (pred == target).sum()
            loss.backward()
            optimizer.step()

            # meters update and visualize
            loss_meter.add(loss.item())

        model.save()

        model.eval()
        val_correct_cnt = 0.0
        for batch_idx, (input, target) in enumerate(val_dataloader):
            if opt.use_gpu:
                input, target = input.cuda(), target.cuda()
            with t.no_grad():
                output = model(input)
                val_loss = criterion(output, target)
                pred = t.argmax(output, 1, False)
                val_correct_cnt += (pred == target).sum()
            val_loss_meter.add(val_loss.item())

        vis.plot("loss", [loss_meter.value()[0], val_loss_meter.value()[0]],
                 opts=dict(legend=["train", "val"]))
        vis.plot_xy("epoch-time", epoch, train_time_meter.value())
        vis.plot_xy("time-loss", train_time_meter.value(), loss_meter.value()[0])
        vis.plot("acc", [float(train_correct_cnt)/len(train_set), float(val_correct_cnt)/len(val_set)],
                 opts=dict(legend=["train", "val"]))
        # vis.plot("accuracy", [])
        # vis.plot('val_accuracy', val_accuracy)
        vis.log(f"epoch:{epoch},lr:{lr},"
                f"loss:{loss_meter.value()[0]},"
                f"val_loss:{val_loss_meter.value()[0]}.")


def help():
    '''
    打印帮助的信息： python file.py help
    '''

    print('''
    usage : python file.py <function> [--args=value]
    <function> := train | test | help
    example: 
            python {0} train --env='env0701' --lr=0.01
            python {0} test --dataset='path/to/dataset/root/'
            python {0} help
    avaiable args:'''.format(__file__))

    from inspect import getsource
    source = (getsource(opt.__class__))
    print(source)


if __name__ == '__main__':
    import fire
    fire.Fire()
