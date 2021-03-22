# -*- coding: utf-8 -*-
import config
from utils import meter
from torch import nn
from torch import optim
from models import PVRNet
from torch.utils.data import DataLoader
from datasets import *
import argparse


def train(train_loader, net, criterion, optimizer, lr_scheduler, epoch):
    """
    train for one epoch on the training set
    """
    batch_time = meter.TimeMeter(True)
    data_time = meter.TimeMeter(True)
    losses = meter.AverageValueMeter()
    prec = meter.ClassErrorMeter(topk=[1], accuracy=True)
    # training mode
    net.train()
    alpha = 0.01

    for i, (views, dps, pcs, labels) in enumerate(train_loader):
        batch_time.reset()

        # Data
        views = views.to(device=config.device)
        pcs = pcs.to(device=config.device)
        dps = dps.to(device=config.device)
        labels = labels.to(device=config.device)
        # print(f'DataSize:\nmulti-views:{views.size()}\ndepth-images:{dps.size()}\npoint-cloud:{pcs.size()}\nlabels:{labels.size()}')
        # Network
        f_pc, f_mv, f_dp, _, _, _, de_p, de_v, de_d, dis_p, dis_v, dis_d, cls_p, cls_v, cls_d, fea, preds = net(pcs, views, dps)  # bz x C x H x W

        # Generator
        optimizer[0].zero_grad()
        rl1 = criterion[1](de_d, f_dp) + criterion[1](de_p, f_pc) + criterion[1](de_v, f_mv)
        valid = torch.FloatTensor(20, 1).fill_(1.0).to(device=config.device)
        fake = torch.FloatTensor(20, 1).fill_(0.0).to(device=config.device)
        g_loss = alpha * (criterion[0](dis_v, valid) + criterion[0](dis_p, valid)) + (1 - alpha) * rl1
        g_loss.backward(retain_graph=True)
        optimizer[0].step()
        lr_scheduler[0].step(epoch=epoch)

        # Classifier
        optimizer[2].zero_grad()
        c_loss = criterion[2](cls_p, f_pc) + criterion[2](cls_v, f_pc) + criterion[2](cls_d, f_pc)  # Different from ARGF
        c_loss.backward(retain_graph=True)
        optimizer[2].step()
        lr_scheduler[2].step(epoch=epoch)

        # Discriminator
        optimizer[1].zero_grad()
        real_loss = criterion[0](dis_d, valid)
        # avg_v += torch.sum(dis_d.squeeze().data) / (len(train_loader) * config.pvd_net.train.batch_sz)
        fake_loss = criterion[0](dis_p, fake) + criterion[0](dis_v, fake)
        d_loss = 0.5 * (real_loss + fake_loss)
        d_loss.backward(retain_graph=True)
        optimizer[1].step()
        lr_scheduler[1].step(epoch=epoch)

        # Fusion
        optimizer[3].zero_grad()
        loss = criterion[3](preds, labels)
        loss.backward()
        optimizer[3].step()
        lr_scheduler[3].step(epoch=epoch)
        prec.add(preds.detach(), labels.detach())
        losses.add(loss.item())  # batchsize

        if i % config.print_freq == 0:
            print(f'Epoch: [{epoch}][{i}/{len(train_loader)}]\t'
                  f'Batch Time {batch_time.value():.3f}\t'
                  f'Epoch Time {data_time.value():.3f}\t'
                  f'Loss {losses.value()[0]:.4f} \t'
                  f'Prec@1 {prec.value(1):.3f}\t')

    print(f'prec at epoch {epoch}: {prec.value(1)} ')


def validate(val_loader, net, epoch):
    """
    validation for one epoch on the val set
    """
    batch_time = meter.TimeMeter(True)
    data_time = meter.TimeMeter(True)
    prec = meter.ClassErrorMeter(topk=[1], accuracy=True)
    retrieval_map = meter.RetrievalMAPMeter()

    # testing mode
    net.eval()

    total_seen_class = [0 for _ in range(40)]
    total_right_class = [0 for _ in range(40)]

    for i, (views, dps, pcs, labels) in enumerate(val_loader):
        batch_time.reset()

        views = views.to(device=config.device)
        pcs = pcs.to(device=config.device)
        dps = views.to(device=config.device)
        labels = labels.to(device=config.device)

        f_pc, f_mv, f_dp, _, _, _, de_p, de_v, de_d, dis_p, dis_v, dis_d, cls_p, cls_v, cls_d, fts, preds = net(pcs, views, dps)  # bz x C x H x W
        # prec.add(preds.data, labels.data)

        prec.add(preds.data, labels.data)
        retrieval_map.add(fts.detach() / torch.norm(fts.detach(), 2, 1, True), labels.detach())
        for j in range(views.size(0)):
            total_seen_class[labels.data[j]] += 1
            total_right_class[labels.data[j]] += (np.argmax(preds.data.cpu(), 1)[j] == labels.cpu()[j])

        if i % config.print_freq == 0:
            print(f'Epoch: [{epoch}][{i}/{len(val_loader)}]\t'
                  f'Batch Time {batch_time.value():.3f}\t'
                  f'Epoch Time {data_time.value():.3f}\t'
                  f'Prec@1 {prec.value(1):.3f}\t')

    mAP = retrieval_map.mAP()
    print(f' instance accuracy at epoch {epoch}: {prec.value(1)} ')
    print(
        f' mean class accuracy at epoch {epoch}: {(np.mean(np.array(total_right_class) / np.array(total_seen_class, dtype=np.float)))} ')
    print(f' map at epoch {epoch}: {mAP} ')
    return prec.value(1), mAP


def save_ckpt(epoch, epoch_pc, epoch_pc_view_dp, best_prec1, net, optimizer,
              training_conf=config.pvd_net):
    ckpt = dict(
        epoch=epoch,
        epoch_pc=epoch_pc,
        epoch_all=epoch_pc_view_dp,
        best_prec1=best_prec1,
        model=net.module.state_dict(),
        optimizer_G=optimizer[0].state_dict(),
        optimizer_D=optimizer[1].state_dict(),
        optimizer_E=optimizer[2].state_dict(),
        optimizer_F=optimizer[3].state_dict(),
        training_conf=training_conf
    )
    torch.save(ckpt, config.pvd_net.ckpt_file)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Main",
    )
    parser.add_argument("-batch_size", '-b', type=int, default=32, help="Batch size")
    parser.add_argument('-gpu', '-g', type=str, default=None, help='GPUS used')
    parser.add_argument(
        "-epochs", '-e', type=int, default=None, help="Number of epochs to train for"
    )
    return parser.parse_args()


def main():
    print('Training Process\nInitializing...\n')
    config.init_env()
    args = parse_args()

    total_batch_sz = config.pvd_net.train.batch_sz * len(config.available_gpus.split(','))
    total_epoch = config.pvd_net.train.max_epoch

    if args.gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
        total_batch_sz = config.pvd_net.train.batch_sz * len(args.gpu.split(','))
    if args.epochs is not None:
        total_epoch = args.epochs

    train_dataset = pc_view_dp_data(config.pvd_net.pc_root,
                                    config.pvd_net.view_root,
                                    config.pvd_net.depth_root,
                                    status=STATUS_TRAIN,
                                    base_model_name=config.base_model_name)
    val_dataset = pc_view_dp_data(config.pvd_net.pc_root,
                                  config.pvd_net.view_root,
                                  config.pvd_net.depth_root,
                                  status=STATUS_TEST,
                                  base_model_name=config.base_model_name)

    train_loader = DataLoader(train_dataset, batch_size=total_batch_sz,
                              num_workers=config.num_workers, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=total_batch_sz,
                            num_workers=config.num_workers, shuffle=True)

    best_prec1 = 0
    best_map = 0
    resume_epoch = 0

    epoch_pc_view_dp = 0
    epoch_pc = 0

    # create model
    net = PVRNet()
    net = net.to(device=config.device)
    net = nn.DataParallel(net)

    # optimizer
    g_params = [{'params': v} for k, v in net.named_parameters() if 'GEN' in k]
    en_params = [{'params': v} for k, v in net.named_parameters() if ('GEN_EN' or 'CLS') in k]
    dis_params = [{'params': v} for k, v in net.named_parameters() if 'DIS' in k]
    f_params = [{'params': v} for k, v in net.named_parameters() if not ('GEN' or 'DIS' or 'CLS') in k]
    if config.pvd_net.train.optim == 'Adam':
        b1 = 0.5
        b2 = 0.999

        optimizer_G = torch.optim.Adam(g_params, weight_decay=config.pvd_net.train.weight_decay,
                                       lr=config.pvd_net.train.all_lr, betas=(b1, b2))
        optimizer_D = torch.optim.Adam(dis_params, lr=config.pvd_net.train.all_lr, betas=(b1, b2),
                                       weight_decay=config.pvd_net.train.weight_decay)
        optimizer_E = torch.optim.Adam(en_params, lr=config.pvd_net.train.all_lr, betas=(b1, b2),
                                       weight_decay=config.pvd_net.train.weight_decay)
        optimizer_F = torch.optim.Adam(f_params, lr=config.pvd_net.train.all_lr, betas=(b1, b2),
                                       weight_decay=config.pvd_net.train.weight_decay)
        optimizer = [optimizer_G, optimizer_D, optimizer_E, optimizer_F]
    else:
        raise NotImplementedError
    print(f'use {config.pvd_net.train.optim} optimizer')
    # print(f'Sclae:{net.module.n_scale} ')

    if config.pvd_net.train.resume:
        print(f'loading pretrained model from {config.pvd_net.ckpt_file}')
        checkpoint = torch.load(config.pvd_net.ckpt_file)
        state_dict = checkpoint['model']
        net.module.load_state_dict(checkpoint['model'])
        # optimizer_fc.load_state_dict(checkpoint['optimizer_pc'])
        # optimizer_all.load_state_dict(checkpoint['optimizer_all'])
        best_prec1 = checkpoint['best_prec1']
        epoch_pc_view_dp = checkpoint['epoch_all']
        epoch_pc = checkpoint['epoch_pc']
        if config.pvd_net.train.resume_epoch is not None:
            resume_epoch = config.pvd_net.train.resume_epoch
        else:
            resume_epoch = max(checkpoint['epoch_pc'], checkpoint['epoch_all'])

    if not config.pvd_net.train.iter_train:  # iter_train 在训练时进行了梯度下降迭代更新参数
        print('No iter')
        lr_scheduler_g = torch.optim.lr_scheduler.StepLR(optimizer_G, 5, 0.3)
        lr_scheduler_d = torch.optim.lr_scheduler.StepLR(optimizer_D, 5, 0.3)
        lr_scheduler_e = torch.optim.lr_scheduler.StepLR(optimizer_E, 5, 0.3)
        lr_scheduler_f = torch.optim.lr_scheduler.StepLR(optimizer_F, 5, 0.3)
    else:
        print('VCIter')
        lr_scheduler_g = torch.optim.lr_scheduler.StepLR(optimizer_G, 6, 0.3)
        lr_scheduler_d = torch.optim.lr_scheduler.StepLR(optimizer_D, 6, 0.3)
        lr_scheduler_e = torch.optim.lr_scheduler.StepLR(optimizer_E, 6, 0.3)
        lr_scheduler_f = torch.optim.lr_scheduler.StepLR(optimizer_F, 5, 0.3)
    lr_scheduler = [lr_scheduler_g, lr_scheduler_d, lr_scheduler_e, lr_scheduler_f]

    for epoch in range(resume_epoch, total_epoch):
        criterion_adv = torch.nn.BCELoss().to(device=config.device)
        criterion_pix = torch.nn.L1Loss(reduction='sum').to(device=config.device)
        criterion_cls = nn.L1Loss().to(device=config.device)
        criterion_fusion = nn.CrossEntropyLoss().to(device=config.device)
        criterion = [criterion_adv, criterion_pix, criterion_cls, criterion_fusion]

        if config.pvd_net.train.iter_train:
            train(train_loader, net, criterion, optimizer, lr_scheduler, epoch)
            # print('Generator_lr:\t' + str(lr_scheduler_g.get_lr()), '\n', 'Discriminator_lr:\t' + str(lr_scheduler_d.get_lr()), '\n', 'Adv_Classifier_lr:\t' + str(lr_scheduler_e.get_lr()), '\n',
            #       'Fusion_Classifier_lr:\t' + str(lr_scheduler_c.get_lr()))
            epoch_pc_view_dp += 1
        else:
            train(train_loader, net, criterion, optimizer, lr_scheduler, epoch)
            # print('Generator_lr:\t' + str(lr_scheduler_g.get_lr()), '\n', 'Discriminator_lr:\t' + str(lr_scheduler_d.get_lr()), '\n', 'Adv_Classifier_lr:\t' + str(lr_scheduler_e.get_lr()), '\n',
            #       'Fusion_Classifier_lr:\t' + str(lr_scheduler_c.get_lr()))
            epoch_pc_view_dp += 1

        with torch.no_grad():
            prec1, retrieval_map = validate(val_loader, net, epoch)

        # save checkpoints
        if best_prec1 < prec1:
            best_prec1 = prec1
            save_ckpt(epoch, epoch_pc, epoch_pc_view_dp, best_prec1, net, optimizer)
        if best_map < retrieval_map:
            best_map = retrieval_map

        print('curr accuracy: ', prec1)
        print('best accuracy: ', best_prec1)
        print('best map: ', best_map)

    print('Train Finished!')


if __name__ == '__main__':
    main()
