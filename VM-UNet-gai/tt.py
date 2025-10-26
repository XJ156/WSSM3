import torch
from torch.utils.data import DataLoader
import timm
from datasets.dataset import NPY_datasets
from tensorboardX import SummaryWriter
from models.vmunet.vmunet import VMUNet

from engine import *
import os
import sys

from utils import *
from configs.config_setting import setting_config

import warnings

warnings.filterwarnings("ignore")


def main(config):
    print('#----------Creating logger----------#')
    sys.path.append(config.work_dir + '/')
    log_dir = os.path.join(config.work_dir, 'log')
    # checkpoint_dir = os.path.join(config.work_dir, 'checkpoints')
    # resume_model = os.path.join(checkpoint_dir, 'latest.pth')
    # outputs = os.path.join(config.work_dir, 'outputs')
    # if not os.path.exists(checkpoint_dir):
    #     os.makedirs(checkpoint_dir)
    # if not os.path.exists(outputs):
    #     os.makedirs(outputs)
    #
    global logger
    logger = get_logger('train', log_dir)
    # global writer
    # writer = SummaryWriter(config.work_dir + 'summary')
    #
    # log_config_info(config, logger)

    print('#----------GPU init----------#')
    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_id
    set_seed(config.seed)
    torch.cuda.empty_cache()

    print('#----------Preparing dataset----------#')
    train_dataset = NPY_datasets(config.data_path, config, train=True)
    train_loader = DataLoader(train_dataset,
                              batch_size=config.batch_size,
                              shuffle=True,
                              pin_memory=True,
                              num_workers=config.num_workers)
    val_dataset = NPY_datasets(config.data_path, config, train=False)
    val_loader = DataLoader(val_dataset,
                            batch_size=1,
                            shuffle=False,
                            pin_memory=True,
                            num_workers=config.num_workers,
                            drop_last=True)

    print('#----------Prepareing Model----------#')
    model_cfg = config.model_config
    if config.network == 'vmunet':
        model = VMUNet(
            num_classes=model_cfg['num_classes'],
            input_channels=model_cfg['input_channels'],
            depths=model_cfg['depths'],
            depths_decoder=model_cfg['depths_decoder'],
            drop_path_rate=model_cfg['drop_path_rate'],
            load_ckpt_path=model_cfg['load_ckpt_path'],
        )
        model.load_from()

    else:
        raise Exception('network in not right!')
    model = model.cuda()

    cal_params_flops(model, 512, logger)
    # cal_params_flops(model, 256, logger)

    print('#----------Prepareing loss, opt, sch and amp----------#')
    criterion = config.criterion
    optimizer = get_optimizer(config, model)
    scheduler = get_scheduler(config, optimizer)

    print('#----------Set other params----------#')
    min_loss = 999
    start_epoch = 1
    min_epoch = 1



    # if os.path.exists(os.path.join(checkpoint_dir, 'best.pth')):
    print('#----------Testing----------#')
    best_weight = torch.load('/disk/sdb/zhangqian0620/VM-UNet-main/results/vmunet_olp_box_Tuesday_30_April_2024_12h_18m_50s/checkpoints/best-epoch274-loss0.7143.pth', map_location=torch.device('cpu'))
    model.load_state_dict(best_weight)
    loss = test(
        val_loader,
        model,
        criterion,
        logger,
        config,
    )
    # os.rename(
    #     os.path.join(checkpoint_dir, 'best.pth'),
    #     os.path.join(checkpoint_dir, f'best-epoch{min_epoch}-loss{min_loss:.4f}.pth')
    # )

def test(test_loader,
                    model,
                    criterion,
                    config,
                    output_dir='output_olp/'):
    # 切换到评估模式
    model.eval()
    preds = []
    gts = []
    loss_list = []
    # if not os.path.exists(output_dir):
    #     os.makedirs(output_dir)
    with torch.no_grad():
        for i, data in enumerate(tqdm(test_loader)):

            img, msk,name = data
            # print(name)
            b=name[0]

            img, msk = img.cuda(non_blocking=True).float(), msk.cuda(non_blocking=True).float()
            out = model(img)
            loss = criterion(out, msk)
            loss_list.append(loss.item())
            msk = msk.squeeze(1).cpu().detach().numpy()
            gts.append(msk)
            if type(out) is tuple:
                out = out[0]
            out = out.squeeze(1).cpu().detach().numpy()
            preds.append(out)


            save(img, msk, out, i, b,'/disk/sdb/zhangqian0620/VM-UNet-main/output_olp_300_test')


def save(img, msk, msk_pred, i, b, save_path):


    img = img.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
    img = img / 255. if img.max() > 1.1 else img
    # if datasets == 'retinal':
    #     msk = np.squeeze(msk, axis=0)
    #     msk_pred = np.squeeze(msk_pred, axis=0)
    # else:
    #     msk = np.where(np.squeeze(msk, axis=0) > 0.5, 1, 0)
    #     msk_pred = np.where(np.squeeze(msk_pred, axis=0) > threshold, 1, 0)

    msk = np.where(np.squeeze(msk, axis=0) > 0.5, 1, 0)
    msk_pred = np.where(np.squeeze(msk_pred, axis=0) > 0.5, 1, 0)

    from PIL import Image

    # 将numpy数组转换为PIL图像
    image = Image.fromarray((msk_pred * 255).astype(np.uint8))

    # 保存图像
    image.save(os.path.join(save_path, b))


if __name__ == '__main__':
    config = setting_config
    main(config)