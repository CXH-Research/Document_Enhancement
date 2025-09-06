import os
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.optim as optim
from accelerate import Accelerator
from torch.utils.data import DataLoader
from torchmetrics.functional import peak_signal_noise_ratio, structural_similarity_index_measure
from tqdm import tqdm

from config import Config
from data import get_training_data, get_test_data
from models import model_registry
from utils import seed_everything, save_checkpoint

opt = Config('config.yml')

seed_everything(opt.OPTIM.SEED)

os.makedirs(opt.TRAINING.SAVE_DIR, exist_ok=True)

def train():
    # Accelerate
    accelerator = Accelerator(log_with='wandb') if opt.OPTIM.WANDB else Accelerator()

    config = {
        "dataset": opt.TRAINING.TRAIN_DIR,
        "model": opt.MODEL.SESSION
    }
    accelerator.init_trackers("halation", config=config)
    loss_mse = torch.nn.MSELoss()

    # Data Loader
    train_dir = os.path.join('../dataset', opt.MODEL.SESSION, 'train')
    val_dir = os.path.join('../dataset', opt.MODEL.SESSION, 'test')

    train_dataset = get_training_data(train_dir, opt.MODEL.TARGET, img_options={'w': opt.TRAINING.PS_W, 'h': opt.TRAINING.PS_H})
    trainloader = DataLoader(dataset=train_dataset, batch_size=opt.OPTIM.BATCH_SIZE, shuffle=True, num_workers=16, drop_last=False, pin_memory=True)
    val_dataset = get_test_data(val_dir, opt.MODEL.TARGET, img_options={'w': opt.TRAINING.PS_W, 'h': opt.TRAINING.PS_H})
    testloader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=False, num_workers=8, drop_last=False, pin_memory=True)

    # Model
    model = model_registry.get(opt.MODEL.MODEL_NAME)()

    # Optimizer & Scheduler
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=opt.OPTIM.LR_INITIAL,
                            betas=(0.9, 0.999), eps=1e-8)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, opt.OPTIM.NUM_EPOCHS, eta_min=opt.OPTIM.LR_MIN)

    trainloader, testloader = accelerator.prepare(trainloader, testloader)
    model = accelerator.prepare(model)
    optimizer, scheduler = accelerator.prepare(optimizer, scheduler)

    start_epoch = 1
    best_psnr = 0
    best_epoch = 0

    size = len(testloader)

    # training
    for epoch in range(start_epoch, opt.OPTIM.NUM_EPOCHS + 1):
        model.train()

        for _, data in enumerate(tqdm(trainloader)):
            inp = data[0].contiguous()
            tar = data[1]

            # forward
            optimizer.zero_grad()
            res = model(inp).clamp(0, 1)

            train_loss = loss_mse(res, tar) + 0.2 * (1 - structural_similarity_index_measure(res, tar, data_range=1))

            # backward
            accelerator.backward(train_loss)
            optimizer.step()

        scheduler.step()

        # testing
        if epoch % opt.TRAINING.VAL_AFTER_EVERY == 0:
            model.eval()
            with torch.no_grad():
                psnr = 0
                ssim = 0
                for _, test_data in enumerate(tqdm(testloader)):
                    inp = test_data[0].contiguous()
                    tar = test_data[1]

                    res = model(inp).clamp(0, 1)
                    all_res, all_tar = accelerator.gather((res, tar))
                    psnr += peak_signal_noise_ratio(all_res, all_tar, data_range=1)
                    ssim += structural_similarity_index_measure(all_res, all_tar, data_range=1)

                psnr /= size
                ssim /= size

                if psnr > best_psnr:
                    # save model
                    best_psnr = psnr
                    best_epoch = epoch
                    save_checkpoint({
                        'epoch': epoch,
                        'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                    }, epoch, opt.TRAINING.SAVE_DIR, opt.MODEL.MODEL_NAME, opt.MODEL.SESSION)

                accelerator.log({
                    "PSNR": psnr,
                    "SSIM": ssim,
                }, step=epoch)

                print("epoch: {}, PSNR: {}, SSIM: {}, best PSNR: {}, best epoch: {}".format(epoch, psnr, ssim, best_psnr, best_epoch))

    accelerator.end_training()


if __name__ == '__main__':
    train()