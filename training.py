import torch
import torch.optim as optim
import torch.optim.lr_scheduler as scheduler
import torchvision
from torch.utils.data import DataLoader
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator

import External.utils as utils
from External.engine import train_one_epoch, evaluate

from datasets import TankDataset, get_transform
from eval import generate_predictions, generate_metrics

torch.manual_seed(42)


def get_model(n_class, device='cuda'):
    backbone = torchvision.models.mobilenet_v3_large(weights="DEFAULT").features
    backbone.out_channels = 960

    anchor_generator = AnchorGenerator(
        sizes=((32, 64, 128, 256),),
        aspect_ratios=((0.5, 1.0, 2.0),)
    )

    roi_pooler = torchvision.ops.MultiScaleRoIAlign(
        featmap_names=['0'],
        output_size=7,
        sampling_ratio=2
    )

    model = FasterRCNN(
        backbone,
        num_classes=n_class,
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=roi_pooler,
        box_score_thresh=0.75,
    )

    model.to(device)
    return model


def main():
    val_size = 0.10
    test_size = 0.20

    dataset_train = TankDataset('./Data/Oil Tanks/image_patches',
                                './Data/Oil Tanks/labels.json',
                                data_cutoffs=(0.0, 1.0 - val_size - test_size),
                                img_dup=4, transform=get_transform(train=True))

    dataset_val = TankDataset('./Data/Oil Tanks/image_patches',
                              './Data/Oil Tanks/labels.json',
                              data_cutoffs=(1.0 - val_size - test_size, 1.0 - test_size),
                              transform=get_transform(train=False))

    dataset_test = TankDataset('./Data/Oil Tanks/image_patches',
                               './Data/Oil Tanks/labels.json',
                               data_cutoffs=(1.0 - test_size, 1.0),
                               transform=get_transform(train=False))

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    data_loader_train = DataLoader(
        dataset_train,
        batch_size=6,
        shuffle=True,
        num_workers=5,
        collate_fn=utils.collate_fn
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=6,
        shuffle=False,
        num_workers=5,
        collate_fn=utils.collate_fn
    )

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=6,
        shuffle=False,
        num_workers=5,
        collate_fn=utils.collate_fn
    )

    model = get_model(n_class=2)
    params = [p for p in model.parameters() if p.requires_grad]

    optimizer = optim.Adam(
        params,
        lr=0.00075
    )

    lr_scheduler = scheduler.StepLR(
        optimizer,
        step_size=2,
        gamma=0.2
    )

    GEN_PREDICTIONS = False
    GEN_METRICS = True
    N_EPOCH = 0
    LOAD_STATE = True
    LOAD_DIR = './Saved States/state_dict4.pt'

    if LOAD_STATE:
        model.load_state_dict(torch.load(LOAD_DIR))

    for epoch in range(N_EPOCH):
        # train for one epoch, printing every 12 iterations
        train_one_epoch(model, optimizer, data_loader_train, device, epoch, print_freq=15)

        # update the learning rate
        lr_scheduler.step()
        # save model
        torch.save(model.state_dict(), f'./Saved States/state_dict{epoch}.pt')
        # evaludate on validation set
        evaluate(model, data_loader_val, device=device)

    if GEN_PREDICTIONS:
        generate_predictions(model, data_loader_test, './Results/Predictions/')
    if GEN_METRICS:
        stats = evaluate(model, data_loader_test, device=device)
        generate_metrics(stats, './Results/Metrics/')


if __name__ == '__main__':
    main()
