import sys; print('Python %s on %s' % (sys.version, sys.platform))
sys.path.extend(['C:\\Users\\ae20067\\PycharmProjects\\CoralWORMS'])

import torch

import albumentations as A
import albumentations.augmentations.functional as F
import matplotlib.pyplot as plt
from albumentations.pytorch import ToTensorV2
import numpy as np
import ternausnet.models
from tqdm import tqdm
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim
from torch.utils.data import Dataset, DataLoader
import os
from Utils_Windows.classes import *
from datetime import datetime

cudnn.benchmark = True
print(f"PyTorch version: {torch.__version__}")


def train(train_loader, model, criterion, optimizer, epoch, params):
    metric_monitor = MetricMonitor()
    model.train()
    stream = tqdm(train_loader)
    for i, (images, target) in enumerate(stream, start=1):
        images = images.to(params["device"], non_blocking=True)
        target = target.to(params["device"], non_blocking=True)
        output = model(images).squeeze(1)
        loss = criterion(output, target)
        metric_monitor.update("Loss", loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        stream.set_description(
            "Epoch: {epoch}. Train.      {metric_monitor}".format(epoch=epoch, metric_monitor=metric_monitor)
        )


def validate(val_loader, model, criterion, epoch, params):
    metric_monitor = MetricMonitor()
    model.eval()
    stream = tqdm(val_loader)
    with torch.no_grad():
        for i, (images, target) in enumerate(stream, start=1):
            images = images.to(params["device"], non_blocking=True)
            target = target.to(params["device"], non_blocking=True)
            output = model(images).squeeze(1)
            loss = criterion(output, target)
            metric_monitor.update("Loss", loss.item())
            stream.set_description(
                "Epoch: {epoch}. Validation. {metric_monitor}".format(epoch=epoch, metric_monitor=metric_monitor)
            )


def create_model(params):
    model = getattr(ternausnet.models, params["model"])(pretrained=True)
    model = model.to(params["device"])
    return model


def train_and_validate(model, train_dataset, val_dataset, params):
    train_loader = DataLoader(
        train_dataset,
        batch_size=params["batch_size"],
        shuffle=True,
        num_workers=params["num_workers"],
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=params["batch_size"],
        shuffle=False,
        num_workers=params["num_workers"],
        pin_memory=True,
    )
    criterion = nn.BCEWithLogitsLoss().to(params["device"])
    optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"])

    for epoch in range(1, params["epochs"] + 1):
        train(train_loader, model, criterion, optimizer, epoch, params)
        validate(val_loader, model, criterion, epoch, params)

    return model


def predict(model, params, test_dataset, batch_size):
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=params["num_workers"], pin_memory=True,
    )
    model.eval()
    predictions = []
    with torch.no_grad():
        for images, (original_heights, original_widths) in test_loader:
            images = images.to(params["device"], non_blocking=True)
            output = model(images)
            probabilities = torch.sigmoid(output.squeeze(1))
            predicted_masks = (probabilities >= 0.5).float() * 1
            predicted_masks = predicted_masks.cpu().numpy()
            for predicted_mask, original_height, original_width in zip(
                    predicted_masks, original_heights.numpy(), original_widths.numpy()
            ):
                predictions.append((predicted_mask, original_height, original_width))
    return predictions


params = {
    "model": "UNet11",
    "device": "cuda",
    "lr": 0.001,
    "batch_size": 16,
    "num_workers": 0,
    "epochs": 20,
}

# Set the device
if __name__ == "__main__":

    # defining directories
    dataset_directory = 'E:\\CoralReefRock1'
    root_directory = os.path.join(dataset_directory)
    images_directory = os.path.join(root_directory, "TRAINING_DATA\\PATCHES\\RAW_AUG_CROPPED")
    masks_directory = os.path.join(root_directory, "TRAINING_DATA\\PATCHES\\LABELS_AUG_CROPPED")

    images_filenames = list(sorted(os.listdir(images_directory)))

    np.random.seed(42)
    np.random.shuffle(images_filenames)

    # doing splits 60 30 10
    n_train = int(len(images_filenames) * 0.6)
    n_test = int(len(images_filenames) * 0.1)

    train_images_filenames = images_filenames[:n_train]
    val_images_filenames = images_filenames[n_train:-n_test]
    test_images_filenames = images_filenames[-n_test:]

    # print(len(train_images_filenames), len(val_images_filenames), len(test_images_filenames))

    train_transform = A.Compose(
        [
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])

    val_transform = A.Compose(
        [A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), ToTensorV2()])

    train_dataset = WormDataset(train_images_filenames, images_directory, masks_directory, transform=train_transform, )
    val_dataset = WormDataset(val_images_filenames, images_directory, masks_directory, transform=val_transform, )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    print(f"Using device: {device}")

    model = create_model(params)
    model = train_and_validate(model, train_dataset, val_dataset, params)
    model_save_dir = 'E:\\CoralReefRock1\\MODELS'
    currentDateAndTime = datetime.now().strftime("%Y%m%d-%H%M%S")
    model_name = 'model-' + currentDateAndTime + '.pth'
    torch.save(model.state_dict(), os.path.join(model_save_dir, model_name))

    # TODO import model here from saved state
    model2 = create_model(params)
    model2.load_state_dict(torch.load(os.path.join(model_save_dir, model_name)))

    test_transform = A.Compose(
        [A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), ToTensorV2()])

    test_dataset = InferenceWorm(test_images_filenames, images_directory, transform=test_transform, )

    predictions = predict(model2, params, test_dataset, batch_size=16)
    pred_dir = os.path.join(dataset_directory, 'PREDICTIONS_TEST')
    os.makedirs(pred_dir, exist_ok=True)

    for indx in range(0, len(predictions)):
        original_name = test_dataset.images_filenames[indx]
        original_img = cv2.imread(os.path.join(test_dataset.images_directory, test_dataset.images_filenames[indx]),
                                  cv2.IMREAD_GRAYSCALE)
        pred_data = predictions[indx][0]
        uint_img = np.array(pred_data * 255).astype('uint8')
        pred_name = original_name.split('.tif')[0] + '_PRED.png'

        if np.max(pred_data) > 0:
            print(f"the image {original_name}' with index {indx} has worms in it)")

            # saving cropped augmented
            cv2.imwrite(os.path.join(pred_dir, original_name), original_img)
            cv2.imwrite(os.path.join(pred_dir, pred_name), uint_img)
            print(f"the image {original_name}' with index {indx} has worms in it)")

            # saving cropped augmented
            cv2.imwrite(os.path.join(pred_dir, original_name), original_img)
            cv2.imwrite(os.path.join(pred_dir, pred_name), uint_img)

            # plt.imshow(original_img)
            # plt.show()
            # plt.imshow(pred_data)
            # plt.show()
