import sys

sys.path.append('c:\\users\\ctlablovelace\\appdata\\local\\programs\\python\\python39\lib\\site-packages')
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
from torch.utils.tensorboard import SummaryWriter
from sklearn import metrics
import pandas as pd

cudnn.benchmark = True
print(f"PyTorch version: {torch.__version__}")


def train(train_loader, model, criterion, optimizer, epoch, params):
    metric_monitor = MetricMonitor()
    model.train()
    stream = tqdm(train_loader)
    running_loss = 0
    batch_size = params['batch_size']

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

    running_loss = metric_monitor.metrics["Loss"]['avg']

    # Log the training loss averaged per batch
    writer.add_scalars('Test Worms 3',
                       {'Training Loss': running_loss},
                       epoch)
    writer.flush()


def validate(val_loader, model, criterion, epoch, params):
    metric_monitor = MetricMonitor()
    model.eval()
    stream = tqdm(val_loader)
    vloss = 0
    batch_size = params['batch_size']

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

    vloss = metric_monitor.metrics["Loss"]['avg']

    # Log the training loss averaged per batch
    writer.add_scalars('Test Worms 3',
                       {'Validation Loss': vloss},
                       epoch)
    writer.flush()


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

    # # TODO Visualize the model
    # # grab a single mini-batch of images
    # dataiter = iter(train_loader)
    # images, labels = next(dataiter)
    # images, labels = images.to(device), labels.to(device)
    # # add_graph() will trace the sample input through your model,
    # # and render it as a graph.
    # model.eval()
    # writer.add_graph(model, images)
    # writer.flush()


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
    dataset_directory = 'E:\\CoralWORMS'
    root_directory = os.path.join(dataset_directory)
    images_directory = os.path.join(root_directory, "TRAINING_DATA\\PATCHES\\RAW_AUG_CROPPED")
    masks_directory = os.path.join(root_directory, "TRAINING_DATA\\PATCHES\\LABELS_AUG_CROPPED")
    writer = SummaryWriter('E:\\CoralWORMS\\MODELS\\runs\\')

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
    model_save_dir = "E:\\CoralWORMS\\MODELS"
    currentDateAndTime = datetime.now().strftime("%Y%m%d-%H%M%S")
    model_name = 'model-dummy' + currentDateAndTime + '.pth'
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

    DATA_FRAME = pd.DataFrame({"Image_Patch": [],
                  "Precision": [],
                  "Recall": [],
                  "Accuracy1": [],
                  "Accuracy2": [],
                  "F1_score": []}
                 )

    Image_Patch_names = []
    Precision = []
    Recall = []
    Accuracy1 = []
    Accuracy2 = []
    F1 = []
    gt_img_all = []
    pred_data_all = []

    for indx in range(0, len(predictions)):
        original_name = test_dataset.images_filenames[indx]
        original_img = cv2.imread(os.path.join(test_dataset.images_directory, test_dataset.images_filenames[indx]),
                                  cv2.IMREAD_GRAYSCALE)
        pred_data = predictions[indx][0]
        uint_img = np.array(pred_data * 255).astype('uint8')
        pred_name = original_name.split('.tif')[0] + '_PRED.png'

        gt_img_name = original_name.split('raw.tif')[0] + 'label.png'
        gt_img = cv2.imread(os.path.join(test_dataset.images_directory.replace('RAW', 'LABELS'), gt_img_name),
                            cv2.IMREAD_GRAYSCALE)
        gt_img = np.array(gt_img / 255).astype('int8')

        Image_Patch_names.append(original_name)

        #print('F1: {}'.format(metrics.f1_score(gt_img, pred_data, average="weighted", zero_division=0)))
        F1.append(metrics.f1_score(gt_img, pred_data , average="weighted", zero_division=0))

        #print('Precision: {}'.format(metrics.precision_score(gt_img, pred_data, average="weighted", zero_division=0)))
        Precision.append(metrics.precision_score(gt_img, pred_data, average="weighted", zero_division=0))

        #print('Recall: {}'.format(metrics.recall_score(gt_img, pred_data, average="weighted", zero_division=0)))
        Recall.append(metrics.recall_score(gt_img, pred_data, average="weighted", zero_division=0))

        gt_unique, gt_counts = np.unique(gt_img, return_counts=True)
        pred_unique, pred_counts = np.unique(pred_data, return_counts=True)

        #print('Accuracy: {}'.format(metrics.accuracy_score(gt_img, pred_data)))
        Accuracy1.append(metrics.accuracy_score(gt_img, pred_data))

        if len(gt_unique) >1 and len(pred_unique) >1:
            Accuracy2.append(np.round(pred_counts[1]/gt_counts[1],3))
        else:
            Accuracy2.append(-9999)

    DATA_FRAME = pd.DataFrame({"Image_Patch": Image_Patch_names,
                               "Precision": Precision,
                               "Recall": Recall,
                               "Accuracy1": Accuracy1,
                               "Accuracy12": Accuracy2,
                               "F1_score": F1}
                              )

    file_name = os.path.join(model_save_dir, 'Prediction_Metrics_'+model_name.split('.pth')[0]+'.xlsx')
    DATA_FRAME.to_excel(file_name)


        # if np.max(pred_data) > 0:
        #     print(f"the image {original_name}' with index {indx} has worms in it)")
        #
        #     # saving cropped augmented
        #     cv2.imwrite(os.path.join(pred_dir, original_name), original_img)
        #     cv2.imwrite(os.path.join(pred_dir, pred_name), uint_img)
        #     print(f"the image {original_name}' with index {indx} has worms in it)")
        #
        #     # saving cropped augmented
        #     cv2.imwrite(os.path.join(pred_dir, original_name), original_img)
        #     cv2.imwrite(os.path.join(pred_dir, pred_name), uint_img)
        #
        #     plt.imshow(original_img)
        #     plt.show()
        #     plt.imshow(pred_data)
        #     plt.show()
