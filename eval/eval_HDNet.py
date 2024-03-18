import torch
import numpy as np
from tqdm import tqdm
import cv2


def fast_hist(label_pred, label_true, num_classes):
    mask = (label_true >= 0) & (label_true < num_classes)
    hist = np.bincount(
        num_classes * label_true[mask].astype(int) +
        label_pred[mask], minlength=num_classes ** 2).reshape(num_classes,
                                                              num_classes)
    return hist


def mask_to_boundary(mask, dilation_ratio):
    """
    Convert binary mask to boundary mask.
    From https://github.com/bowenc0221/boundary-iou-api/
    :param mask (numpy array, uint8): binary mask
    :param dilation_ratio (float): ratio to calculate dilation =
        dilation_ratio * image_diagonal
    :return: boundary mask (numpy array)
    """

    h, w = mask.shape
    img_diag = np.sqrt(h ** 2 + w ** 2)
    dilation = int(round(dilation_ratio * img_diag))
    if dilation < 1:
        dilation = 1

    # Pad image so mask truncated by the image border is also considered as
    # boundary.
    new_mask = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT,
                                  value=0)
    kernel = np.ones((3, 3), dtype=np.uint8)
    new_mask_erode = cv2.erode(new_mask, kernel, iterations=dilation)
    mask_erode = new_mask_erode[1: h + 1, 1: w + 1]
    # G_d intersects G in the paper.
    return mask - mask_erode


def biou(label_pred, label_true, dilation_ratio):
    """
    Calculate Boundary Intersection over Union (BIOU) based on mask to
    boundary conversion.
    :param label_pred (numpy array): batch of predicted labels
    :param label_true (numpy array): batch of true labels
    :return: BIOU (float)
    """

    biou_tot = 0

    # Loop over each label in the batch
    for i in range(label_pred.shape[0]):
        # Convert masks to boundaries
        boundary_pred = mask_to_boundary(label_pred[i], dilation_ratio)
        boundary_true = mask_to_boundary(label_true[i], dilation_ratio)

        # Compute the intersection and union of the boundaries
        intersection = np.logical_and(boundary_pred, boundary_true)
        union = np.logical_or(boundary_pred, boundary_true)

        # Compute the BIOU for this label and add it to the total BIOU
        if np.sum(union) > 0:
            biou_tot += np.sum(intersection) / np.sum(union)
        else:
            biou_tot += 1

    return biou_tot / label_pred.shape[0]


def boundary_hist(label_pred, label_true, dilation_ratio, num_classes):

    boundary_hist = 0

    for i in range(label_pred.shape[0]):
        boundary_pred = mask_to_boundary(label_pred[i], dilation_ratio)
        boundary_true = mask_to_boundary(label_true[i], dilation_ratio)
        boundary_hist += fast_hist(boundary_pred.flatten(),
                                   boundary_true.flatten(),
                                   num_classes)

    return boundary_hist


def eval_net(net, loader, device, savename=''):
    """Evaluation without the densecrf with the dice coefficient"""
    net.eval()
    mask_type = torch.float32
    # n_val = len(loader)  # the number of batch
    hist = 0
    # biou_total = 0
    hist_boundary = 0

    for num, batch in enumerate(tqdm(loader)):
        imgs, true_labels = batch['image'], batch['label']
        imgs = imgs.to(device=device, dtype=torch.float32)
        true_labels = (true_labels > 0).to(device=device, dtype=mask_type)
        with torch.no_grad():
            pred = net(imgs)
        pred1 = (pred[0] > 0).float()
        hist += fast_hist(pred1.flatten().cpu().detach().int().numpy(),
                          true_labels.flatten().cpu().int().numpy(), 2)

        # calculate BIOU
        label_pred = pred1.squeeze().cpu().int().numpy().astype('uint8')
        label_true = true_labels.squeeze().cpu().int().numpy().astype('uint8')

        # biou_total += biou(label_pred, label_true, 0.01)
        hist_boundary += boundary_hist(label_pred, label_true, 0.005, 2)

    IOU = (np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) -
                            np.diag(hist)))[-1]
    acc_global_OA = np.diag(hist).sum() / hist.sum()
    acc_R = np.diag(hist) / hist.sum(1) * 100
    acc_P = np.diag(hist) / hist.sum(0) * 100
    F1score = 2 * acc_R * acc_P / (acc_R + acc_P)
    # avg_biou = biou_total / n_val  # calculate average BIOU
    BIOU = (np.diag(hist_boundary) / (hist_boundary.sum(axis=1) +
                                      hist_boundary.sum(axis=0) -
                                      np.diag(hist_boundary)))[-1]

    print()
    print('IOU:', IOU)
    print('OA:', acc_global_OA)
    print('Recall:', acc_R)
    print('Precision:', acc_P)
    print('F1_score:', F1score)
    # print('BIOU:', avg_biou)  # print average BIOU
    print('BIOU:', BIOU)  # print BIOU
    print('hist:', hist)
    print('hist_boundary', hist_boundary)

    return IOU
