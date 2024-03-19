import torch
from torch.utils.data import DataLoader
from utils.dataset import BuildingDataset
from model.HDNet import HighResolutionDecoupledNet
import matplotlib
import os
from tqdm import tqdm
import cv2

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

matplotlib.use('tkagg')
batchsize = 16
num_workers = 0
read_name = 'HDNet_Inria_best'
Dataset = 'Inria'
assert Dataset in ['WHU', 'Inria', 'Mass']
net = HighResolutionDecoupledNet(base_channel=48, num_classes=1)
print('Number of parameters: ', sum(p.numel() for p in net.parameters()))


def predict(net, device, batch_size, data_dir, weight_dir):
    dataset = BuildingDataset(
        dataset_dir=data_dir,
        training=False,
        txt_name="test.txt",
        data_name=Dataset)

    loader = DataLoader(dataset,
                        batch_size=batch_size,
                        shuffle=False,
                        num_workers=num_workers,
                        drop_last=False)

    for batch in tqdm(loader):
        imgs = batch['image']
        imgs = imgs.to(device=device, dtype=torch.float32)

        with torch.no_grad():
            pred = net(imgs)
        pred1 = (pred[0] > 0).float()
        label_pred = pred1.squeeze().cpu().int().numpy().astype('uint8') * 255
        for i in range(len(pred1)):
            img_name = batch['name'][i].split('/')[-1]
            save_path = os.path.join(data_dir, 'predictions')
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            # print('Saving to', os.path.join(save_path, img_name))
            wr = cv2.imwrite(os.path.join(save_path, img_name), label_pred[i])
            if not wr:
                print('Save failed!')


# data_dir = "data/Inria/"
# dir_checkpoint = 'save_weights/pretrain/'
# if __name__ == '__main__':
#     logging.basicConfig(level=logging.INFO,
#                         format='%(levelname)s: %(message)s')
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     logging.info(f'Using device {device}')
#     if read_name != '':
#         net_state_dict = net.state_dict()
#         state_dict = torch.load(
#             dir_checkpoint + read_name + '.pth', map_location=device)
#         net_state_dict.update(state_dict)
#         net.load_state_dict(net_state_dict, strict=False)  # 删除了down1-3
#         logging.info('Model loaded from ' + read_name + '.pth')

#     net = convert_model(net)
#     net = torch.nn.parallel.DataParallel(net.to(device))
#     torch.backends.cudnn.benchmark = True
#     predict(net=net,
#          batch_size=batchsize,
#          device=device,
#          data_dir=data_dir,
#          weight_dir=dir_checkpoint)
