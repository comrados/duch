import os

from utils import read_json, write_hdf5, get_image_file_names, shuffle_file_names_list, reshuffle_embeddings
from configs.config_img_aug import cfg

import torch
import torchvision
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from PIL import Image
import random


class ImagesDataset(torch.utils.data.Dataset):
    """
    Dataset for images
    """

    def __init__(self, image_file_names, images_folder, img_transforms_dicts, img_aug_set):
        self.image_file_names = image_file_names
        self.images_folder = images_folder
        self.img_transforms_dicts = img_transforms_dicts
        self.img_aug_set = img_aug_set
        self.img_transforms = self.init_transforms()
        self.to_tensor = torchvision.transforms.ToTensor()

    def __getitem__(self, idx):

        img = self.load_single_image(self.image_file_names[idx])

        # if only one sequence in self.img_transforms - it will be always applied
        transform = random.choice(self.img_transforms)

        img_aug = transform(img)

        return idx, img_aug, self.to_tensor(img)

    def __len__(self):
        return len(self.image_file_names)

    def load_single_image(self, img_name):
        """
        Load single image from the disc by name.

        :return: PIL.Image array
        """
        return Image.open(os.path.join(self.images_folder, img_name))

    def init_transforms(self):
        """
        Initialize transforms.

        :return: list of transforms sequences (may be only one), one will be selected and applied to image
        """
        if self.img_aug_set == 'each_img_random':
            return self.init_transforms_random()
        else:
            return [self.init_transforms_not_random(self.img_transforms_dicts[self.img_aug_set])]

    def init_transforms_random(self):
        """
        Initialize transforms randomly from transforms dictionary.

        :return: list of transforms sequences, one will be selected and applied to image
        """
        transforms = []
        for ts in self.img_transforms_dicts[self.img_aug_set]:
            transforms.append(self.init_transforms_not_random(self.img_transforms_dicts[ts]))
        return transforms

    @staticmethod
    def init_transforms_not_random(transform_dict):
        """
        Initialize transforms non-randomly from transforms dictionary.
        :param transform_dict: transforms dictionary from config file

        :return: sequence of transforms to apply to each image
        """
        def _rotation_transform(values):
            return torchvision.transforms.RandomChoice([torchvision.transforms.RandomRotation(val) for val in values])

        def _affine_transform(values):
            return torchvision.transforms.RandomChoice([torchvision.transforms.RandomAffine(val) for val in values])

        def _gaussian_blur_transform(values):
            return torchvision.transforms.GaussianBlur(*values)

        def _center_crop_transform(values):
            return torchvision.transforms.CenterCrop(values)

        def _random_crop_transform(values):
            return torchvision.transforms.RandomCrop(values)

        def _color_jittering(values):
            return torchvision.transforms.ColorJitter(0.8 * values, 0.8 * values, 0.8 * values, 0.2 * values)

        image_transform_funcs = {'rotation': _rotation_transform,
                                 'affine': _affine_transform,
                                 'blur': _gaussian_blur_transform,
                                 'center_crop': _center_crop_transform,
                                 'random_crop': _random_crop_transform,
                                 'jitter': _color_jittering}

        transforms_list = []

        for k, v in transform_dict.items():
            transforms_list.append(image_transform_funcs[k](v))

        transforms_list.append(torchvision.transforms.Resize(224))
        transforms_list.append(torchvision.transforms.ToTensor())
        return torchvision.transforms.Compose(transforms_list)


def get_embeddings(model, dataloader, device):
    """
    Get Embeddings

    :param model: model
    :param dataloader: data loader with images
    :param device: CUDA device
    :return:
    """
    with torch.no_grad():  # no need to call Tensor.backward(), saves memory
        model = model.to(device)  # to gpu (if presented)

        batch_outputs = []

        for idx, x, _ in tqdm(dataloader, desc='Getting Embeddings (batches): '):
            x = x.to(device)
            batch_outputs.append(model(x))

        output = torch.vstack(batch_outputs)  # (batches, batch_size, output_dim) -> (batches * batch_size, output_dim)

        embeddings = output.squeeze().cpu().numpy()  # return to cpu (or do nothing), convert to numpy
        print('Embeddings shape:', embeddings.shape)
        return embeddings


def get_resnet_model_for_embedding(model=None):
    """
    Remove the last layer to get embeddings

    :param model: pretrained model (optionally)

    :return: pretrained model without last (classification layer)
    """
    if model is None:
        model = torchvision.models.resnet18(pretrained=True)
    model = torch.nn.Sequential(*(list(model.children())[:-1]))
    return model


def show_rand_imgs(dataset):
    # visualization
    def imshow(imgs):
        fig = plt.figure(figsize=(len(imgs) * 3, 6))
        for i, img in enumerate(imgs):
            ax1 = fig.add_subplot(2, len(imgs), len(imgs) + i + 1)
            npimg1 = img[1].numpy()
            plt.imshow(np.transpose(npimg1, (1, 2, 0)))
            plt.axis('off')

            ax2 = fig.add_subplot(2, len(imgs), i + 1)
            npimg2 = img[2].numpy()
            plt.imshow(np.transpose(npimg2, (1, 2, 0)))
            plt.axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join('plots', 'tmp.png'))

    import matplotlib.pyplot as plt
    choice = np.random.choice(len(dataset), 10)
    chosen = [dataset[c] for c in choice]
    imshow(chosen)


if __name__ == '__main__':
    print("CREATE AUGMENTED IMAGE EMBEDDINGS")
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    # device
    device = torch.device(cfg.cuda_device if torch.cuda.is_available() else "cpu")

    # read captions from JSON file
    data = read_json(cfg.dataset_json_file)

    # get file names
    file_names = get_image_file_names(data)

    # shuffle images to avoid errors caused by batch normalization layer in ResNet18 (batch size shall also be big)
    file_names_permutated, permutations = shuffle_file_names_list(file_names)

    # create dataset and dataloader
    images_dataset = ImagesDataset(file_names_permutated,
                                   cfg.dataset_image_folder_path,
                                   cfg.image_aug_transform_sets, cfg.img_aug_set)
    # show_rand_imgs(images_dataset)
    images_dataloader = DataLoader(images_dataset, batch_size=cfg.image_emb_batch_size, shuffle=False)

    # load pretrained ResNet without last (classification layer)
    # resnet = get_resnet_model_for_embedding(torch.load(os.path.join('fine_tuned', 'resnet18ft.pth')))
    resnet = get_resnet_model_for_embedding()

    embeddings = get_embeddings(resnet, images_dataloader, device)

    # return embeddings back to original order of images
    embeddings_orig_order = reshuffle_embeddings(embeddings, permutations)

    # save embeddings
    write_hdf5(cfg.image_emb_aug_file, embeddings_orig_order.astype(np.float32), 'image_emb')

    print("DONE\n\n\n")
