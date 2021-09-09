import torchvision.models as models
import torch
from ptflops import get_model_complexity_info
from thop import profile
from torch import nn
from configs.config import cfg


def test_ptflops():
    net = models.resnet50()
    macs, params = get_model_complexity_info(net, (3, 224, 224), as_strings=False, print_per_layer_stat=False)
    print('{:<40}  {:<8}'.format('Computational complexity (MACs):', macs))
    print('{:<40}  {:<8}'.format('Number of parameters:', params))


def test_thop():
    model = models.resnet50()
    input = torch.randn(1, 3, 224, 224)
    macs, params = profile(model, inputs=(input,), verbose=False)
    print('{:<40}  {:<8}'.format('Computational complexity (MACs):', macs))
    print('{:<40}  {:<8}'.format('Number of parameters:', params))


def count_ptflops(model, inputs_dim, tag):
    macs, params = get_model_complexity_info(model, inputs_dim, as_strings=False, print_per_layer_stat=False, )
    print('{:<10} {:<40}  {:<8}'.format(tag.upper(), 'Computational complexity (MACs):', macs))
    print('{:<10} {:<40}  {:<8}'.format(tag.upper(), 'Number of parameters:', params))
    return macs, params


def count_thop(model, inputs_dim, tag):
    input = torch.randn((1,) + inputs_dim)
    macs, params = profile(model, inputs=(input,), verbose=False)
    print('{:<10} {:<40}  {:<8}'.format(tag.upper(), 'Computational complexity (MACs):', macs))
    print('{:<10} {:<40}  {:<8}'.format(tag.upper(), 'Number of parameters:', params))
    return macs, params


class UNHDI(nn.Module):

    def __init__(self, image_dim, text_dim, hidden_dim, hash_dim):
        super(UNHDI, self).__init__()
        self.module_name = 'UNHD'
        self.image_dim = image_dim
        self.text_dim = text_dim
        self.hidden_dim = hidden_dim
        self.hash_dim = hash_dim

        self.image_module = nn.Sequential(
            nn.Linear(image_dim, hidden_dim, bias=True),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, hash_dim, bias=True),
            nn.Tanh()
        )

    def forward(self, r_img):
        h_img = self.image_module(r_img).squeeze()
        return h_img


class UNHDT(nn.Module):

    def __init__(self, image_dim, text_dim, hidden_dim, hash_dim):
        super(UNHDT, self).__init__()
        self.module_name = 'UNHD'
        self.image_dim = image_dim
        self.text_dim = text_dim
        self.hidden_dim = hidden_dim
        self.hash_dim = hash_dim

        self.text_module = nn.Sequential(
            nn.Linear(text_dim, hidden_dim, bias=True),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, hash_dim, bias=True),
            nn.Tanh()
        )

    def forward(self, r_txt):
        h_txt = self.text_module(r_txt).squeeze()
        return h_txt


class UNHDD(nn.Module):

    def __init__(self, image_dim, text_dim, hidden_dim, hash_dim):
        super(UNHDD, self).__init__()
        self.module_name = 'UNHD'
        self.image_dim = image_dim
        self.text_dim = text_dim
        self.hidden_dim = hidden_dim
        self.hash_dim = hash_dim

        # hash discriminator
        self.hash_dis = nn.Sequential(
            nn.Linear(self.hash_dim, self.hash_dim * 2, bias=True),
            nn.BatchNorm1d(self.hash_dim * 2),
            nn.ReLU(True),
            nn.Linear(self.hash_dim * 2, 1, bias=True)
        )

    def forward(self, h):
        d = self.hash_dis(h).squeeze()
        return d


mi = UNHDI(cfg.image_dim, cfg.text_dim, cfg.hidden_dim, cfg.hash_dim)
mt = UNHDT(cfg.image_dim, cfg.text_dim, cfg.hidden_dim, cfg.hash_dim)
md = UNHDD(cfg.image_dim, cfg.text_dim, cfg.hidden_dim, cfg.hash_dim)

device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")

input_dims = (512,)
model = mi


def calculate_stats_for_unhd(method='ptflops'):
    if method == 'ptflops':
        f = count_ptflops
    else:
        f = count_thop

    print('\n\n\n' + method + '\n')
    print('Module stats:')
    macsi, paramsi = f(mi, (cfg.image_dim,), 'img')
    macst, paramst = f(mt, (cfg.text_dim,), 'txt')
    macsd, paramsd = f(md, (cfg.hash_dim,), 'disc')

    total_params = paramsi + paramst + paramsd
    total_macs = macsi * 2 + macst * 2 + macsd * 4  # img + aug_img = 2 hashes, txt + aug_txt = 2 hashes

    print('\nTotal stats:')
    print('{:<40}  {:<8}'.format('Computational complexity (MACs):', total_macs))
    print('{:<40}  {:<8}'.format('Computational complexity (FLOPs):', total_macs * 2))
    print('{:<40}  {:<8}'.format('Number of parameters:', total_params))


def calculate_stats():
    calculate_stats_for_unhd()
    calculate_stats_for_unhd('thop')


if device.type == 'cpu':
    # test_ptflops()
    # test_thop()
    calculate_stats()
else:
    with torch.cuda.device(device):
        # test_ptflops()
        # test_thop()
        calculate_stats()
