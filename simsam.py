import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def D(p, z, version='simplified'):  # negative cosine similarity
    if version == 'original':
        z = z.detach()  # stop gradient
        p = F.normalize(p, dim=1)  # l2-normalize
        z = F.normalize(z, dim=1)  # l2-normalize
        return -(p * z).sum(dim=1).mean()

    elif version == 'simplified':  # same thing, much faster. Scroll down, speed test in __main__
        return - F.cosine_similarity(p, z.detach(), dim=-1).mean()
    else:
        raise Exception


class projection_MLP(nn.Module):
    def __init__(self, in_dim=128, hidden_dim=2048, out_dim=2048):
        super().__init__()
        ''' page 3 baseline setting
        Projection MLP. The projection MLP (in f) has BN ap-
        plied to each fully-connected (fc) layer, including its out- 
        put fc. Its output fc has no ReLU. The hidden fc is 2048-d. 
        This MLP has 3 layers.
        '''
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer3 = nn.Sequential(
            nn.Linear(hidden_dim, out_dim),
            nn.BatchNorm1d(hidden_dim)
        )
        self.num_layers = 3

    def set_layers(self, num_layers):
        self.num_layers = num_layers

    def forward(self, x):
        if self.num_layers == 3:
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
        elif self.num_layers == 2:
            x = self.layer1(x)
            x = self.layer3(x)
        else:
            raise Exception
        return x


class prediction_MLP(nn.Module):
    def __init__(self, in_dim=2048, hidden_dim=512, out_dim=2048):  # bottleneck structure
        super().__init__()
        ''' page 3 baseline setting
        Prediction MLP. The prediction MLP (h) has BN applied 
        to its hidden fc layers. Its output fc does not have BN
        (ablation in Sec. 4.4) or ReLU. This MLP has 2 layers. 
        The dimension of h’s input and output (z and p) is d = 2048, 
        and h’s hidden layer’s dimension is 512, making h a 
        bottleneck structure (ablation in supplement). 
        '''
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Linear(hidden_dim, out_dim)
        """
        Adding BN to the output of the prediction MLP h does not work
        well (Table 3d). We find that this is not about collapsing. 
        The training is unstable and the loss oscillates.
        """

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x


class SimSiam(nn.Module):
    def __init__(self, ModelBase=None,dim = 128,args = None):
        super().__init__()

        self.backbone = ModelBase(*args)
        self.projector = projection_MLP(dim)

        self.encoder = nn.Sequential(  # f encoder
            self.backbone,
            self.projector
        )

        self.predictor = prediction_MLP()

    def forward(self, x1, x2 = None):
        if x2 is None:
            out,_ = self.backbone(x1)
            return out
        f, p, h = self.backbone,self.projector, self.predictor
        out,z1 = f(x1)
        _, z2 = f(x2)
        z1,z2 = p(z1),p(z2)
        p1, p2 = h(z1), h(z2)
        L = D(p1, z2) / 2 + D(p2, z1) / 2
        return out,L.mean()



