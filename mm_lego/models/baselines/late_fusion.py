# import snn and amil
from mm_lego.models.baselines.mcat import SNN, MILAttentionNet, BilinearFusion
import torch.nn as nn
from typing import *
import torch
import tensorly as tl


class Ensemble(nn.Module):
    """
    Takes in two encoders and predicts average of logits
    """
    def __init__(self, encoders: List[nn.Module]):
        super().__init__()
        self.encoders = nn.ModuleList(encoders)
    def forward(self, x: List[torch.Tensor]) -> torch.Tensor:
        try:
            logits = [encoder(x[i]) for i, encoder in enumerate(self.encoders)]
        except:
            logits = [encoder(x[i].unsqueeze(0)) for i, encoder in enumerate(self.encoders)]
        avg_logits = torch.mean(torch.stack(logits), dim=0)
        return avg_logits


class LateFusion(nn.Module):

    def __init__(self, input_dims: Tuple, method: str = "concat", n_classes: int=4,
                 dropout=0.25, **kwargs):
        super().__init__()
        tab_dims, img_dims = input_dims

        self.method = method
        self.tab_enc = SNN(input_dim = tab_dims, final_head=False)
        self.img_enc = MILAttentionNet(img_dims, final_head=False,
                                       size_arg="tcga")
        # print(self.img_enc)

        # get dims from tabular encoder (note to pass in as list)
        tab_out = self.tab_enc([torch.randn(1, tab_dims)]).squeeze()
        img_out = self.img_enc([torch.randn(1, *img_dims)]).squeeze()

        if self.method == "concat":
            self.head = nn.Sequential(*[nn.Linear(in_features=tab_out.shape[0] + img_out.shape[0], out_features=n_classes)])
        elif self.method == "bilinear":
            self.fusion = BilinearFusion(dim1=tab_out.shape[0], dim2=img_out.shape[0], scale_dim1=8, scale_dim2=8, mmhid=256)
            self.head = nn.Linear(in_features=256, out_features=n_classes)



        print(tab_out.shape, img_out.shape)


    def forward(self, x: List[torch.Tensor]) -> torch.Tensor:
        tab, img = x
        h_tab = self.tab_enc([tab]).squeeze()
        h_img = self.img_enc([img]).squeeze()

        if self.method == "concat":
            h = torch.cat([h_tab, h_img], dim=1)
        if self.method == "bilinear":
            h = self.fusion(h_tab, h_img)

        return self.head(h)



if __name__ == "__main__":
    b = 10
    # tabular
    # Note - dimensions always denoted as
    t_c = 1  # number of channels (1 for tabular) ; note that channels correspond to modality input/features
    t_d = 2189  # dimensions of each channel
    i_c = 100  # number of patches
    i_d = 2048  # dimensions per patch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tabular_data = torch.randn(b, t_c, t_d).to(device)
    image_data = torch.randn(b, i_c, i_d).to(device)

    lf = LateFusion(input_dims=(tabular_data.shape[2], image_data.shape[1:]), method="bilinear")
    lf.to(device)
    lf([tabular_data, image_data])

    # print(lf)


