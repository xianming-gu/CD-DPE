import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions.normal import Normal
from Module.FFEM_module import DecoderLayer, SparseDispatcher
from Module.FDM_module import ista_unet
import numpy as np
from Options_CDDPE import args


class CDDPE(nn.Module):  # 2-2
    def __init__(self, image_size=256, in_chans=1, emb_dim=64):
        super(CDDPE, self).__init__()
        self.encoder = DecoupleModule(in_chans, emb_dim)
        self.fusion = FrePromptMoE(img_size=image_size, input_size=3, output_size=1,
                                   num_experts=4, hidden_size=emb_dim, noisy_gating=True, k=2,
                                   trainingmode=True)

    def forward(self, x, y):
        x_rec, y_rec, xu, xc, yu, yc, y_warp, offset = self.encoder(x, y)
        z, loss = self.fusion(torch.cat((xu, xc, yc), dim=1), torch.cat((yu, xc, yc), dim=1))
        return z, x_rec, y_rec, xu, xc, yu, yc, y_warp, offset


class FrePromptMoE(nn.Module):
    def __init__(self, img_size, input_size, output_size, num_experts, hidden_size, noisy_gating=True, k=4,
                 trainingmode=True):
        super(FrePromptMoE, self).__init__()
        self.noisy_gating = noisy_gating
        self.num_experts = num_experts
        self.output_size = output_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.training = trainingmode
        self.k = k
        self.freq_channels = 2 * input_size  # Real + Imag

        # Pooling
        self.avgpool = nn.AdaptiveAvgPool2d((img_size, img_size // 2 + 1))
        self.avgpool1x1 = nn.AdaptiveAvgPool2d((1, 1))
        self.maxpool = nn.AdaptiveMaxPool2d((img_size, img_size // 2 + 1))

        # Frequency Modulator
        self.freq_modulator = nn.Sequential(
            nn.Conv2d(self.freq_channels * 2, self.freq_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.freq_channels, self.freq_channels, kernel_size=1),
        )
        self.freq_enhance = nn.Sequential(
            nn.Conv2d(self.freq_channels, self.freq_channels, kernel_size=1),
            nn.ReLU(inplace=True)
        )
        # instantiate experts
        self.experts = nn.ModuleList(
            [DecoderLayer(self.input_size, self.output_size, self.hidden_size) for i in range(self.num_experts)])
        self.w_gate = nn.Parameter(torch.zeros(img_size ** 2 * input_size, num_experts), requires_grad=True)
        self.w_noise = nn.Parameter(torch.zeros(img_size ** 2 * input_size, num_experts), requires_grad=True)
        self.fre_prompt = nn.Parameter(torch.zeros(self.freq_channels, img_size, img_size // 2 + 1),
                                       requires_grad=True)

        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(1)
        self.register_buffer("mean", torch.tensor([0.0]))
        self.register_buffer("std", torch.tensor([1.0]))
        assert (self.k <= self.num_experts)

    def cv_squared(self, x):
        eps = 1e-10
        # if only num_experts = 1

        if x.shape[0] == 1:
            return torch.tensor([0], device=x.device, dtype=x.dtype)
        return x.float().var() / (x.float().mean() ** 2 + eps)

    def _gates_to_load(self, gates):
        return (gates > 0).sum(0)

    def _prob_in_top_k(self, clean_values, noisy_values, noise_stddev, noisy_top_values):
        batch = clean_values.size(0)
        m = noisy_top_values.size(1)
        top_values_flat = noisy_top_values.flatten()

        threshold_positions_if_in = torch.arange(batch, device=clean_values.device) * m + self.k
        threshold_if_in = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_in), 1)
        is_in = torch.gt(noisy_values, threshold_if_in)
        threshold_positions_if_out = threshold_positions_if_in - 1
        threshold_if_out = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_out), 1)
        # is each value currently in the top k.
        normal = Normal(self.mean, self.std)
        prob_if_in = normal.cdf((clean_values - threshold_if_in) / noise_stddev)
        prob_if_out = normal.cdf((clean_values - threshold_if_out) / noise_stddev)
        prob = torch.where(is_in, prob_if_in, prob_if_out)
        return prob

    def frequency_prompt(self, x):

        # is_image = x.ndim == 4
        # Prompt Generation
        B, C, H, W = x.shape

        # FFT to frequency domain
        fft = torch.fft.rfft2(x, norm='ortho')  # Shape: (B, C, H, W//2 + 1)
        real = fft.real
        imag = fft.imag
        freq = torch.cat([real, imag], dim=1)  # Shape: (B, 2C, H, W//2+1)

        pooled = torch.cat([self.avgpool(freq), self.maxpool(freq)], dim=1)  # (B, 4C, H, W//2+1)

        # Frequency Modulator
        modulated = self.freq_modulator(pooled)  # (B, 2C, 1, 1)
        prompt = self.fre_prompt  # (2C, 1, 1)
        prompt = prompt.unsqueeze(0)  # (1, 2C, 1, 1)
        enhanced_freq = modulated * prompt

        # Enhance and project back
        enhanced_freq = self.freq_enhance(enhanced_freq)  # Conv + ReLU
        real, imag = torch.chunk(enhanced_freq, 2, dim=1)
        complex_freq = torch.complex(real, imag)  # Reconstruct complex tensor

        # Inverse FFT
        out = torch.fft.irfft2(complex_freq, s=(H, W), norm='ortho')  # Back to spatial domain

        return F.softmax(self.avgpool1x1(out), dim=1)

    def noisy_top_k_gating(self, x, train, noise_epsilon=1e-2):
        clean_logits = x @ self.w_gate
        if self.noisy_gating and train:
            raw_noise_stddev = x @ self.w_noise
            noise_stddev = ((self.softplus(raw_noise_stddev) + noise_epsilon))
            noisy_logits = clean_logits + (torch.randn_like(clean_logits) * noise_stddev)
            logits = noisy_logits
        else:
            logits = clean_logits

        # calculate topk + 1 that will be needed for the noisy gates
        top_logits, top_indices = logits.topk(min(self.k + 1, self.num_experts), dim=1)
        top_k_logits = top_logits[:, :self.k]
        top_k_indices = top_indices[:, :self.k]
        top_k_gates = self.softmax(top_k_logits)

        zeros = torch.zeros_like(logits, requires_grad=True)
        gates = zeros.scatter(1, top_k_indices, top_k_gates)

        if self.noisy_gating and self.k < self.num_experts and train:
            load = (self._prob_in_top_k(clean_logits, noisy_logits, noise_stddev, top_logits)).sum(0)
        else:
            load = self._gates_to_load(gates)
        return gates, load

    def forward(self, x, ref, loss_coef=1e-2):
        fre_prompt = self.frequency_prompt(ref)  # b c h w
        x_ds = (x * fre_prompt + x).contiguous().view(x.shape[0], -1)
        gates, load = self.noisy_top_k_gating(x_ds, self.training)
        # calculate importance loss
        importance = gates.sum(0)

        loss = self.cv_squared(importance) + self.cv_squared(load)
        loss *= loss_coef

        dispatcher = SparseDispatcher(self.num_experts, gates)
        expert_inputs = dispatcher.dispatch(x)
        gates = dispatcher.expert_to_gates()
        expert_outputs = [self.experts[i](expert_inputs[i]) for i in range(self.num_experts)]
        y = dispatcher.combine(expert_outputs)
        return y, loss


class DecoupleModule(nn.Module):
    def __init__(self, in_channel=1, channel_fea=64):
        super(DecoupleModule, self).__init__()

        self.in_channel = in_channel
        self.channel_fea = channel_fea

        self.predict_ista = ista_unet(out_channel=1, kernel_size=3, hidden_layer_width_list=[128, 96, 64],
                                      n_classes=self.channel_fea, ista_num_steps=3)


    def forward(self, x, y):
        x_rec, y_rec, x_cur_common, x_cur_unique, y_cur_common, y_cur_unique, warped_y, offset = \
            self.predict_ista(x, y, torch.cat([x, y], 1))

        return x_rec, y_rec, x_cur_common, x_cur_unique, y_cur_common, y_cur_unique, warped_y, offset


loss_l1 = nn.L1Loss()


def loss_fn(img_sr, x_rec, y_rec, x_unique, x_common, y_unique, y_common, warped_y, img_hr, img_ref_hr):
    mi_loss = MutualInformation()
    loss_sr = loss_l1(img_sr, img_hr)
    loss1 = loss_l1(x_rec, img_hr)
    loss2 = loss_l1(y_rec, img_ref_hr)
    loss3 = mi_loss(x_common, x_unique)
    loss4 = mi_loss(y_common, y_unique)
    loss = loss1 + loss_sr + 0.01 * loss2 + 0.1 * (loss3 + loss4)
    return loss


class MutualInformation(torch.nn.Module):
    """
    Mutual Information
    """

    def __init__(self, sigma_ratio=1, minval=0., maxval=1., num_bin=32):
        super(MutualInformation, self).__init__()

        """Create bin centers"""
        bin_centers = np.linspace(minval, maxval, num=num_bin)
        vol_bin_centers = Variable(torch.linspace(minval, maxval, num_bin), requires_grad=False).to(args.DEVICE)
        num_bins = len(bin_centers)

        """Sigma for Gaussian approx."""
        sigma = np.mean(np.diff(bin_centers)) * sigma_ratio

        self.preterm = 1 / (2 * sigma ** 2)
        self.bin_centers = bin_centers
        self.max_clip = maxval
        self.num_bins = num_bins
        self.vol_bin_centers = vol_bin_centers

    def mi(self, y_true, y_pred):
        device = y_pred.device
        y_pred = torch.clamp(y_pred, 0., self.max_clip)
        y_true = torch.clamp(y_true, 0, self.max_clip)

        y_true = y_true.view(y_true.shape[0], -1)
        y_true = torch.unsqueeze(y_true, 2)
        y_pred = y_pred.view(y_pred.shape[0], -1)
        y_pred = torch.unsqueeze(y_pred, 2)

        nb_voxels = y_pred.shape[1]  # total num of voxels

        """Reshape bin centers"""
        o = [1, 1, np.prod(self.vol_bin_centers.shape)]
        vbc = torch.reshape(self.vol_bin_centers, o).to(device)

        """compute image terms by approx. Gaussian dist."""
        I_a = torch.exp(- self.preterm * torch.square(y_true - vbc))
        I_a = I_a / torch.sum(I_a, dim=-1, keepdim=True)

        I_b = torch.exp(- self.preterm * torch.square(y_pred - vbc))
        I_b = I_b / torch.sum(I_b, dim=-1, keepdim=True)

        # compute probabilities
        pab = torch.bmm(I_a.permute(0, 2, 1), I_b)
        pab = pab / nb_voxels
        pa = torch.mean(I_a, dim=1, keepdim=True)
        pb = torch.mean(I_b, dim=1, keepdim=True)

        papb = torch.bmm(pa.permute(0, 2, 1), pb) + 1e-6
        mi = torch.sum(torch.sum(pab * torch.log(pab / papb + 1e-6), dim=1), dim=1)
        return mi.mean()  # average across batch

    def forward(self, y_true, y_pred):
        return self.mi(y_true, y_pred)


if __name__ == "__main__":
    device = "cuda:4"
    x = torch.randn(1, 1, 256, 256).to(device)
    y = torch.randn(1, 1, 256, 256).to(device)
    model = CDDPE().to(device)
    z, z2, z3, z4, z5, z6 = model(x, y)

    print(z.shape)
    print(z2.shape)
    print(z3.shape)
    print(z4.shape)
    print(z5.shape)
    print(z6.shape)
