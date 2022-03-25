import torch
import torch.nn as nn
import numpy as np
import random
import my_config
GPU_ID = my_config.Config.GPU_id
device = torch.device('cuda:{}'.format(GPU_ID) if torch.cuda.is_available() else "cpu")


class encoderLayer_ensemble(nn.Module):
    def __init__(self, alpha_list, encoder, conv2Doutput, inc, outc, kernel_size, pad, stride, bias, sample_len, is_Train=True, alpha=2):
        super(encoderLayer_ensemble, self).__init__()
        #alpha_list #[2,4,8,10,16,32,256]
        self.list_alpha = list(alpha_list)
        self.encoder = encoder
        self.isTrain = is_Train
        self.sample_len = sample_len
        self.alpha = alpha
        self.new_alpha = self.alpha
        self.conv2DOutput = conv2Doutput
        self.auxiliary_conv = nn.Conv2d(in_channels=inc, out_channels=outc, kernel_size=kernel_size, padding=pad, stride=stride,
                                        bias=bias)
        self.weight_mat = np.ones(shape=[outc, inc, self.auxiliary_conv.kernel_size[0], self.auxiliary_conv.kernel_size[1]])
        self.fix_encodings_base = torch.range(0,  sample_len , 1, dtype=torch.float32)  # .to(device)

        self.auxiliary_conv.weight = nn.Parameter(torch.Tensor(self.weight_mat), requires_grad=False)

        if self.encoder == 'sinusoidal':
            self.scale_factor = nn.Parameter(torch.zeros(1, ), requires_grad=True)  # .to(device)
            self.scale_factor.requires_grad_ = True

            for i in range(self.sample_len):
                if i % 2 == 0:
                    self.fix_encodings_base[i] = torch.sin(self.fix_encodings_base[i] / self.new_alpha)
                else:
                    self.fix_encodings_base[i] = torch.cos(self.fix_encodings_base[i] / self.new_alpha)
        else:
            self.scale_factor = 0

    def updata_fix_coding(self):

        self.new_alpha = random.sample(self.list_alpha,1)[0]
        self.fix_encodings_base = torch.range(0,  self.sample_len , 1, dtype=torch.float32)  # .to(device)

        if self.encoder == 'sinusoidal':
            for i in range(self.sample_len):
                if i % 2 == 0:
                    self.fix_encodings_base[i] = torch.sin(self.fix_encodings_base[i] / self.new_alpha)
                else:
                    self.fix_encodings_base[i] = torch.cos(self.fix_encodings_base[i] / self.new_alpha)

        return self.fix_encodings_base, self.new_alpha


    def forward(self, data):
        if self.isTrain is True:
            self.fix_encodings_base, self.new_alpha = self.updata_fix_coding()
        else:
            self.new_alpha = self.alpha
            self.fix_encodings_base = torch.range(0, self.sample_len, 1, dtype=torch.float32)  # .to(device)
            if self.encoder == 'sinusoidal':
                for i in range(self.sample_len):
                    if i % 2 == 0:
                        self.fix_encodings_base[i] = torch.sin(self.fix_encodings_base[i] / self.new_alpha)
                    else:
                        self.fix_encodings_base[i] = torch.cos(self.fix_encodings_base[i] / self.new_alpha)

        try:
            newconv = torch.add(self.conv2DOutput(data),
                               0.1* torch.mul(self.auxiliary_conv(data), torch.reshape(self.fix_encodings_base * self.scale_factor, [1, 1, 1, -1])))
            # self.fix_encodings_base=self.fix_encodings_base.to(device)
        except:
            self.fix_encodings_base = self.fix_encodings_base.to(device)
            torch.reshape(self.fix_encodings_base * self.scale_factor, [1, 1, 1, -1]).to(device)
            newconv = torch.add(self.conv2DOutput(data),
                                0.1 * torch.mul(self.auxiliary_conv(data),
                                          torch.reshape(self.fix_encodings_base * self.scale_factor, [1, 1, 1, -1])))
        return newconv
