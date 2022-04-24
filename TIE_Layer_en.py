import torch
import torch.nn as nn
import numpy as np
import random
import my_config
GPU_ID = my_config.Config.GPU_id
device = torch.device('cuda:{}'.format(GPU_ID) if torch.cuda.is_available() else "cpu")


class TIE_Layer(nn.Module):
    def __init__(self, pool,alpha_list, tie, conv2Doutput, inc, outc, kernel_size, pad, stride, bias, sample_len, is_Train=True, alpha=2):
        super(TIE_Layer, self).__init__()
        #alpha_list #[2,4,8,10,16,32,256]
        self.list_alpha = list(alpha_list)
        self.TIE = tie
        self.isTrain = is_Train
        self.sample_len = sample_len
        self.alpha = alpha
        self.pool=pool
        self.new_alpha = self.alpha
        self.conv2DOutput = conv2Doutput
        if self.pool=='Avg':
            self.pooling = nn.AvgPool2d(kernel_size=kernel_size, stride=stride, padding=pad)
        elif self.pool=='Max':
            self.pooling = nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=pad)
        else:
            self.pooling = nn.Conv2d(in_channels=inc, out_channels=outc, kernel_size=kernel_size, padding=pad, stride=stride,
                                        bias=bias)  # 辅助卷积
            self.weight_mat = np.ones(shape=[outc, inc, self.pooling.kernel_size[0], self.pooling.kernel_size[1]])
            self.pooling.weight = nn.Parameter(torch.Tensor(self.weight_mat), requires_grad=False)

        self.fix_encodings_base = torch.range(0,  sample_len , 1, dtype=torch.float32)  # .to(device)

        if self.TIE == 'linear':
            self.scale_factor = nn.Parameter(torch.zeros(1, ), requires_grad=True)
            self.scale_factor = self.scale_factor  # .to(device)
            self.scale_factor.requires_grad_ = True

        elif self.TIE == 'sinusoidal':
            self.scale_factor = nn.Parameter(torch.zeros(1, ), requires_grad=True)  # .to(device)
            self.scale_factor.requires_grad_ = True

            for i in range(self.sample_len):
                if i % 2 == 0:
                    self.fix_encodings_base[i] = torch.sin(self.fix_encodings_base[i] / self.new_alpha)
                else:
                    self.fix_encodings_base[i] = torch.cos(self.fix_encodings_base[i] / self.new_alpha)
        else:
            self.scale_factor = 0


    def forward(self, data):

        try:
            newconv = torch.add(self.conv2DOutput(data),
                               torch.mul(self.pooling(data), torch.reshape(self.fix_encodings_base * self.scale_factor, [1, 1, 1, -1])))
        except:
            self.fix_encodings_base = self.fix_encodings_base.to(device)
            torch.reshape(self.fix_encodings_base * self.scale_factor, [1, 1, 1, -1]).to(device)
            newconv = torch.add(self.conv2DOutput(data),
                                torch.mul(self.pooling(data),
                                          torch.reshape(self.fix_encodings_base * self.scale_factor, [1, 1, 1, -1])))
        return newconv
