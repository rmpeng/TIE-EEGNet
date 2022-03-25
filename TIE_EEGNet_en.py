import torch.nn as nn
from TIE_Layer_en import encoderLayer_ensemble
import torch
import my_config
GPU_ID = my_config.Config.GPU_id
device = torch.device('cuda:{}'.format(GPU_ID) if torch.cuda.is_available() else "cpu")
class Conv2dWithConstraint(nn.Conv2d):
    def __init__(self, *args, max_norm = 1, **kwargs):
        self.max_norm = max_norm
        super(Conv2dWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, x):
        self.weight.data = torch.renorm(self.weight.data, p=2, dim=0, maxnorm=self.max_norm)
        return super(Conv2dWithConstraint, self).forward(x)

class TIE_EEGNet(nn.Module):
    def CalculateOutSize(self, model, channels, samples):
        data = torch.rand(1,1,channels, samples)
        model.eval()
        out = model(data).shape
        return out[2:]

    def ClassifierBlock(self, inputSize, n_class):
        return nn.Sequential(
            nn.Linear(inputSize, n_class, bias= False),
            nn.Softmax(dim = 1)
        )

    def __init__(self, alpha_list, alpha=2, n_class = 4, channels = 20, samples = 512, dropoutRate = 0.5,
                 kernel_length = 64, kernel_length2 = 16, F1 = 8, F2 = 16, D = 2,
                 encoder = 'sinusoidal', isTrain= True):
        super(TIE_EEGNet, self).__init__()
        self.F1 = F1
        self.F2 = F2
        self.D = D
        self.samples = samples
        self.n_class = n_class
        self.channels = channels
        self.dropoutRate = dropoutRate
        self.kernel_length = kernel_length
        self.kernel_length2 = kernel_length2
        self.encoder = encoder
        self.isTrain = isTrain
        self.alpha_list=alpha_list
        self.alpha=alpha


        #self.Conv2d_1 = nn.Conv2d(1, self.F1, (1, self.kernel_length), padding='same', bias = False)#(padding =(1, self.kernel_size //2))
        self.Conv2d_1 = nn.Conv2d(1, self.F1, (1, self.kernel_length), padding=(0, self.kernel_length // 2), bias = False)#'same'

        self.encoder_Layer = encoderLayer_ensemble(self.alpha_list, encoder= self.encoder, conv2Doutput= self.Conv2d_1, inc = 1, outc = self.F1,
                                  kernel_size = (1, self.kernel_length), pad=(0, self.kernel_length // 2), stride = 1, bias = False,
                                  sample_len= self.samples,is_Train=self.isTrain,alpha=self.alpha)
        self.BatchNorm_1_1 = nn.BatchNorm2d(self.F1, momentum=0.01, affine=True, eps = 1e-3)
        self.Depthwise_Conv2d = Conv2dWithConstraint(self.F1, self.F1 * self.D, (self.channels, 1), stride=1, max_norm= 1, groups= self.F1, bias = False) #, padding='valid'

        self.BatchNorm_1_2 = nn.BatchNorm2d(self.F1 * self.D, momentum=0.01, affine=True, eps = 1e-3)
        self.avg_pool_1 = nn.AvgPool2d((1, 4), stride= 4)
        self.Dropout_1 = nn.Dropout(p= self.dropoutRate)
        self.Separable_Conv2d_1 = nn.Conv2d(self.F1 * self.D, self.F1 * self.D, (1, self.kernel_length2), padding=(0, self.kernel_length // 2), bias= False, groups= self.F1 * self.D) # 'same'
        self.Separable_Conv2d_2 = nn.Conv2d(self.F1 * self.D, self.F2, 1, padding= (0, 0), bias= False, groups= 1)
        self.BatchNorm_2 = nn.BatchNorm2d(self.F2, momentum=0.01, affine=True, eps = 1e-3)
        self.avg_pool_2 = nn.AvgPool2d((1, 8), stride= 8)
        self.Dropout_2 = nn.Dropout(p= self.dropoutRate)

        self.fea_model = nn.Sequential(self.encoder_Layer,
                                       self.BatchNorm_1_1,
                                       self.Depthwise_Conv2d,
                                       self.BatchNorm_1_2,
                                       nn.ELU(),
                                       self.avg_pool_1,
                                       self.Dropout_1,
                                       self.Separable_Conv2d_1,
                                       self.Separable_Conv2d_2,
                                       self.BatchNorm_2,
                                       nn.ELU(),
                                       self.avg_pool_2,
                                       self.Dropout_2)

        self.fea_out_size = self.CalculateOutSize(self.fea_model, self.channels, self.samples)
        self.classifierBlock = self.ClassifierBlock(self.F2 * self.fea_out_size[1], self.n_class)

    def forward(self, data):
        conv_data = self.fea_model(data)
        flatten_data = conv_data.view(conv_data.size()[0], -1)
        pred_label = self.classifierBlock(flatten_data)

        return pred_label




