import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import timm

class AudioModel(torch.nn.Module):
  def __init__(self, arch_name='densenet201',Family="Densenet201", pretrained=True, fc_size=512, out_size=10, **kwargs):
        super(AudioModel, self).__init__()

        self.arch = timm.create_model(arch_name, pretrained=pretrained)

        if Family =='Densenet201' :
            head = torch.nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            head.weight = torch.nn.Parameter(self.arch.features.conv0.weight.sum(dim=1, keepdim=True))

            self.arch.features.conv0 = head
            fc_size = self.arch.classifier.in_features
            self.arch.classifier = nn.Sequential(nn.Linear(fc_size, out_size))
        elif Family =='Densenet161' :
            head = torch.nn.Conv2d(1,  96, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3))
            head.weight = torch.nn.Parameter(self.arch.features.conv0.weight.sum(dim=1, keepdim=True))

            self.arch.features.conv0 = head
            fc_size = self.arch.classifier.in_features
            self.arch.classifier = nn.Sequential(nn.Linear(fc_size, out_size))

  def forward(self, x):
    x = self.arch(x)
    return x


#######################################################################################################################################
################################## EFFICIENTNET MODEL #################################################################################
#######################################################################################################################################

####### ATTENTION MODULES

'''
Borrowed from https://www.kaggle.com/jy2tong/efficientnet-b2-soft-attention
'''

class PAM_Module(nn.Module):
    ''' Position attention module'''
    #Ref from SAGAN
    def __init__(self, in_dim):
        super(PAM_Module, self).__init__()
        self.chanel_in  = in_dim
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv   = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma      = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        '''
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        '''
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width*height).permute(0, 2, 1)
        proj_key   = self.key_conv(x).view(m_batchsize, -1, width*height)
        energy     = torch.bmm(proj_query, proj_key)
        attention  = torch.softmax(energy, dim=-1)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width*height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma*out + x
        return out


class CAM_Module(nn.Module):
    ''' Channel attention module'''
    def __init__(self, in_dim):
        super(CAM_Module, self).__init__()
        self.chanel_in = in_dim
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self,x):
        '''
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X C X C
        '''
        m_batchsize, C, height, width = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        proj_key   = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy     = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy
        attention  = torch.softmax(energy_new, dim=-1)
        proj_value = x.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)
        out = self.gamma*out + x
        
        return out


class CBAM(nn.Module):
    def __init__(self, in_channels):
        # def __init__(self):
        super(CBAM, self).__init__()
        inter_channels = in_channels // 4
        self.conv1_c = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                     nn.BatchNorm2d(inter_channels),
                                     nn.ReLU())
        
        self.conv1_s = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                     nn.BatchNorm2d(inter_channels),
                                     nn.ReLU())

        self.channel_gate = CAM_Module(inter_channels)
        self.spatial_gate = PAM_Module(inter_channels)

        self.conv2_c = nn.Sequential(nn.Conv2d(inter_channels, in_channels, 3, padding=1, bias=False),
                                     nn.BatchNorm2d(in_channels),
                                     nn.ReLU())
        self.conv2_a = nn.Sequential(nn.Conv2d(inter_channels, in_channels, 3, padding=1, bias=False),
                                     nn.BatchNorm2d(in_channels),
                                     nn.ReLU())

    def forward(self, x):
        feat1    = self.conv1_c(x)
        chnl_att = self.channel_gate(feat1)
        chnl_att = self.conv2_c(chnl_att)

        feat2    = self.conv1_s(x)
        spat_att = self.spatial_gate(feat2)
        spat_att = self.conv2_a(spat_att)

        x_out = chnl_att + spat_att

        return x_out

class model_with_attention(nn.Module):
    
    def __init__(self, CFG):
        super().__init__()
        self.backbone            = timm.create_model(model_name = CFG.model_name, pretrained = True)
        self.backbone._dropout   = nn.Dropout(0.1)
        n_features               = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Linear(n_features, CFG.target_size)
        self.local_fe            = CBAM(n_features)
        self.dropout             = nn.Dropout(0.1)
        self.classifier          = nn.Sequential(nn.Linear(n_features + n_features, n_features),
                                                nn.BatchNorm1d(n_features),
                                                nn.Dropout(0.1),
                                                nn.ReLU(),
                                                nn.Linear(n_features, CFG.target_size))

    def forward(self, image):
        enc_feas    = self.backbone.forward_features(image)
        global_feas = self.backbone.global_pool(enc_feas)
        global_feas = global_feas.flatten(start_dim = 1)
        global_feas = self.dropout(global_feas)
        local_feas  = self.local_fe(enc_feas)
        local_feas  = torch.sum(local_feas, dim = [2, 3])
        local_feas  = self.dropout(local_feas)
        all_feas    = torch.cat([global_feas, local_feas], dim = 1)
        outputs     = self.classifier(all_feas)
        return outputs
    