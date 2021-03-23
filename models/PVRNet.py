from models import *
import config
import numpy as np
import torch.nn.functional as F


class PVRNet(nn.Module):
    def __init__(self, n_classes=40, init_weights=True):
        super(PVRNet, self).__init__()

        self.view_num = 12
        self.fea_dim = 1024
        # self.num_bottleneck = 512
        self.hidden_size = 512  # TODO: How to Decide this param?
        self.e_dropout = 0.5
        self.n_scale = [2, 3, 4]
        self.n_neighbor = config.pvd_net.n_neighbor

        self.mvcnn = BaseFeatureNet(base_model_name=config.base_model_name)

        # Point cloud net
        self.trans_net = transform_net(6, 3)
        self.conv2d1 = conv_2d(6, 64, 1)
        self.conv2d2 = conv_2d(128, 64, 1)
        self.conv2d3 = conv_2d(128, 64, 1)
        self.conv2d4 = conv_2d(128, 128, 1)
        self.conv2d5 = conv_2d(320, 1024, 1)

        # max-pooling for views' features
        self.maxpol_mv = nn.Sequential(
            nn.MaxPool2d((12, 1), stride=1),
        )

        # FC for views' features
        self.fc_mv = nn.Sequential(
            fc_layer(4096, 1024, True),
        )

        # Encoder5 in ARGF
        self.GEN_EN_norm = nn.BatchNorm1d(self.hidden_size)
        self.GEN_EN_norm2 = nn.BatchNorm1d(self.fea_dim * 10)
        self.GEN_EN_norm3 = nn.BatchNorm1d(self.hidden_size * 10)
        self.GEN_EN_drop = nn.Dropout(p=self.e_dropout)
        self.GEN_EN_linear_1 = nn.Linear(self.fea_dim, self.fea_dim * 10)
        self.GEN_EN_linear_2 = nn.Linear(self.fea_dim * 10, self.hidden_size * 10)
        self.GEN_EN_linear_3 = nn.Linear(self.hidden_size * 10, self.hidden_size)
        self.GEN_EN_linear_4 = nn.Linear(self.hidden_size, self.hidden_size)

        # Decoder2 in ARGF
        self.GEN_DEC_model_decode = nn.Sequential(
            nn.Linear(self.hidden_size, 512),
            nn.Dropout(0.5),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 64),
            nn.Dropout(0.5),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(64, self.fea_dim),
            nn.Tanh()
        )

        # Discriminator for Adversarial Learning
        self.DIS_model_discrim = nn.Sequential(
            nn.Linear(self.fea_dim, 64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(64, 16),
            nn.Tanh(),
            nn.Linear(16, 1),
            nn.Sigmoid()
            #    nn.Tanh(),
            #   nn.ReLU()
        )

        # Classifier3 for Adversarial Learning
        self.CLS_norm = nn.BatchNorm1d(self.hidden_size)
        self.CLS_drop = nn.Dropout(p=0.5)
        self.CLS_linear_1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.CLS_linear_2 = nn.Linear(self.hidden_size, self.fea_dim)

        self.fusion_mlp2 = nn.Sequential(
            fc_layer(3072, 1024, True),
            fc_layer(1024, 256, True),
            nn.Dropout(p=0.5)
        )
        self.fusion_mlp3 = nn.Sequential(
            nn.Linear(256, n_classes),
            # nn.Linear(n_classes, 1),
        )
        if init_weights:
            self.init_mvcnn_view()
            self.init_dgcnn()

    def init_mvcnn_view(self):
        print(f'init parameter from mvcnn {config.base_model_name}')
        mvcnn_state_dict = torch.load(config.view_net.ckpt_load_file)['model']
        pvrnet_state_dict = self.state_dict()

        mvcnn_state_dict = {k.replace('features', 'mvcnn', 1): v for k, v in mvcnn_state_dict.items()}
        mvcnn_state_dict = {k: v for k, v in mvcnn_state_dict.items() if k in pvrnet_state_dict.keys()}
        pvrnet_state_dict.update(mvcnn_state_dict)
        self.load_state_dict(pvrnet_state_dict)
        print(f'load ckpt from {config.view_net.ckpt_load_file}')

    def init_dgcnn(self):
        print(f'init parameter from dgcnn')
        dgcnn_state_dict = torch.load(config.pc_net.ckpt_load_file)['model']
        pvrnet_state_dict = self.state_dict()

        dgcnn_state_dict = {k: v for k, v in dgcnn_state_dict.items() if k in pvrnet_state_dict.keys()}
        pvrnet_state_dict.update(dgcnn_state_dict)
        self.load_state_dict(pvrnet_state_dict)
        print(f'load ckpt from {config.pc_net.ckpt_load_file}')

    def forward(self, pc, mv, dp, get_fea=False):
        batch_size = pc.size(0)
        self.view_num = mv.size(1)
        mv, mv_view = self.mvcnn(mv)
        dp, dp_view = self.mvcnn(dp)

        # point-cloud feature extraction
        x_edge = get_edge_feature(pc, self.n_neighbor)
        x_trans = self.trans_net(x_edge)
        x = pc.squeeze(-1).transpose(2, 1)
        x = torch.bmm(x, x_trans)
        x = x.transpose(2, 1)

        x1 = get_edge_feature(x, self.n_neighbor)
        x1 = self.conv2d1(x1)
        x1, _ = torch.max(x1, dim=-1, keepdim=True)

        x2 = get_edge_feature(x1, self.n_neighbor)
        x2 = self.conv2d2(x2)
        x2, _ = torch.max(x2, dim=-1, keepdim=True)

        x3 = get_edge_feature(x2, self.n_neighbor)
        x3 = self.conv2d3(x3)
        x3, _ = torch.max(x3, dim=-1, keepdim=True)

        x4 = get_edge_feature(x3, self.n_neighbor)
        x4 = self.conv2d4(x4)
        x4, _ = torch.max(x4, dim=-1, keepdim=True)

        x5 = torch.cat((x1, x2, x3, x4), dim=1)
        x5 = self.conv2d5(x5)
        x5, _ = torch.max(x5, dim=-2, keepdim=True)

        pc = x5.squeeze()

        # multi-view feature extraction
        mv_view = self.maxpol_mv(mv_view).squeeze()
        mv_view = self.fc_mv(mv_view)

        # depth-image feature extraction
        dp_view = self.maxpol_mv(dp_view).squeeze()
        dp_view = self.fc_mv(dp_view)

        # encoder
        encoder_in = [pc, mv_view, dp_view]
        encoder_out = []
        for feature in encoder_in:
            normed = feature  # self.norm(x)
            #   dropped = self.drop(normed)
            y_1 = F.leaky_relu(self.GEN_EN_norm2(self.GEN_EN_drop(self.GEN_EN_linear_1(normed))))
            y_2 = F.leaky_relu(self.GEN_EN_norm3(self.GEN_EN_drop(self.GEN_EN_linear_2(y_1))))
            y_2 = F.leaky_relu(self.GEN_EN_norm(self.GEN_EN_drop(self.GEN_EN_linear_3(y_2))))
            y_3 = torch.tanh(self.GEN_EN_linear_4(y_2))
            encoder_out.append(y_3)

        # decoder
        decoder_out = []
        for encoded_fea in encoder_out:
            img_flat = self.GEN_DEC_model_decode(encoded_fea)
            img = img_flat.view(img_flat.shape[0], -1)
            decoder_out.append(img)

        # discriminator
        dis_pc = self.DIS_model_discrim(decoder_out[0])
        dis_mv = self.DIS_model_discrim(decoder_out[1])
        dis_dp = self.DIS_model_discrim(decoder_out[2])

        # classifier
        classifier_out = []
        for encoded_fea in encoder_out:
            normed = self.CLS_norm(encoded_fea)
            dropped = self.CLS_drop(normed)
            y_1 = self.CLS_drop(torch.tanh(self.CLS_linear_1(dropped)))
            y_2 = F.softmax(self.CLS_linear_2(y_1), 1)
            classifier_out.append(y_2)

        # fusion
        fea = torch.cat((pc, mv_view, dp_view), 1)

        # Final FC Layers
        net_fea = self.fusion_mlp2(fea)
        pred = self.fusion_mlp3(net_fea)

        return pc, mv_view, dp_view, encoder_out[0], encoder_out[1], encoder_out[2], decoder_out[0], decoder_out[1], \
               decoder_out[2], dis_pc, dis_mv, dis_dp, classifier_out[0], classifier_out[1], classifier_out[2], net_fea,\
               pred
