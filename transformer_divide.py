from pointnet_util import index_points, square_distance
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)  # xx:[B,1,N]
    pairwise_distance = -xx - inner - xx.transpose(2, 1)  # -(xi-xj)2,距离越大，值越小；距离越小，值越大
    # 取一个tensor的topk元素（降序后的前k个大小的元素值及索引）--取最近的K个元素
    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
    return idx
def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.

    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst

    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist
def knn(x, k):
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True) #xx:[B,1,N]
    pairwise_distance = -xx - inner - xx.transpose(2, 1) #-(xi-xj)2,距离越大，值越小；距离越小，值越大
    #取一个tensor的topk元素（降序后的前k个大小的元素值及索引）--取最近的K个元素
    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
    return idx


def get_graph_feature(x, k=20, idx=None, dim9=False,dim6=False):
    batch_size = x.size(0)
    num_points = x.size(1)
    x = x.contiguous().view(batch_size, -1, num_points)
    if idx is None:
        if dim9 == False and dim6 == False:
            idx = knn(x, k=k)   # (batch_size, num_points, k)
        elif dim9 == True:
            idx = knn(x[:, 6:], k=k)
        elif dim6 == True:
            idx=knn(x[:,0:3],k=k)
    device = torch.device('cuda')
#??????????????????????????????????????????????????????????????
    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points #[[[0],[N],[2N],[3N]......[BN]]  [B,1,1]
    idx2=idx
    idx = idx + idx_base

    idx1 = idx.view(-1) #B*N*k
 
    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size*num_points, -1)[idx1, :]
    feature = feature.view(batch_size, num_points, k, num_dims) #[B,N,k,C]
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    
    feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2).contiguous()#[B,C,N,k]
    # print(idx.shape)
    return feature,idx2      # (batch_size, 2*num_dims, num_points, k)
    # return feature
class GT(nn.Module):
    # def __init__(self, d_points, d_model, k) -> None:
    def __init__(self, d_points, d_model,k) -> None:
        super().__init__()
        self.k=k
        self.fc1 = nn.Linear(d_points, d_model)
        self.fc2 = nn.Linear(d_model, d_model)
        self.bn=nn.BatchNorm1d(d_model)
        self.relu=nn.ReLU()
        self.fc_delta = nn.Sequential(
            nn.Linear(3, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        self.fb=nn.Sequential(
            nn.Linear(d_points*2, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        self.fc_gamma = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        self.w_qs = nn.Linear(d_model, d_model, bias=False)

        self.w_ks = nn.Linear(d_model, d_model, bias=False)
        self.w_vs = nn.Linear(d_model, d_model, bias=False)


        # self.w_ks1 = nn.Linear(d_model*4, d_model, bias=False)
        # self.w_vs1 = nn.Linear(d_model*4, d_model, bias=False)

        self.sa=SA_Layer(d_model)
        # self.conv =nn.Conv1d(d_model,d_model, kernel_size=1, bias=False)
                                  
        # self.r=r
        # self.nsample=nsample

    # xyz: b x n x 3, features: b x n x f
    def forward(self,features,dim9=False,dim6=False):
        #b,n,c
        knn_features, knn_idx = get_graph_feature(features, k=self.k,dim9=dim9,dim6=dim6)
        features=features.permute(0,2,1)
        knn_features=knn_features.permute(0,2,3,1)
        batch_size,_,num_points=features.shape
        knn_features=self.fb(knn_features)
        x = self.fc1(features)

        # x = x.permute(0, 2, 1)
        #
        # x = F.leaky_relu(self.bn(x))

        # assert not torch.any(torch.isnan(x))
        # x=x.permute(0,2,1)
        # print(x.shape)
        k=self.w_ks(x)
        # k1=GroupLayer(k,self.k,idx_EI)
        # k2=GroupLayer(k,self.k,idx_EU)
        # k=torch.cat((k1,k2),dim = -1)
        # k=self.w_ks1(k)
        # print(k.shape)
        v=self.w_vs(x)
        # v1=GroupLayer(v,self.k,idx_EI)
        # v2=GroupLayer(v,self.k,idx_EU)
        # v=torch.cat((v1,v2),dim=-1)
        # v=self.w_vs1(v)
        # print(v.shape)
        # print(knn_idx.shape)
        q, k, v = self.w_qs(x), index_points(k, knn_idx), index_points(v, knn_idx)
        # q,k,v=self.w_qs(x),k,v
        # print(xyz.shape)
        # print(knn_xyz.shape)
        # print(xyz.shape)
        # print(knn_xyz.shape)
        # print(xyz.shape)
        # print(knn_xyz.shape)

        # knn_xyz=knn_xyz.permute(0,2,3,1)
        # pos_enc = self.fc_delta(knn_xyz)
        attn = self.fc_gamma(q[:, :, None] - k + knn_features)
        # attn = self.fc_gamma(q[:, :, None] - k)
        # assert not torch.any(torch.isnan(attn))
        attn = F.log_softmax(attn / np.sqrt(k.size(-1)), dim=-2)  # b x n x k x f

        # assert not torch.any(torch.isnan(attn))
        res = torch.einsum('bmnf,bmnf->bmf', attn, v + knn_features)
        # res = torch.einsum('bmnf,bmnf->bmf', attn, v)
        # print(res.shape)
        # print(pre.shape)
        # res = self.fc2(res) + x

        res = self.relu(self.bn(self.fc2(res).permute(0,2,1)))+x.permute(0,2,1)
        # res = res.permute(0, 2, 1)
        # res= F.leaky_relu(self.bn(res))
        # res=self.conv(res)
        # x = x.permute(0, 2, 1)
        # res=x+res
        # print("res")
        # print("res:")
        # print(res.data)
        # print(res.shape)
        # res=res.permute(0,2,1)
        # print(res.shape)
        #
        # torch.cuda.empty_cache()
        res =self.sa(res)
        res = res.permute(0, 2, 1)
        # torch.cuda.empty_cache()
        return res, attn

class TransformerBlock(nn.Module):
    # def __init__(self, d_points, d_model, k) -> None:
    def __init__(self, d_points, d_model, k=20) -> None:
        super().__init__()
        self.fc1 = nn.Linear(d_points, d_model)
        self.fc2 = nn.Linear(d_model, d_model)
        self.bn=nn.BatchNorm1d(d_model)
        self.relu=nn.ReLU()
        self.fc_delta = nn.Sequential(
            nn.Linear(3, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        self.fc_gamma = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        self.w_qs = nn.Linear(d_model, d_model, bias=False)

        self.w_ks = nn.Linear(d_model, d_model, bias=False)
        self.w_vs = nn.Linear(d_model, d_model, bias=False)


        # self.w_ks1 = nn.Linear(d_model*4, d_model, bias=False)
        # self.w_vs1 = nn.Linear(d_model*4, d_model, bias=False)
        self.k = k
        self.sa=SA_Layer(d_model)
        # self.r=r
        # self.nsample=nsample

    # xyz: b x n x 3, features: b x n x f
    def forward(self,xyz, features,dim=False):
        batch_size,num_points,_=xyz.shape
    # def forward(self,xyz, features):
        knn_idx = knn(features.permute(0,2,1), k=self.k)
    #     if dim==True:
    #         # print(features.shape)
    #         # print(xyz.shape)
    #         knn_idx= knn(xyz.permute(0,2,1), k=self.k)
    #         # knn_idx=query_ball_point(self.r,self.nsample,xyz,xyz)
    #         # idx_EU, idx_EI = eigen_Graph(xyz.permute(0, 2, 1).contiguous(), k=self.k)
    #
    #     else:
    #         knn_idx = knn(features.permute(0, 2, 1), k=self.k)
            # knn_idx = query_ball_point(self.r, self.nsample, features, features)
            # idx_EU, idx_EI = eigen_Graph(features.permute(0, 2, 1).contiguous(), k=self.k)
    # features = features.permute(0, 2, 1)
        # xyz = xyz.permute(0, 2, 1)

        knn_xyz = index_points(xyz, knn_idx)
    #     print("xyz:shape")
    #     print(xyz.shape)
        # print(knn_xyz.shape)

     # print(features.shape)

        x = self.fc1(features)

        # x = x.permute(0, 2, 1)
        #
        # x = F.leaky_relu(self.bn(x))

        # assert not torch.any(torch.isnan(x))
        # x=x.permute(0,2,1)
        # print(x.shape)
        k=self.w_ks(x)
        # k1=GroupLayer(k,self.k,idx_EI)
        # k2=GroupLayer(k,self.k,idx_EU)
        # k=torch.cat((k1,k2),dim = -1)
        # k=self.w_ks1(k)
        # print(k.shape)
        v=self.w_vs(x)
        # v1=GroupLayer(v,self.k,idx_EI)
        # v2=GroupLayer(v,self.k,idx_EU)
        # v=torch.cat((v1,v2),dim=-1)
        # v=self.w_vs1(v)
        # print(v.shape)
        # print(knn_idx.shape)
        q, k, v = self.w_qs(x), index_points(k, knn_idx), index_points(v, knn_idx)
        # q,k,v=self.w_qs(x),k,v
        # print(xyz.shape)
        # print(knn_xyz.shape)
        # print(xyz.shape)
        # print(knn_xyz.shape)
        # print(xyz.shape)
        # print(knn_xyz.shape)
        pos_enc = self.fc_delta(xyz[:, :, None] - knn_xyz)  # b x n x k x f
        # knn_xyz=knn_xyz.permute(0,2,3,1)
        # pos_enc = self.fc_delta(knn_xyz)
        attn = self.fc_gamma(q[:, :, None] - k + pos_enc)
        # attn = self.fc_gamma(q[:, :, None] - k)
        # assert not torch.any(torch.isnan(attn))
        attn = F.log_softmax(attn / np.sqrt(k.size(-1)), dim=-2)  # b x n x k x f

        # assert not torch.any(torch.isnan(attn))
        res = torch.einsum('bmnf,bmnf->bmf', attn, v + pos_enc)
        # res = torch.einsum('bmnf,bmnf->bmf', attn, v)
        # print(res.shape)
        # print(pre.shape)
        res = self.fc2(res) + x

        # res = self.relu(self.bn(self.fc2(res).permute(0,2,1)))+x.permute(0,2,1)
        res = res.permute(0, 2, 1)
        res= F.leaky_relu(self.bn(res))

        # x = x.permute(0, 2, 1)
        # res=x+res
        # print("res")
        # print("res:")
        # print(res.data)
        # print(res.shape)
        # res=res.permute(0,2,1)
        # print(res.shape)
        #
        res =self.sa(res)
        res = res.permute(0, 2, 1)
        return res,attn




class Attention(nn.Module):
    def __init__(self, dim,out_dim=None, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.out_dim=out_dim
        if out_dim is not None:
            print(out_dim)
            self.xianxing=nn.Linear(dim,out_dim)
            dim=out_dim
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        # x=x.permute(0,2,1)
        B, N, C = x.shape  #[B,128,384]
        # print(x.shape)
        if self.out_dim is not None:
            # print(self.out_dim)
            x=self.xianxing(x)
            B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4) #[B,N,3,6,64]-->[3,B,6,N,64]
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple) [B,6,N,64]

        attn = (q @ k.transpose(-2, -1)) * self.scale #内积  [B,6,N,N]
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)#[B,6,N,64]->[B,N,6,64]->[B,N,384]
        x = self.proj(x)
        x = self.proj_drop(x)
        # print("shape:")
        # print(x.shape)

        torch.cuda.empty_cache()

        return x
class SA_Layer(nn.Module):
    def __init__(self, channels):
        super(SA_Layer, self).__init__()
        self.q_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.k_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.q_conv.weight = self.k_conv.weight
        self.q_conv.bias = self.k_conv.bias

        self.v_conv = nn.Conv1d(channels, channels, 1)
        self.trans_conv = nn.Conv1d(channels, channels, 1)
        self.after_norm = nn.BatchNorm1d(channels)
        self.act = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # b, n, c
        # x_q = self.q_conv(x)
        x_q = self.q_conv(x).permute(0, 2, 1)
        # b, c, n
        x_k = self.k_conv(x)
        x_v = self.v_conv(x)

        # b, n, n
        # assert not torch.any(torch.isnan(x_q))
        # assert not torch.any(torch.isnan(x_k))


        energy = x_q @ x_k
        # assert not torch.any(torch.isnan(energy))
        # print(x_q.data)
        # print(x_k.data)
        # print(x_v.data)
        # print(energy.data)
        attention = self.softmax(energy)
        # assert not torch.any(torch.isnan(attention))
        # print(attention.data)

        attn=1e-9 + attention.sum(dim=1, keepdim=True)
        attention = attention / attn
        # assert not torch.any(torch.isnan(attention))
        # b, c, n


        x_r = x_v @ attention
        x_r = self.act(self.after_norm(self.trans_conv(x - x_r)))
        x=x + x_r
        # assert not torch.any(torch.isnan(x))
        return x #[b,c,n]
class PointNet(nn.Module):
    def __init__(self, k = 64):
        super(PointNet, self).__init__()
        self.k = k
        self.conv1 = torch.nn.Conv1d(self.k, self.k, 1)
        self.conv2 = torch.nn.Conv1d(self.k, self.k, 1)

        self.bn1 = nn.BatchNorm1d(self.k)
        self.bn2 = nn.BatchNorm1d(self.k)


    def forward(self, x):
        batchsize = x.size()[0]
        n_pts = x.size()[2]

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        return x
