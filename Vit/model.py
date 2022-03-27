import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionEmbs(nn.Module): # 位置编码模块
    def __init__(self, num_patches, emb_dim, dropout_rate=0.1):
        super(PositionEmbs, self).__init__()
        # 随机初始化位置信息，pos_embedding通过训练学习
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, emb_dim))
        if dropout_rate > 0:
            self.dropout = nn.Dropout(dropout_rate)
        else:
            self.dropout = None

    def forward(self, x):
        out = x + self.pos_embedding
        if self.dropout:
            out = self.dropout(out)
        return out


class MlpBlock(nn.Module): # 前馈网络模块
    def __init__(self, in_dim, mlp_dim, out_dim, dropout_rate=0.1):
        super(MlpBlock, self).__init__()
        self.fc1 = nn.Linear(in_dim, mlp_dim)
        self.fc2 = nn.Linear(mlp_dim, out_dim)
        # GELU激活函数
        self.act = nn.GELU()
        if dropout_rate > 0.0:
            self.dropout1 = nn.Dropout(dropout_rate)
            self.dropout2 = nn.Dropout(dropout_rate)
        else:
            self.dropout1 = None
            self.dropout2 = None

    def forward(self, x):
        out = self.fc1(x)
        out = self.act(out)
        if self.dropout1:
            out = self.dropout1(out)
        out = self.fc2(out)
        out = self.dropout2(out)
        return out

class LinearGeneral(nn.Module): # 矩阵乘法模块
    def __init__(self, in_dim=(768,), feat_dim=(12, 64)):
        super(LinearGeneral, self).__init__()
        self.weight = nn.Parameter(torch.randn(*in_dim, *feat_dim))
        self.bias = nn.Parameter(torch.zeros(*feat_dim))

    def forward(self, x, dims):
        a = torch.tensordot(x, self.weight, dims=dims) + self.bias
        return a


class SelfAttention(nn.Module): # 自注意力机制模块
    def __init__(self, in_dim, heads=8, dropout_rate=0.1):
        super(SelfAttention, self).__init__()
        self.heads = heads
        self.head_dim = in_dim // heads
        self.scale = self.head_dim ** 0.5

        self.query = LinearGeneral((in_dim,), (self.heads, self.head_dim))
        self.key = LinearGeneral((in_dim,), (self.heads, self.head_dim))
        self.value = LinearGeneral((in_dim,), (self.heads, self.head_dim))
        self.out = LinearGeneral((self.heads, self.head_dim), (in_dim,))

        if dropout_rate > 0:
            self.dropout = nn.Dropout(dropout_rate)
        else:
            self.dropout = None

    def forward(self, x):
        # weight=[768,12,64]，b=[12,64]，x=[B,257,768]
        # tensordot(x,weight,dims)+bias
        # 即对257个768维的vector降维到64，一共12个heads
        q = self.query(x, dims=([2], [0])) # [B,257,12,64]
        k = self.key(x, dims=([2], [0])) # [B,257,12,64]
        v = self.value(x, dims=([2], [0])) # [B,257,12,64]
        # [B,12,257,64]代表有该批次的图像数量为B，12个heads，每个图片有257个vector，每个vector的维度为64
        q = q.permute(0, 2, 1, 3) # [B,12,257,64]
        k = k.permute(0, 2, 1, 3) # [B,12,257,64]
        v = v.permute(0, 2, 1, 3) # [B,12,257,64]
        # [257,64]*[64,257]=[257,257]
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / self.scale # [B,12,257,257]
        attn_weights = F.softmax(attn_weights, dim=-1) # [B,12,257,257]
        # [257,257]*[257,64]=[257,64]
        out = torch.matmul(attn_weights, v) # [B,12,257,64]
        out = out.permute(0, 2, 1, 3) # [B,257,12,64]
        # weight=[12,64,768]，b=[768]，out=[B,257,12,64]
        # tensordot(out,weight,dims)+bias
        out = self.out(out, dims=([2, 3], [0, 1])) # [B,257,768]

        return out


class EncoderBlock(nn.Module):
    def __init__(self, in_dim, mlp_dim, num_heads, dropout_rate=0.1, attn_dropout_rate=0.1):
        super(EncoderBlock, self).__init__()

        self.norm1 = nn.LayerNorm(in_dim)
        self.attn = SelfAttention(in_dim, heads=num_heads, dropout_rate=attn_dropout_rate)
        if dropout_rate > 0:
            self.dropout = nn.Dropout(dropout_rate)
        else:
            self.dropout = None
        self.norm2 = nn.LayerNorm(in_dim)
        self.mlp = MlpBlock(in_dim, mlp_dim, in_dim, dropout_rate)

    def forward(self, x):
        # [B,257,768]=>[B,257,768]
        residual = x
        out = self.norm1(x)
        out = self.attn(out)
        if self.dropout:
            out = self.dropout(out)
        out += residual
        residual = out

        out = self.norm2(out)
        out = self.mlp(out)
        out += residual
        return out


class Encoder(nn.Module):
    def __init__(self, num_patches, emb_dim, mlp_dim, num_layers=12, num_heads=12, dropout_rate=0.1,
                 attn_dropout_rate=0.0):
        super(Encoder, self).__init__()

        # positional embedding
        self.pos_embedding = PositionEmbs(num_patches, emb_dim, dropout_rate)

        # encoder blocks
        in_dim = emb_dim
        self.encoder_layers = nn.ModuleList()
        for i in range(num_layers):
            layer = EncoderBlock(in_dim, mlp_dim, num_heads, dropout_rate, attn_dropout_rate)
            self.encoder_layers.append(layer)
        self.norm = nn.LayerNorm(in_dim)

    def forward(self, x):
        # [B,257,768]=>[B,257,768]
        out = self.pos_embedding(x)
        
        for layer in self.encoder_layers:
            out = layer(out)

        out = self.norm(out)
        return out


class VisionTransformer(nn.Module):

    def __init__(self,
                 image_size=(256, 256), # 原图尺寸
                 patch_size=(16, 16), # 每次patch尺寸
                 emb_dim=768, # 每个pctch转换为vector后的维度768=3*16*16
                 mlp_dim=3072, # 前馈神经网络隐藏层神经元个数
                 num_heads=12, # 自注意力机制多头数量
                 num_layers=12, # 网络层数
                 num_classes=5, # 类别个数
                 attn_dropout_rate=0.0,
                 dropout_rate=0.1):
        super(VisionTransformer, self).__init__()
        # 原图像宽高，图像大小为[B,3,256,256]
        h, w = image_size
        # 每个patch宽高，每一块大小为[B,3,16,16]
        fh, fw = patch_size
        # 计算patch数量，num_patches=16*16=256
        gh, gw = h // fh, w // fw
        num_patches = gh * gw
        # 用卷积提取出patch[N,3,256,256]=>[B,emb_dim,16,16]，emb_dim=768=3*16*16
        # embedding里每一个值代表一个长度为emb_dim的vector，一共有16*16=256个vector
        self.embedding = nn.Conv2d(3, emb_dim, kernel_size=(fh, fw), stride=(fh, fw))
        # 每组seq增加一个cls_token，大小为[1,1,emb_dim]
        self.cls_token = nn.Parameter(torch.zeros(1, 1, emb_dim))

        # transformer
        self.transformer = Encoder(
            num_patches=num_patches,
            emb_dim=emb_dim,
            mlp_dim=mlp_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout_rate=dropout_rate,
            attn_dropout_rate=attn_dropout_rate)

        # classfier
        self.classifier = nn.Linear(emb_dim, num_classes)

    def forward(self, x):
        emb = self.embedding(x)  # [B,3,256,256]=>[B,768,16,16]
        emb = emb.permute(0, 2, 3, 1)  # [B,768,16,16]=>[B,16,16,768]
        b, h, w, c = emb.shape
        emb = emb.reshape(b, h * w, c) # [B,16,16,768]=>[B,256,768]，批次为B，256个vector，每个vector768维

        # cls_token拼接
        cls_token = self.cls_token.repeat(b, 1, 1) # cls_token在dim=0上复制B(批次)份
        emb = torch.cat([cls_token, emb], dim=1) # 与原emb拼接在一起，[B,257,768]

        # transformer
        feat = self.transformer(emb) # [B, 257, 768]

        # classifier
        logits = self.classifier(feat[:, 0]) # 取出cls_token进行分类，即feat[:, 0]=[B,768]
        return logits