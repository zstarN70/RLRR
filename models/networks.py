import math
import torch
import torch.nn as nn
import numpy as np
from torch.nn.modules.utils import _pair
from scipy import ndimage
import torch.nn.functional as F
from models import configs

ATTENTION_Q = "MultiHeadDotProductAttention_1/query"
ATTENTION_K = "MultiHeadDotProductAttention_1/key"
ATTENTION_V = "MultiHeadDotProductAttention_1/value"
ATTENTION_OUT = "MultiHeadDotProductAttention_1/out"
FC_0 = "MlpBlock_3/Dense_0"
FC_1 = "MlpBlock_3/Dense_1"
ATTENTION_NORM = "LayerNorm_0"
MLP_NORM = "LayerNorm_2"

CONFIGS = {
    'ViT-B_16': configs.get_b16_config(),
    'ViT-B_32': configs.get_b32_config(),
    'ViT-L_16': configs.get_l16_config(),
    'ViT-L_32': configs.get_l32_config(),
    'ViT-H_14': configs.get_h14_config(),
    'R50-ViT-B_16': configs.get_r50_b16_config(),
    'testing': configs.get_testing(),
}


def np2th(weights, conv=False):
    """Possibly convert HWIO to OIHW."""
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)


class RLRRLinear(nn.Module):
    def __init__(self, size_in, size_out, bias=True, enable_rlrr=False):
        super(RLRRLinear, self).__init__()
        self.enable_rlrr = enable_rlrr
        self.size_in = size_in
        self.size_out = size_out
        self.has_bias = bias
        self.mlp = nn.Linear(size_in, size_out, bias=bias)
        if self.enable_rlrr:
            self.scale_col = nn.Parameter(torch.empty(1, self.size_in), requires_grad=True)
            self.scale_line = nn.Parameter(torch.empty(self.size_out, 1), requires_grad=True)
            self.shift_bias = nn.Parameter(torch.empty(1, self.size_out), requires_grad=True)

            # SSF
            # nn.init.normal_(self.scale_line, mean=1, std=0.02)
            # nn.init.normal_(self.scale_col, mean=1, std=0.02)
            # nn.init.normal_(self.shift_bias, mean=0, std=0.02)

            # our
            nn.init.kaiming_uniform_(self.scale_col)
            nn.init.kaiming_uniform_(self.scale_line)
            nn.init.zeros_(self.shift_bias)

        self._frozen_param()

    def _frozen_param(self):
        for param in self.mlp.parameters():
            param.requires_grad = False

    def forward(self, x):           
        if self.enable_rlrr:
            weight = (self.mlp.weight * self.scale_col * self.scale_line) + self.mlp.weight
            bias = self.mlp.bias + self.shift_bias
            return F.linear(x, weight, bias)
        else:
            return self.mlp(x)


class RLRRAttention(nn.Module):
    def __init__(self, config, vis, enable_rlrr=True):
        super(RLRRAttention, self).__init__()
        self.vis = vis
        self.num_attention_heads = config.transformer["num_heads"]
        self.attention_head_size = int(config.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = RLRRLinear(config.hidden_size, self.all_head_size, bias=True, enable_rlrr=enable_rlrr)
        self.key = RLRRLinear(config.hidden_size, self.all_head_size, bias=True, enable_rlrr=enable_rlrr)
        self.value = RLRRLinear(config.hidden_size, self.all_head_size, bias=True, enable_rlrr=enable_rlrr)
        self.out = RLRRLinear(config.hidden_size, config.hidden_size, bias=True, enable_rlrr=enable_rlrr)
        self.attn_dropout = nn.Dropout(config.transformer["attention_dropout_rate"])
        self.proj_dropout = nn.Dropout(config.transformer["attention_dropout_rate"])
        self.softmax = nn.Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        # weights = attention_probs if self.vis else None
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output  # , weights


class RLRRMLP(nn.Module):
    def __init__(self, config, enable_rlrr=True):
        super(RLRRMLP, self).__init__()
        self.fc1 = RLRRLinear(config.hidden_size, config.transformer["mlp_dim"], bias=True, enable_rlrr=enable_rlrr)
        self.fc2 = RLRRLinear(config.transformer["mlp_dim"], config.hidden_size, bias=True, enable_rlrr=enable_rlrr)

        self.act = nn.GELU()
        self.dropout = nn.Dropout(config.transformer["dropout_rate"])

    def forward(self, x):
        x = self.act(self.fc1(x))
        if self.training:
            x = self.dropout(x)
        x = self.fc2(x)
        if self.training:
            x = self.dropout(x)
        return x
    
def init_ssf_scale_shift(dim):
    scale = nn.Parameter(torch.ones(dim))
    shift = nn.Parameter(torch.zeros(dim))

    nn.init.normal_(scale, mean=1, std=.02)
    nn.init.normal_(shift, std=.02)

    return scale, shift


def ssf_ada(x, scale, shift):
    assert scale.shape == shift.shape
    if x.shape[-1] == scale.shape[0]:
        return x * scale + shift
    elif x.shape[1] == scale.shape[0]:
        return x * scale.view(1, -1, 1, 1) + shift.view(1, -1, 1, 1)
    else:
        raise ValueError('the input tensor shape does not match the shape of the scale factor.')

class RLRRBlock(nn.Module):
    def __init__(self, config, vis, drop_path=0.0, enable_rlrr=True):
        super(RLRRBlock, self).__init__()
        self.hidden_size = config.hidden_size
        self.attention_norm = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn_norm = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn = RLRRMLP(config, enable_rlrr=enable_rlrr)
        self.attn = RLRRAttention(config, vis, enable_rlrr=enable_rlrr)
        self.enable_rlrr = enable_rlrr
        self.drop_path1 = nn.Dropout(p=drop_path)
        self.drop_path2 = nn.Dropout(p=drop_path)
        if self.enable_rlrr:
            self.ssf_scale_attn_norm, self.ssf_shift_attn_norm = init_ssf_scale_shift(self.hidden_size)
            self.ssf_scale_ffn_norm, self.ssf_shift_ffn_norm = init_ssf_scale_shift(self.hidden_size)

    def forward(self, x):
        if self.enable_rlrr:
            x = x + self.drop_path1(self.attn(ssf_ada(self.attention_norm(x), self.ssf_scale_attn_norm, self.ssf_shift_attn_norm)))
            x = x + self.drop_path2(self.ffn(ssf_ada(self.ffn_norm(x), self.ssf_scale_ffn_norm, self.ssf_shift_ffn_norm)))
        else:
            x = x + self.drop_path1(self.attn(self.attention_norm(x)))
            x = x + self.drop_path2(self.ffn(self.ffn_norm(x)))
        return x

    def _fc_load_weight(self, ROOT, Key, Weights, unit):
        mat_weights = Weights[ROOT + '/' + Key + '/' + "kernel"]
        mat_bias = Weights[ROOT + '/' + Key + '/' + "bias"]
        if Key == ATTENTION_OUT:
            mat_weights = mat_weights.reshape(-1, mat_weights.shape[-1])
        else:
            mat_weights = mat_weights.reshape(mat_weights.shape[0], -1)

        unit.mlp.weight.copy_(np2th(mat_weights).t())
        unit.mlp.bias.copy_(np2th(mat_bias).view(-1))

    def load_from(self, weights, n_block):
        ROOT = f"Transformer/encoderblock_{n_block}"
        with torch.no_grad():
            self._fc_load_weight(ROOT, ATTENTION_Q, weights, self.attn.query)
            self._fc_load_weight(ROOT, ATTENTION_K, weights, self.attn.key)
            self._fc_load_weight(ROOT, ATTENTION_V, weights, self.attn.value)
            self._fc_load_weight(ROOT, ATTENTION_OUT, weights, self.attn.out)
            self._fc_load_weight(ROOT, FC_0, weights, self.ffn.fc1)
            self._fc_load_weight(ROOT, FC_1, weights, self.ffn.fc2)

            self.attention_norm.weight.copy_(np2th(weights[ROOT + '/' + ATTENTION_NORM + '/' + "scale"]))
            self.attention_norm.bias.copy_(np2th(weights[ROOT + '/' + ATTENTION_NORM + '/' + "bias"]))
            self.ffn_norm.weight.copy_(np2th(weights[ROOT + '/' + MLP_NORM + '/' + "scale"]))
            self.ffn_norm.bias.copy_(np2th(weights[ROOT + '/' + MLP_NORM + '/' + "bias"]))


class RLRREncoder(nn.Module):
    def __init__(self, config, vis, drop_path=0.0, enable_rlrr=True):
        super(RLRREncoder, self).__init__()
        self.vis = vis
        self.enable_rlrr = enable_rlrr
        self.layer = nn.ModuleList()
        self.encoder_norm = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.num_blocks = config.transformer["num_layers"]
        # fellow SSF
        dpr = [x.item() for x in torch.linspace(0, drop_path, self.num_blocks)]  # stochastic depth decay rule
        for i in range(config.transformer["num_layers"]):
            self.layer.append(RLRRBlock(config, vis, drop_path=dpr[i], enable_rlrr=enable_rlrr))
        
        if self.enable_rlrr:
            self.ssf_scale_enc_norm, self.ssf_shift_enc_norm = init_ssf_scale_shift(config.hidden_size)

    def forward(self, hidden_states):
        for layer_block in self.layer:
            hidden_states = layer_block(hidden_states)
        encoded = self.encoder_norm(hidden_states)
        if self.enable_rlrr: 
            encoded = ssf_ada(encoded, self.ssf_scale_enc_norm, self.ssf_shift_enc_norm)
        return encoded


class RLRREmbeddings(nn.Module):
    def __init__(self, config, img_size, in_channels=3, enable_rlrr=True):
        super(RLRREmbeddings, self).__init__()
        self.hybrid = None
        img_size = _pair(img_size)

        patch_size = _pair(config.patches["size"])
        n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
        self.hybrid = False

        self.patch_embeddings = nn.Conv2d(in_channels=in_channels,
                                          out_channels=config.hidden_size,
                                          kernel_size=patch_size,
                                          stride=patch_size)
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches + 1, config.hidden_size))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))

        self.dropout = nn.Dropout(config.transformer["dropout_rate"])
        self.enable_rlrr = enable_rlrr
        if self.enable_rlrr:
            self.ssf_scale_patch, self.ssf_shift_patch = init_ssf_scale_shift(config.hidden_size)

    def forward(self, x):
        B = x.size(0)
        cls_tokens = self.cls_token.expand(B, -1, -1)

        x = self.patch_embeddings(x)
        x = x.flatten(2)
        x = x.transpose(-1, -2)
        if self.enable_rlrr:  
            x = ssf_ada(x, self.ssf_scale_patch, self.ssf_shift_patch)
        x = torch.cat((cls_tokens, x), dim=1)
        embeddings = x + self.position_embeddings
        
        return embeddings


class RLRRTransformer(nn.Module):
    def __init__(self, config, img_size, vis, drop_path=0.0, enable_rlrr=True):
        super(RLRRTransformer, self).__init__()
        self.embeddings = RLRREmbeddings(config, img_size=img_size)
        self.encoder = RLRREncoder(config, vis, drop_path=drop_path, enable_rlrr=enable_rlrr)
        self._frozen_param()
        self.enable_rlrr = enable_rlrr

    def _frozen_param(self):
        for param in self.embeddings.parameters():
            param.requires_grad = False


    def forward(self, input_ids):
        embedding_output = self.embeddings(input_ids)
        encoded = self.encoder(embedding_output)
        return encoded


class RLRRVisionTransformer(nn.Module):
    def __init__(self, config, img_size=224, num_classes=21843, zero_head=False, vis=False, enable_rlrr=True, drop_path=0.0):
        super(RLRRVisionTransformer, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.classifier = config.classifier
        self.transformer = RLRRTransformer(config, img_size, vis, drop_path=drop_path, enable_rlrr=enable_rlrr)
        self.head = nn.Linear(config.hidden_size, num_classes)
        self.loss_fct = nn.CrossEntropyLoss()

    def get_parameters(self, lr, weight_decay):
        wd_params = []
        no_wd_params = []
        for name, param in self.named_parameters():
            if 'bias' in name or 'norm' in name:
                no_wd_params.append(param)
            else:
                wd_params.append(param)

        params = [
            {"params": wd_params, "lr": lr, "weight_decay": weight_decay},
            {"params": no_wd_params, "lr": lr, "weight_decay": 0.}
        ]

        return params

    def forward(self, x, labels=None):
        x = self.transformer(x)

        logits = self.head(x[:, 0])
        if labels is not None:
            loss = self.loss_fct(logits.view(-1, self.num_classes), labels.view(-1))
            return loss
        else:
            return logits

    def load_from(self, weights):
        with torch.no_grad():
            if self.zero_head:
                nn.init.zeros_(self.head.weight)
                nn.init.zeros_(self.head.bias)
            else:
                self.head.weight.copy_(np2th(weights["head/kernel"]).t())
                self.head.bias.copy_(np2th(weights["head/bias"]).t())

            self.transformer.embeddings.patch_embeddings.weight.copy_(np2th(weights["embedding/kernel"], conv=True))
            self.transformer.embeddings.patch_embeddings.bias.copy_(np2th(weights["embedding/bias"]))
            self.transformer.embeddings.cls_token.copy_(np2th(weights["cls"]))
            self.transformer.encoder.encoder_norm.weight.copy_(np2th(weights["Transformer/encoder_norm/scale"]))
            self.transformer.encoder.encoder_norm.bias.copy_(np2th(weights["Transformer/encoder_norm/bias"]))

            posemb = np2th(weights["Transformer/posembed_input/pos_embedding"])
            posemb_new = self.transformer.embeddings.position_embeddings
            if posemb.size() == posemb_new.size():
                self.transformer.embeddings.position_embeddings.copy_(posemb)
            else:
                print("load_pretrained: resized variant: %s to %s" % (posemb.size(), posemb_new.size()))
                ntok_new = posemb_new.size(1)

                if self.classifier == "token":
                    posemb_tok, posemb_grid = posemb[:, :1], posemb[0, 1:]
                    ntok_new -= 1
                else:
                    posemb_tok, posemb_grid = posemb[:, :0], posemb[0]

                gs_old = int(np.sqrt(len(posemb_grid)))
                gs_new = int(np.sqrt(ntok_new))
                print('load_pretrained: grid-size from %s to %s' % (gs_old, gs_new))
                posemb_grid = posemb_grid.reshape(gs_old, gs_old, -1)

                zoom = (gs_new / gs_old, gs_new / gs_old, 1)
                posemb_grid = ndimage.zoom(posemb_grid, zoom, order=1)
                posemb_grid = posemb_grid.reshape(1, gs_new * gs_new, -1)
                posemb = np.concatenate([posemb_tok, posemb_grid], axis=1)
                self.transformer.embeddings.position_embeddings.copy_(np2th(posemb))

            for bname, block in self.transformer.encoder.named_children():
                for uname, unit in block.named_children():
                    unit.load_from(weights, n_block=uname)



if __name__ == '__main__':
    config = CONFIGS['ViT-B_16']
    model = RLRRVisionTransformer(config, 224, zero_head=False, num_classes=1000)
    m = np.load('a.npz')
    model.load_from(m)

    import torchvision.transforms as transforms
    from torch.utils.data import DataLoader
    from torchvision import datasets
    from train_vtab import valid

    device = torch.device('cuda:0')
    transform_ = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    model.to(device)
    dataset = datasets.ImageFolder('/home/Datasets/image_classification/ImageNet2012/imagenet/val/',
                                   transform=transform_)
    loader = DataLoader(dataset, batch_size=256, num_workers=6)
    valid(model, loader, device)
