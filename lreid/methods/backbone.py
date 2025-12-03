# This code is modified from https://github.com/facebookresearch/low-shot-shrink-hallucinate

import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from torch.nn.utils import weight_norm

# Basic ResNet model

def init_layer(L):
    # Initialization using fan-in
    if isinstance(L, nn.Conv2d):
        n = L.kernel_size[0]*L.kernel_size[1]*L.out_channels
        L.weight.data.normal_(0,math.sqrt(2.0/float(n)))
    elif isinstance(L, nn.BatchNorm2d):
        L.weight.data.fill_(1)
        L.bias.data.fill_(0)

class distLinear(nn.Module):
    def __init__(self, indim, outdim):
        super(distLinear, self).__init__()
        self.L = weight_norm(nn.Linear(indim, outdim, bias=False), name='weight', dim=0)
        self.relu = nn.ReLU()

    def forward(self, x):
        x_norm = torch.norm(x, p=2, dim =1).unsqueeze(1).expand_as(x)
        x_normalized = x.div(x_norm + 0.00001)
        L_norm = torch.norm(self.L.weight.data, p=2, dim =1).unsqueeze(1).expand_as(self.L.weight.data)
        self.L.weight.data = self.L.weight.data.div(L_norm + 0.00001)
        cos_dist = self.L(x_normalized) #matrix product by forward function
        scores = 10 * cos_dist #a fixed scale factor to scale the output of cos value into a reasonably large input for softmax

        return scores

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


class Linear_fw(nn.Linear): #used in MAML to forward input with fast weight
    def __init__(self, in_features, out_features):
        super(Linear_fw, self).__init__(in_features, out_features)
        self.weight.fast = None #Lazy hack to add fast weight link
        self.bias.fast = None

    def forward(self, x):
        if self.weight.fast is not None and self.bias.fast is not None:
            out = F.linear(x, self.weight.fast, self.bias.fast)
        else:
            out = super(Linear_fw, self).forward(x)
        return out

class Conv2d_fw(nn.Conv2d): #used in MAML to forward input with fast weight
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,padding=0, bias = True):
        super(Conv2d_fw, self).__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)
        self.weight.fast = None
        if not self.bias is None:
            self.bias.fast = None

    def forward(self, x):
        if self.bias is None:
            if self.weight.fast is not None:
                out = F.conv2d(x, self.weight.fast, None, stride= self.stride, padding=self.padding)
            else:
                out = super(Conv2d_fw, self).forward(x)
        else:
            if self.weight.fast is not None and self.bias.fast is not None:
                out = F.conv2d(x, self.weight.fast, self.bias.fast, stride= self.stride, padding=self.padding)
            else:
                out = super(Conv2d_fw, self).forward(x)

        return out

class BatchNorm2d_fw(nn.BatchNorm2d): #used in MAML to forward input with fast weight
    def __init__(self, num_features):
        super(BatchNorm2d_fw, self).__init__(num_features)
        self.weight.fast = None
        self.bias.fast = None

    def forward(self, x):
        running_mean = torch.zeros(x.data.size()[1]).cuda()
        running_var = torch.ones(x.data.size()[1]).cuda()
        if self.weight.fast is not None and self.bias.fast is not None:
            out = F.batch_norm(x, running_mean, running_var, self.weight.fast, self.bias.fast, training = True, momentum = 1)
            #batch_norm momentum hack: follow hack of Kate Rakelly in pytorch-maml/src/layers.py
        else:
            out = F.batch_norm(x, running_mean, running_var, self.weight, self.bias, training = True, momentum = 1)
        return out

# Simple Conv Block
class ConvBlock(nn.Module):
    maml = False #Default
    def __init__(self, indim, outdim, pool = True, padding = 1):
        super(ConvBlock, self).__init__()
        self.indim  = indim
        self.outdim = outdim
        if self.maml:
            self.C      = Conv2d_fw(indim, outdim, 3, padding = padding)
            self.BN     = BatchNorm2d_fw(outdim)
        else:
            self.C      = nn.Conv2d(indim, outdim, 3, padding= padding)
            self.BN     = nn.BatchNorm2d(outdim)
        self.relu   = nn.ReLU(inplace=True)

        self.parametrized_layers = [self.C, self.BN, self.relu]
        if pool:
            self.pool   = nn.MaxPool2d(2)
            self.parametrized_layers.append(self.pool)

        for layer in self.parametrized_layers:
            init_layer(layer)

        self.trunk = nn.Sequential(*self.parametrized_layers)


    def forward(self,x):
        out = self.trunk(x)
        return out

# Simple ResNet Block
class SimpleBlock(nn.Module):
    maml = False #Default
    def __init__(self, indim, outdim, half_res):
        super(SimpleBlock, self).__init__()
        self.indim = indim
        self.outdim = outdim
        if self.maml:
            self.C1 = Conv2d_fw(indim, outdim, kernel_size=3, stride=2 if half_res else 1, padding=1, bias=False)
            self.BN1 = BatchNorm2d_fw(outdim)
            self.C2 = Conv2d_fw(outdim, outdim,kernel_size=3, padding=1,bias=False)
            self.BN2 = BatchNorm2d_fw(outdim)
        else:
            self.C1 = nn.Conv2d(indim, outdim, kernel_size=3, stride=2 if half_res else 1, padding=1, bias=False)
            self.BN1 = nn.BatchNorm2d(outdim)
            self.C2 = nn.Conv2d(outdim, outdim,kernel_size=3, padding=1,bias=False)
            self.BN2 = nn.BatchNorm2d(outdim)
        self.relu1 = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)

        self.parametrized_layers = [self.C1, self.C2, self.BN1, self.BN2]

        self.half_res = half_res


        # if the input number of channels is not equal to the output, then need a 1x1 convolution
        if indim!=outdim:
            if self.maml:
                self.shortcut = Conv2d_fw(indim, outdim, 1, 2 if half_res else 1, bias=False)
                self.BNshortcut = BatchNorm2d_fw(outdim)
            else:
                self.shortcut = nn.Conv2d(indim, outdim, 1, 2 if half_res else 1, bias=False)
                self.BNshortcut = nn.BatchNorm2d(outdim)

            self.parametrized_layers.append(self.shortcut)
            self.parametrized_layers.append(self.BNshortcut)
            self.shortcut_type = '1x1'
        else:
            self.shortcut_type = 'identity'

        for layer in self.parametrized_layers:
            init_layer(layer)

    def forward(self, x):
        out = self.C1(x)
        out = self.BN1(out)
        out = self.relu1(out)
        out = self.C2(out)
        out = self.BN2(out)

        short_out = x if self.shortcut_type == 'identity' else self.BNshortcut(self.shortcut(x))
        out = out + short_out
        out = self.relu2(out)
        return out

# Bottleneck block
class BottleneckBlock(nn.Module):
    maml = False #Default
    def __init__(self, indim, outdim, half_res):
        super(BottleneckBlock, self).__init__()
        bottleneckdim = int(outdim/4)
        self.indim = indim
        self.outdim = outdim
        if self.maml:
            self.C1 = Conv2d_fw(indim, bottleneckdim, kernel_size=1,  bias=False)
            self.BN1 = BatchNorm2d_fw(bottleneckdim)
            self.C2 = Conv2d_fw(bottleneckdim, bottleneckdim, kernel_size=3, stride=2 if half_res else 1,padding=1)
            self.BN2 = BatchNorm2d_fw(bottleneckdim)
            self.C3 = Conv2d_fw(bottleneckdim, outdim, kernel_size=1, bias=False)
            self.BN3 = BatchNorm2d_fw(outdim)
        else:
            self.C1 = nn.Conv2d(indim, bottleneckdim, kernel_size=1,  bias=False)
            self.BN1 = nn.BatchNorm2d(bottleneckdim)
            self.C2 = nn.Conv2d(bottleneckdim, bottleneckdim, kernel_size=3, stride=2 if half_res else 1,padding=1)
            self.BN2 = nn.BatchNorm2d(bottleneckdim)
            self.C3 = nn.Conv2d(bottleneckdim, outdim, kernel_size=1, bias=False)
            self.BN3 = nn.BatchNorm2d(outdim)

        self.relu = nn.ReLU()
        self.parametrized_layers = [self.C1, self.BN1, self.C2, self.BN2, self.C3, self.BN3]
        self.half_res = half_res

        # if the input number of channels is not equal to the output, then need a 1x1 convolution
        if indim!=outdim:
            if self.maml:
                self.shortcut = Conv2d_fw(indim, outdim, 1, stride=2 if half_res else 1, bias=False)
            else:
                self.shortcut = nn.Conv2d(indim, outdim, 1, stride=2 if half_res else 1, bias=False)

            self.parametrized_layers.append(self.shortcut)
            self.shortcut_type = '1x1'
        else:
            self.shortcut_type = 'identity'

        for layer in self.parametrized_layers:
            init_layer(layer)

    def forward(self, x):

        short_out = x if self.shortcut_type == 'identity' else self.shortcut(x)
        out = self.C1(x)
        out = self.BN1(out)
        out = self.relu(out)
        out = self.C2(out)
        out = self.BN2(out)
        out = self.relu(out)
        out = self.C3(out)
        out = self.BN3(out)
        out = out + short_out

        out = self.relu(out)
        return out

class ConvNet(nn.Module):
    def __init__(self, depth, flatten = True):
        super(ConvNet,self).__init__()
        self.grads = []
        self.fmaps = []
        trunk = []
        for i in range(depth):
            indim = 3 if i == 0 else 64
            outdim = 64
            B = ConvBlock(indim, outdim, pool = ( i <4 ) ) #only pooling for fist 4 layers
            trunk.append(B)

        if flatten:
            trunk.append(Flatten())

        self.trunk = nn.Sequential(*trunk)
        if flatten:
            self.final_feat_dim = 1600
        else:
            self.final_feat_dim = [64, 5, 5]

    def forward(self,x):
        out = self.trunk(x)
        return out

class ResNet(nn.Module):
    maml = True #Default
    def __init__(self, block, list_of_num_layers, list_of_out_dims, flatten=True):
        # list_of_num_layers specifies number of layers in each stage
        # list_of_out_dims specifies number of output channel for each stage
        super(ResNet, self).__init__()
        self.grads = []
        self.fmaps = []
        assert len(list_of_num_layers)==4, 'Can have only four stages'

        # initial layers
        if self.maml:
            conv1 = Conv2d_fw(3, 64, kernel_size=7, stride=2, padding=3,
                                               bias=False)
            bn1 = BatchNorm2d_fw(64)
        else:
            conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                                               bias=False)
            bn1 = nn.BatchNorm2d(64)
        relu = nn.ReLU(inplace=True)
        pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        init_layer(conv1)
        init_layer(bn1)

        # residual blocks
        trunk = [conv1, bn1, relu, pool1]
        indim = 64
        for i in range(4):
            for j in range(list_of_num_layers[i]):
                half_res = (i>=1) and (j==0)
                B = block(indim, list_of_out_dims[i], half_res)
                trunk.append(B)
                indim = list_of_out_dims[i]

        # final pooling
        if flatten:
            avgpool = nn.AvgPool2d(7)
            trunk.append(avgpool)
            trunk.append(Flatten())
            self.final_feat_dim = indim
        else:
            self.final_feat_dim = [ indim, 7, 7]

        self.trunk = nn.Sequential(*trunk)

    def forward(self,x):
        out = self.trunk(x)
        return out

def Conv4(flatten=True):
    return ConvNet(4, flatten)

def Conv6(flatten=True):
    return ConvNet(6, flatten)

def ResNet10(flatten=True):
    return ResNet(SimpleBlock, [1,1,1,1],[64,128,256,512], flatten)

def ResNet18(flatten=True):
    return ResNet(SimpleBlock, [2,2,2,2],[64,128,256,512], flatten)

def ResNet34(flatten=True):
    return ResNet(SimpleBlock, [3,4,6,3],[64,128,256,512], flatten)

def ResNet50(flatten=True):
    return ResNet(BottleneckBlock, [3,4,6,3], [256,512,1024,2048], flatten)

def ResNet101(flatten=True):
    return ResNet(BottleneckBlock, [3,4,23,3],[256,512,1024,2048], flatten)

# ==========================================
# 👇 升级版 ViTBase (支持更换模型架构)
# ==========================================

class ViTBase(nn.Module):
    def __init__(self, model_name='vit_base_patch16_224', flatten=True):
        super(ViTBase, self).__init__()
        try:
            import timm
        except ImportError:
            print("Error: Please install timm")
            
        print(f"****** Building ViT Backbone: {model_name} ******")
        
        # 动态创建模型
        self.model = timm.create_model(model_name, pretrained=True, num_classes=0, img_size=(256, 128), drop_path_rate=0.1)
        
        # 自动获取输出维度 (不用手动写 768 了)
        self.final_feat_dim = self.model.num_features
        self.flatten = flatten

    def forward(self, x):
        feat = self.model(x) 
        return feat

# ==========================================
#  👆 添加结束
# ==========================================

# ==========================================
#  添加sdlara
# ==========================================
class SDLoRALinear(nn.Module):
    def __init__(self, original_linear, r=10):
        """
        SD-LoRA Linear Layer
        Args:
            original_linear: The frozen pre-trained linear layer
            r: Rank (paper uses r=10 for ViT-B/16)
        """
        super(SDLoRALinear, self).__init__()
        self.in_features = original_linear.in_features
        self.out_features = original_linear.out_features
        self.r = r

        # 1. 冻结原始权重 (W0)
        self.weight = original_linear.weight
        self.bias = original_linear.bias
        self.weight.requires_grad = False
        if self.bias is not None:
            self.bias.requires_grad = False

        # 2. 存储旧任务的“方向” (Frozen Directions)
        # 我们不存储巨大的 W 矩阵，而是存储 A 和 B，计算时再恢复方向
        # list of (A, B) tuples
        self.past_directions = nn.ModuleList() 
        
        # 3. 存储所有任务的“幅度” (Trainable Magnitudes)
        # alpha_k in Eq. (4). All alphas are trainable end-to-end.
        self.alphas = nn.ParameterList()

        # 4. 当前任务的 A 和 B (Trainable Direction)
        self.current_A = None
        self.current_B = None
        
        # 初始化第一个任务
        self.new_task()

    def new_task(self):
        """
        Start a new task:
        1. If there was a current task, freeze its direction and move to past_directions.
        2. Initialize new trainable A and B for the new task.
        3. Initialize new trainable alpha.
        """
        device = self.weight.device
        
        # --- Step A: Archive previous task (if exists) ---
        if self.current_A is not None:
            # 冻结当前方向
            self.current_A.requires_grad = False
            self.current_B.requires_grad = False
            
            # 存入历史列表 (作为 Module 注册，保证 device 同步)
            # 我们封装进一个 ModuleList 里的 Module 方便管理
            past_pair = nn.Module()
            past_pair.register_buffer('A', self.current_A.data.clone())
            past_pair.register_buffer('B', self.current_B.data.clone())
            self.past_directions.append(past_pair)

        # --- Step B: Create new task parameters ---
        # Initialize A, B (Direction)
        # Paper Section 3.4: "entries of A0 and B0 are i.i.d. according to N(0, sigma_1)"
        # Standard LoRA init: A=Kaiming, B=0. 
        # SD-LoRA requires non-zero start to have a direction? 
        # Let's stick to standard LoRA init first to be safe, or small random.
        new_A = nn.Parameter(torch.randn(self.in_features, self.r).to(device) * 0.01) # Small random
        new_B = nn.Parameter(torch.zeros(self.r, self.out_features).to(device))       # Zero init
        
        self.current_A = new_A
        self.current_B = new_B
        
        # Initialize Alpha (Magnitude)
        # Paper Section 3.3: "learned magnitudes, all initialized to ones"
        new_alpha = nn.Parameter(torch.tensor(1.0).to(device))
        self.alphas.append(new_alpha)

    def get_normalized_update(self, A, B):
        """
        Compute Normalized Direction: \bar{AB} = AB / ||AB||_F
        """
        # AB shape: [in, r] x [r, out] -> [in, out]
        # 注意：Linear 层的 weight 通常是 [out, in]，这里我们需要根据 pytorch 习惯调整转置
        # PyTorch F.linear(x, W) computes xW^T + b. 
        # So if we want AB to be added to W^T, A should be [in, r], B should be [r, out].
        
        delta_W = A @ B # [in, out]
        norm = torch.norm(delta_W, p='fro') + 1e-6
        return delta_W / norm

    def forward(self, x):
        # 1. 基础输出 W0 * x
        out = F.linear(x, self.weight, self.bias)
        
        # 2. 加上旧任务的贡献 (alpha_k * dir_k * x)
        # Eq. (4): sum( alpha_k * normalized(A_k B_k) )
        for i, past_pair in enumerate(self.past_directions):
            alpha = self.alphas[i]
            norm_dir = self.get_normalized_update(past_pair.A, past_pair.B)
            # x: [batch, in], norm_dir: [in, out]
            out += alpha * (x @ norm_dir)

        # 3. 加上当前任务的贡献
        if self.current_A is not None:
            current_alpha = self.alphas[-1]
            norm_dir = self.get_normalized_update(self.current_A, self.current_B)
            out += current_alpha * (x @ norm_dir)
            
        return out

class ViTSDLora(nn.Module):
    def __init__(self, model_name='vit_base_patch16_224', flatten=True, r=10):
        super(ViTSDLora, self).__init__()
        try:
            import timm
        except ImportError:
            print("Error: Please install timm")
            
        print(f"****** Building SD-LoRA ViT: {model_name} (r={r}) ******")
        # 1. 加载预训练 ViT
        self.model = timm.create_model(model_name, pretrained=True, num_classes=0, img_size=(256, 128), drop_path_rate=0.1)
        self.final_feat_dim = self.model.num_features
        self.flatten = flatten
        self.r = r

        # 2. 注入 SD-LoRA 层
        # Paper Section 4.1: "components are inserted into the attention layers... modifying query and value projections"
        self.inject_sd_lora(self.model)
        
        self.count_parameters()

    def inject_sd_lora(self, model):
        self.lora_layers = [] # Keep track to update tasks later
        for i, block in enumerate(model.blocks):
            # 替换 Query 和 Value
            # timm 的 qkv 是一个 Linear 层，我们需要拆解它或者 wrap 它
            # timm implement qkv as one layer: nn.Linear(dim, dim * 3)
            # 这比较麻烦，我们需要 tricky 一点：
            # 方案：只替换 qkv 层
            
            if hasattr(block.attn, 'qkv'):
                # Original qkv
                org_qkv = block.attn.qkv
                # SD-LoRA wrapper
                # 注意：qkv 是 [dim, dim*3]，我们这里简单起见，对整个 qkv 矩阵做 LoRA
                # 也就是 rank 作用于 dim*3。这也符合 SD-LoRA 的通用性。
                block.attn.qkv = SDLoRALinear(org_qkv, r=self.r)
                self.lora_layers.append(block.attn.qkv)
                
            # 如果你想严格复现论文 "query and value projections"，
            # 由于 timm 合并了 qkv，我们需要自己重写 Attention forward，或者接受对 qkv 同时做 LoRA。
            # 鉴于工程复杂度，对整个 qkv 做 LoRA 是最快且有效的近似。

    def new_task(self):
        """
        通知所有 LoRA 层：开始新任务了！
        冻结当前 params，分配新 params。
        """
        print("==> SD-LoRA: Switching to New Task mode...")
        for layer in self.lora_layers:
            layer.new_task()
        self.count_parameters()

    def count_parameters(self):
        trainable_params = 0
        all_param = 0
        for name, param in self.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        print(f"Total params: {all_param / 1e6:.2f}M")
        print(f"Trainable params: {trainable_params / 1e6:.2f}M")
        print(f"Ratio: {trainable_params / all_param * 100:.2f}%")

    def forward(self, x):
        return self.model(x)
# ==========================================
#  👆 添加结束
# ==========================================

model_dict = dict(Conv4 = Conv4,
                  Conv6 = Conv6,
                  ResNet10 = ResNet10,
                  ResNet18 = ResNet18,
                  ResNet34 = ResNet34,
                  ResNet50 = ResNet50,
                  ResNet101 = ResNet101,
                  # 注册不同的 ViT 配置
                  vit_base = lambda flatten: ViTBase('vit_base_patch16_224', flatten),   # 经典款
                  # 👇 新增 SD-LoRA
                  vit_sd_lora = lambda flatten: ViTSDLora('vit_base_patch16_224', flatten, r=10),
                  vit_small = lambda flatten: ViTBase('vit_small_patch16_224', flatten), # 轻量款 (老师可能想看这个)
                  deit_base = lambda flatten: ViTBase('deit_base_patch16_224', flatten), # 蒸馏版 (效果通常更好)
                  swin_base = lambda flatten: ViTBase('swin_base_patch4_window7_224', flatten) # Swin Transformer (SOTA 级)
)