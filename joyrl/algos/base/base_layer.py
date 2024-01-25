import torch
import torch.nn as nn
import torch.nn.functional as F

activation_dics = {'relu': nn.ReLU, 
                   'tanh': nn.Tanh, 
                   'sigmoid': nn.Sigmoid, 
                   'softmax': nn.Softmax,
                   'leakyrelu': nn.LeakyReLU,
                   'none': nn.Identity}  
pooling_dict = {
    'avg2d': nn.AvgPool2d,
    'max2d': nn.MaxPool2d
}
norm_dict = {
    'layernorm': nn.LayerNorm
}

class LayerConfig:
    ''' layer config class
    '''
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

def get_output_size_with_batch(layers,input_size,dtype=torch.float):
    ''' get output size of a layer with batch size 
    '''
    with torch.no_grad():
        x = torch.zeros([10] + list(input_size[1:]), dtype=dtype)
        out = layers(x)
    output_size = [None] + list(out.size())[1:]
    return output_size 
         
def embedding_layer(input_size, layer_cfg: LayerConfig):
    n_embeddings = layer_cfg.n_embeddings
    embedding_dim = layer_cfg.embedding_dim
    class EmbeddingLayer(nn.Module):
        def __init__(self, n_embeddings, embedding_dim):
            super(EmbeddingLayer, self).__init__()
            self.layer = nn.Embedding(num_embeddings=n_embeddings, embedding_dim=embedding_dim)

        def forward(self, x: torch.Tensor):
            if x.dtype != torch.int:
                x = x.int()
            return self.layer(x)
    layer = EmbeddingLayer(n_embeddings, embedding_dim)
    output_size = get_output_size_with_batch(layer, input_size=input_size, dtype=torch.long)
    return layer, output_size 

class LowRankLinear(nn.Module):
    ''' LoRA linear layer, rank must be smaller than both input_dim and output_dim
    '''
    def __init__(self, input_dim, output_dim, rank):
        super(LowRankLinear, self).__init__()
        ''' input_dim and output_dim are the dimensions of the original linear layer
        '''
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.rank = rank

        self.U = nn.Parameter(torch.randn(output_dim, rank))
        self.V = nn.Parameter(torch.randn(rank, input_dim))

    def forward(self, x):
        weight = self.U @ self.V
        return F.linear(x, weight)

def linear_layer(input_size,layer_cfg: LayerConfig):
    ''' linear layer
    '''
    layer_size = layer_cfg.layer_size
    act_name = layer_cfg.activation.lower()
    in_dim = input_size[-1]
    out_dim = layer_size[0]
    layer = nn.Sequential(nn.Linear(in_dim,out_dim),activation_dics[act_name]())
    return layer, [None, out_dim]

def noisy_linear_layer(input_size,layer_cfg: LayerConfig):
    ''' noisy linear layer
    '''
    layer_size = layer_cfg.layer_size
    act_name = layer_cfg.activation.lower()
    std_init = layer_cfg.std_init if hasattr(layer_cfg,'std_init') else 0.4
    in_dim = input_size[-1]
    out_dim = layer_size[0]
    class NoisyLinear(nn.Module):
        ''' Noisy linear module for NoisyNet
        '''
        def __init__(self, in_dim, out_dim, std_init=0.4):
            super(NoisyLinear, self).__init__()
            
            self.in_dim  = in_dim
            self.out_dim = out_dim
            self.std_init  = std_init # std for noise
            self.weight_mu    = nn.Parameter(torch.empty(out_dim, in_dim))
            self.weight_sigma = nn.Parameter(torch.empty(out_dim, in_dim))
            # register tensor as buffer, not a parameter
            self.register_buffer('weight_epsilon', torch.empty(out_dim, in_dim)) 
            self.bias_mu    = nn.Parameter(torch.empty(out_dim))
            self.bias_sigma = nn.Parameter(torch.empty(out_dim))
            self.register_buffer('bias_epsilon', torch.empty(out_dim))
            self.reset_parameters() # reset parameters
            self.reset_noise()  # reset noise
        
        def forward(self, x):
            if self.training: 
                weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
                bias   = self.bias_mu + self.bias_sigma * self.bias_epsilon
            else:
                weight = self.weight_mu
                bias   = self.bias_mu
            return F.linear(x, weight, bias)
        
        def reset_parameters(self):
            mu_range = 1 / self.in_dim ** 0.5
            self.weight_mu.data.uniform_(-mu_range, mu_range)
            self.weight_sigma.data.fill_(self.std_init / self.in_dim ** 0.5)
            self.bias_mu.data.uniform_(-mu_range, mu_range)
            self.bias_sigma.data.fill_(self.std_init / self.out_dim ** 0.5)
        
        def reset_noise(self):
            epsilon_in  = self._scale_noise(self.in_dim)
            epsilon_out = self._scale_noise(self.out_dim)
            self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
            self.bias_epsilon.copy_(self._scale_noise(self.out_dim))
        
        def _scale_noise(self, size):
            x = torch.randn(size)
            x = x.sign().mul(x.abs().sqrt())
            return x
    layer = nn.Sequential(NoisyLinear(in_dim,out_dim,std_init=std_init),activation_dics[act_name]())
    return layer, [None, out_dim]

def dense_layer(in_dim,out_dim,act_name='relu'):
    """ 生成一个全连接层
        layer_size: 全连接层的输入输出维度
        activation: 激活函数
    """
    
def conv2d_layer(input_size, layer_cfg: LayerConfig):
    ''' conv2d layer
    '''
    act_name = layer_cfg.activation.lower()
    in_channel = layer_cfg.in_channel
    out_channel = layer_cfg.out_channel
    kernel_size = layer_cfg.kernel_size if hasattr(layer_cfg,'kernel_size') else 4
    stride = layer_cfg.stride if hasattr(layer_cfg,'stride') else 4
    padding = 'same' if stride == 1 else 'valid'
    class Conv2dLayer(nn.Module):
        def __init__(self,in_channel,out_channel,kernel_size,stride,padding):
            super(Conv2dLayer,self).__init__()
            self.conv = nn.Conv2d(in_channel,out_channel,kernel_size,stride,padding)
        def forward(self,x):
            return self.conv(x)
    cnn_layer = Conv2dLayer(in_channel, out_channel, kernel_size, stride, padding)
    layer = nn.Sequential(cnn_layer, activation_dics[act_name]())
    output_size = get_output_size_with_batch(layer, input_size)
    return layer, output_size

def flatten_layer(input_size, layer_cfg: LayerConfig):
    ''' flatten layer
    '''
    # class FlattenLayer(nn.Module):
    #     def __init__(self):
    #         super(FlattenLayer,self).__init__()
    #     def forward(self,x):
    #         return x.view(x.size(0),-1)
    layer = nn.Sequential(nn.Flatten())
    output_size = get_output_size_with_batch(layer, input_size)
    return layer, output_size

def pooling_layer(input_size, layer_cfg: LayerConfig):
    pooling_type = layer_cfg.pooling_type.lower()
    kernel_size = layer_cfg.kernel_size
    stride = layer_cfg.stride
    padding = layer_cfg.padding
    if pooling_type not in pooling_dict:
        raise KeyError("pooling_type Error! you can add the pooling_type in joyrl/algos/base/base_layer.py")
    layer = pooling_dict[pooling_type](kernel_size=kernel_size, stride=stride, padding=padding)
    output_size = get_output_size_with_batch(layer, input_size)
    return layer, output_size

def norm_layer(input_size, layer_cfg: LayerConfig):
    norm_type = layer_cfg.norm_type.lower()
    normalized_shape = layer_cfg.normalized_shape
    if norm_type not in norm_dict:
        raise KeyError("norm_type Error! you can add the norm_type in joyrl/algos/base/base_layer.py")
    layer = norm_dict[norm_type](normalized_shape=normalized_shape)
    output_size = get_output_size_with_batch(layer, input_size)
    return layer, output_size

def create_layer(input_size: list, layer_cfg: LayerConfig):
    """ 生成一个层
        layer_type: 层的类型
        layer_size: 层的输入输出维度
        activation: 激活函数
    """
    layer_type = layer_cfg.layer_type.lower()
    if layer_type == "linear":
        return linear_layer(input_size, layer_cfg)
    elif layer_type == "noisy_linear":
        return noisy_linear_layer(input_size, layer_cfg)
    elif layer_type == "conv2d":
        return conv2d_layer(input_size, layer_cfg)
    elif layer_type == "embed":
        return embedding_layer(input_size, layer_cfg)
    elif layer_type == "flatten":
        return flatten_layer(input_size, layer_cfg)
    elif layer_type == "pooling":
        return pooling_layer(input_size, layer_cfg)
    elif layer_type == "norm":
        return norm_layer(input_size, layer_cfg)
    elif layer_type == "none":
        return nn.Identity(), input_size
    else:
        raise NotImplementedError