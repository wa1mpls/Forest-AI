import torch
import torch.nn as nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import yaml
from typing import Tuple, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PreNorm(nn.Module):
    def __init__(self, dim: int, fn: nn.Module):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    
    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim: int, heads: int = 8, dim_head: int = 64, dropout: float = 0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)
        
        self.heads = heads
        self.scale = dim_head ** -0.5
        
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        
        attn = self.attend(dots)
        attn = self.dropout(attn)
        
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(
        self,
        dim: int,
        depth: int,
        heads: int,
        dim_head: int,
        mlp_dim: int,
        dropout: float = 0.
    ):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class ViT(nn.Module):
    def __init__(
        self,
        *,
        image_size: int,
        patch_size: int,
        num_channels: int,
        dim: int,
        depth: int,
        heads: int,
        mlp_dim: int,
        pool: str = 'cls',
        channels: int = 3,
        dim_head: int = 64,
        dropout: float = 0.,
        emb_dropout: float = 0.
    ):
        super().__init__()
        image_height, image_width = (image_size, image_size) if isinstance(image_size, int) else image_size
        patch_height, patch_width = (patch_size, patch_size) if isinstance(patch_size, int) else patch_size
        
        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'
        
        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
            nn.Linear(patch_dim, dim),
        )
        
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)
        
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        
        self.pool = pool
        self.to_latent = nn.Identity()
        
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, 1)
        )
    
    def forward(self, img: torch.Tensor) -> torch.Tensor:
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape
        
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)
        
        x = self.transformer(x)
        
        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]
        
        x = self.to_latent(x)
        return self.mlp_head(x)

def create_vit_model(config_path: str = "configs/model_config.yaml") -> ViT:
    """
    Create Vision Transformer model from configuration
    
    Args:
        config_path: Path to model configuration file
        
    Returns:
        ViT: Vision Transformer model
    """
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Get ViT parameters
    vit_config = config['vit']
    
    # Create model
    model = ViT(
        image_size=224,
        patch_size=vit_config['patch_size'],
        num_channels=10,  # 6 bands + 3 indices + 1 mask
        dim=vit_config['hidden_size'],
        depth=vit_config['num_layers'],
        heads=vit_config['num_heads'],
        mlp_dim=vit_config['mlp_dim'],
        dropout=vit_config['dropout_rate'],
        emb_dropout=vit_config['attention_dropout_rate']
    )
    
    # Load pretrained weights if specified
    if vit_config['pretrained']:
        try:
            from timm.models.vision_transformer import vit_base_patch16_224
            pretrained_model = vit_base_patch16_224(pretrained=True)
            
            # Copy weights
            model.to_patch_embedding[1].weight.data.copy_(pretrained_model.patch_embed.proj.weight.data)
            model.to_patch_embedding[1].bias.data.copy_(pretrained_model.patch_embed.proj.bias.data)
            model.pos_embedding.data.copy_(pretrained_model.pos_embed.data)
            model.cls_token.data.copy_(pretrained_model.cls_token.data)
            
            # Copy transformer weights
            for i, (attn, ff) in enumerate(model.transformer.layers):
                attn.fn.to_qkv.weight.data.copy_(pretrained_model.blocks[i].attn.qkv.weight.data)
                attn.fn.to_qkv.bias.data.copy_(pretrained_model.blocks[i].attn.qkv.bias.data)
                attn.fn.to_out[0].weight.data.copy_(pretrained_model.blocks[i].attn.proj.weight.data)
                attn.fn.to_out[0].bias.data.copy_(pretrained_model.blocks[i].attn.proj.bias.data)
                ff.fn.net[0].weight.data.copy_(pretrained_model.blocks[i].mlp.fc1.weight.data)
                ff.fn.net[0].bias.data.copy_(pretrained_model.blocks[i].mlp.fc1.bias.data)
                ff.fn.net[3].weight.data.copy_(pretrained_model.blocks[i].mlp.fc2.weight.data)
                ff.fn.net[3].bias.data.copy_(pretrained_model.blocks[i].mlp.fc2.bias.data)
            
            logger.info("Loaded pretrained weights successfully")
        except Exception as e:
            logger.warning(f"Failed to load pretrained weights: {e}")
    
    return model 