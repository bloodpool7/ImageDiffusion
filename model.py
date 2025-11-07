import torch
import torch.nn as nn
import torch.nn.functional as F

class Conditioner(nn.Module):
    def __init__(self, d_time, d_embedding, num_classes = None, p_dropout = 0.0, device = 'cpu'):
        """
        Args:
            d_time: int, dimension of the time encoding
            d_embedding: int, dimension of the output embedding
            num_classes: int, number of classes (if None, then no conditioning is done)
            p_dropout: float, dropout probability for turning labels into the null class (only used if num_classes is not None)
        """
        super().__init__()
        self.d_time = d_time
        self.d_embedding = d_embedding
        self.num_classes = num_classes
        self.p_dropout = p_dropout
        self.device = device
        
        self.time_mlp = nn.Sequential(
            nn.Linear(d_time, d_embedding),
            nn.SiLU(),
            nn.Linear(d_embedding, d_embedding),
        )
        
        self.label_emb = nn.Embedding(num_classes + 1, d_embedding) # +1 for the "no class" class
        nn.init.zeros_(self.label_emb.weight[num_classes])
        self.to(device)
    
    def forward(self, time_encoding, labels=None):
        """
        Args:
            time_encoding: [B, encoding_dim], should already have been encoded with sinusoidal embeddings
            labels: [B], should be masked with idx `self.num_classes` as the null class

        Returns:
            [B, embedding_dim], where embedding_dim = d_embedding
        """
        if labels is None:
            return self.time_mlp(time_encoding)
        return self.time_mlp(time_encoding) + self.label_emb(labels)
    
    def get_training_condition(self, timesteps: torch.Tensor, y: torch.Tensor = None):
        """
        This function is used to get the condition for the training (only used in training NOT FOR INFERENCE).
        If doing unconditional training, y should be None.
        
        Args:
            timesteps: [B]
            y: [B], Shouldn't be masked. Masking happens here

        Returns:
            [B, embedding_dim], where embedding_dim = d_embedding
        """
        time_enc = self.get_sinusoidal_embedding(timesteps, self.d_time)
        if y is None:
            return self(time_enc, None)

        labels_mask = torch.rand(y.shape) < self.p_dropout
        labels = y.clone()
        labels[labels_mask] = self.num_classes
        return self(time_enc, labels)
    
    def get_condition_vector(self, timesteps: torch.Tensor, labels: torch.Tensor = None):
        """
        This function is used to get the condition vector for the model (used for inference)
        """
        time_enc = self.get_sinusoidal_embedding(timesteps, self.d_time)
        if labels is None:
            return self(time_enc, None)
        return self(time_enc, labels)

    @staticmethod
    def get_sinusoidal_embedding(timesteps: torch.Tensor,
                                embedding_dim: int,
                                max_period: float = 10000.0,
                                dtype: torch.dtype = torch.float32):
        """
        Sinusoidal timestep embeddings (Transformer/Fairseq style).

        Args:
            timesteps: 1D int/float tensor of shape [B]
            embedding_dim: total embedding dimension (sin and cos concatenated)
            max_period: controls minimum frequency (default 10000)
            dtype: output dtype (default float32)

        Returns:
            Tensor of shape [B, embedding_dim]
        """
        assert timesteps.ndim == 1, "timesteps must be 1D [B]"

        device = timesteps.device
        half_dim = embedding_dim // 2
        if half_dim == 0:
            # degenerate case, return zeros
            return torch.zeros(timesteps.shape[0], embedding_dim, device=device, dtype=dtype)

        # frequencies: exp(-log(max_period) * i / (half_dim - 1)), i=0..half_dim-1
        # gives range [1, 1/max_period]
        freqs = torch.exp(
            -torch.log(torch.tensor(max_period, device=device, dtype=dtype)) *
            torch.arange(half_dim, device=device, dtype=dtype) / max(half_dim - 1, 1)
        )  # [half_dim]

        # outer product: [B, 1] * [1, half_dim] -> [B, half_dim]
        args = timesteps.to(device=device, dtype=dtype)[:, None] * freqs[None, :]

        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)  # [B, 2*half_dim]
        if embedding_dim % 2 == 1:
            emb = F.pad(emb, (0, 1))  # [B, embedding_dim]
        return emb

class ResidualBlock(nn.Module):
    def __init__(self, *, in_channels, out_channels, embedding_dim, num_groups=32):
        """
        Args:
            in_channels: int, number of input channels
            out_channels: int, number of output channels
            embedding_dim: int, embedding dimension for the condition embedding (same as d_embedding of the condition embedding)
            num_groups: int, number of groups for group norm
        """
        super().__init__() 
        self.in_channels = in_channels 
        self.out_channels = out_channels 
        self.embedding_dim = embedding_dim

        max_groups_1 = min(in_channels, num_groups)
        max_groups_2 = min(out_channels, num_groups)

        #Layers
        self.norm1 = nn.GroupNorm(max_groups_1, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(max_groups_2, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)    

        #Condition injection
        self.emb_proj1 = nn.Linear(embedding_dim, in_channels * 2)
        self.emb_proj2 = nn.Linear(embedding_dim, out_channels * 2)
        nn.init.zeros_(self.emb_proj1.weight); nn.init.zeros_(self.emb_proj1.bias)
        nn.init.zeros_(self.emb_proj2.weight); nn.init.zeros_(self.emb_proj2.bias)

        #Misc
        self.skip = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        self.activation = nn.SiLU()

    def forward(self, x, condition):
        """
        Args:
            x: [B, C, H, W]
            condition: [B, embedding_dim], where embedding_dim = d_embedding of the condition embedding. Expects the condition to be pre-embedded into the embedding_dim.

        Returns:
            [B, C, H, W]
        """
        g1, b1 = self.emb_proj1(F.silu(condition)).chunk(2, dim=1)
        g2, b2 = self.emb_proj2(F.silu(condition)).chunk(2, dim=1)

        h = self.norm1(x)
        h = h * (1 + g1[:, :, None, None]) + b1[:, :, None, None]
        h = self.conv1(self.activation(h))

        h = self.norm2(h)
        h = h * (1 + g2[:, :, None, None]) + b2[:, :, None, None]
        h = self.conv2(self.activation(h))
        
        h = h + self.skip(x)

        return h

class DownSample(nn.Module): 
    def __init__(self, in_channels, out_channels): 
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, stride=2,padding=1)

    def forward(self, x): 
        return self.conv(x)

class Upsample(nn.Module): 
    def __init__(self, in_channels, out_channels): 
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)

    def forward(self, x): 
        return self.conv(self.upsample(x))

class MultiheadedAttention(nn.Module):
    def __init__(self, total_embedding_dim, num_heads):
        super().__init__()

        self.num_heads = num_heads
        self.total_embedding_dim = total_embedding_dim 

        self.mha = nn.MultiheadAttention(total_embedding_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(total_embedding_dim)

    
    def forward(self, x): 
        """
        Forward pass: 
        x gets reshaped from [B, C, H, W] to [B, HW, C] where C acts as the embedding dimension. Thus, C must = total_embedding_dim
        x gets passed through the multihead attention layer as q, k and v.

        The output of the multihead attention layer is [B, HW, C] where C = num_heads * head_dim
        The output is then reshaped back to [B, C, H, W]
        """

        assert x.shape[1] == self.total_embedding_dim, "Total embedding dimension must be equal to the input embedding dimension"

        h = x.flatten(2).transpose(1, 2).contiguous()
        h = self.norm(h)
        h, _  = self.mha(h, h, h)
        h = h.transpose(1, 2).contiguous().view(x.shape)

        return h

class UNet(nn.Module):
    def __init__(
        self,
        *,
        in_channels,
        out_channels,
        embedding_dim,
        base_channels=64,
        channel_mults=(1, 2, 4),
        channels=None,
        num_res_blocks_enc=2,
        num_res_blocks_dec=None,
        downsample_at=None,
        attn_stages=None,
        dec_attn_stages=None,
        bottleneck_attn=True,
        attn_num_heads=2,
        dec_attn_num_heads=None,
        bottleneck_num_heads=4,
        attn_block_index=1,
        num_groups=32,
        device='cpu',
    ):
        """
        A configurable U-Net architecture with per-stage control over channels, number of residual blocks,
        down/upsampling locations, and attention placement.

        Args:
            in_channels: number of input channels
            out_channels: number of output channels
            embedding_dim: conditioning embedding dimension
            base_channels: base channel count (used when channels is None)
            channel_mults: multipliers applied to base_channels when channels is None
            channels: optional explicit list of channels per encoder/decoder stage
            num_res_blocks_enc: int or list[int] of residual blocks per encoder stage
            num_res_blocks_dec: int or list[int] of residual blocks per decoder stage (defaults to encoder)
            downsample_at: list[bool] of length (num_stages-1) indicating whether to downsample between stages
            attn_stages: list[bool] of length num_stages indicating encoder stages with attention
            dec_attn_stages: list[bool] for decoder stages (defaults to attn_stages)
            bottleneck_attn: whether to apply attention in the bottleneck
            attn_num_heads: int or list[int] number of heads per encoder stage attention
            dec_attn_num_heads: int or list[int] number of heads per decoder stage attention (defaults to attn_num_heads)
            bottleneck_num_heads: number of attention heads at the bottleneck
            attn_block_index: index (1-based) after which attention is applied within a stage
            num_groups: number of groups used for GroupNorm in residual blocks
            device: torch device
        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.embedding_dim = embedding_dim
        self.base_channels = base_channels
        self.channel_mults = channel_mults
        self.channels = channels
        self.num_res_blocks_enc = num_res_blocks_enc
        self.num_res_blocks_dec = num_res_blocks_dec
        self.downsample_at = downsample_at
        self.attn_stages = attn_stages
        self.dec_attn_stages = dec_attn_stages
        self.bottleneck_attn = bottleneck_attn
        self.attn_num_heads = attn_num_heads
        self.dec_attn_num_heads = dec_attn_num_heads
        self.bottleneck_num_heads = bottleneck_num_heads
        self.attn_block_index = attn_block_index
        self.num_groups = num_groups
        self.device = device
        
        # Derive channels per stage
        if channels is None:
            stage_channels = [base_channels * m for m in channel_mults]
        else:
            stage_channels = list(channels)
        assert len(stage_channels) >= 1, "At least one stage is required"
        num_stages = len(stage_channels)

        # Downsample flags between stages (length = num_stages - 1)
        if downsample_at is None:
            if num_stages > 1:
                downsample_at = [True] * (num_stages - 1)
            else:
                downsample_at = []
        assert len(downsample_at) == max(0, num_stages - 1), "downsample_at must have length num_stages-1"

        # Residual blocks per stage (encoder and decoder)
        if isinstance(num_res_blocks_enc, int):
            num_res_blocks_enc = [num_res_blocks_enc] * num_stages
        else:
            num_res_blocks_enc = list(num_res_blocks_enc)
            assert len(num_res_blocks_enc) == num_stages, "num_res_blocks_enc must have length num_stages"

        if num_res_blocks_dec is None:
            num_res_blocks_dec = list(num_res_blocks_enc)
        elif isinstance(num_res_blocks_dec, int):
            num_res_blocks_dec = [num_res_blocks_dec] * num_stages
        else:
            num_res_blocks_dec = list(num_res_blocks_dec)
            assert len(num_res_blocks_dec) == num_stages, "num_res_blocks_dec must have length num_stages"

        # Attention configuration
        if attn_stages is None:
            # Default: attention at the middle stage if there are >= 3 stages; else no attention
            if num_stages >= 3:
                mid = num_stages // 2
                attn_stages = [i == mid for i in range(num_stages)]
            else:
                attn_stages = [False] * num_stages
        else:
            attn_stages = list(attn_stages)
            assert len(attn_stages) == num_stages, "attn_stages must have length num_stages"

        if dec_attn_stages is None:
            dec_attn_stages = list(attn_stages)
        else:
            dec_attn_stages = list(dec_attn_stages)
            assert len(dec_attn_stages) == num_stages, "dec_attn_stages must have length num_stages"

        def _expand_heads(heads, default):
            if isinstance(heads, int):
                return [heads] * num_stages
            if heads is None:
                return [default] * num_stages
            heads = list(heads)
            assert len(heads) == num_stages, "heads list must have length num_stages"
            return heads

        enc_heads = _expand_heads(attn_num_heads, 2)
        dec_heads = _expand_heads(dec_attn_num_heads, enc_heads[0] if len(enc_heads) > 0 else 2)

        # Store for forward
        self.stage_channels = stage_channels
        self.enc_heads = enc_heads
        self.dec_heads = dec_heads
        

        # Build encoder blocks
        self.enc_blocks = nn.ModuleList()
        self.enc_attn_modules = nn.ModuleList()
        self.enc_transitions = nn.ModuleList()

        for s in range(num_stages):
            blocks = nn.ModuleList()
            # First block of stage s receives the output of previous transition
            first_in = in_channels if s == 0 else stage_channels[s]
            for i in range(num_res_blocks_enc[s]):
                out_ch = stage_channels[s]
                in_ch = first_in if i == 0 else stage_channels[s]
                blocks.append(ResidualBlock(in_channels=in_ch, out_channels=out_ch, embedding_dim=embedding_dim, num_groups=num_groups))
            self.enc_blocks.append(blocks)

            # Stage attention (applied after block index "attn_block_index")
            if attn_stages[s]:
                self.enc_attn_modules.append(MultiheadedAttention(total_embedding_dim=stage_channels[s], num_heads=enc_heads[s]))
            else:
                self.enc_attn_modules.append(nn.Identity())

            # Transition to next stage
            if s < num_stages - 1:
                if downsample_at[s]:
                    self.enc_transitions.append(DownSample(in_channels=stage_channels[s], out_channels=stage_channels[s + 1]))
                else:
                    if stage_channels[s] == stage_channels[s + 1]:
                        self.enc_transitions.append(nn.Identity())
                    else:
                        self.enc_transitions.append(nn.Conv2d(stage_channels[s], stage_channels[s + 1], 1))
        
        

        # Bottleneck
        bottleneck_channels = stage_channels[-1]
        self.b1 = ResidualBlock(in_channels=bottleneck_channels, out_channels=bottleneck_channels, embedding_dim=embedding_dim, num_groups=num_groups)
        self.b_attn = MultiheadedAttention(total_embedding_dim=bottleneck_channels, num_heads=bottleneck_num_heads) if bottleneck_attn else nn.Identity()
        self.b2 = ResidualBlock(in_channels=bottleneck_channels, out_channels=bottleneck_channels, embedding_dim=embedding_dim, num_groups=num_groups)

        # Decoder blocks
        self.dec_blocks = nn.ModuleList([None] * num_stages)
        self.dec_attn_modules = nn.ModuleList([None] * num_stages)
        # Upsampling or channel adjust modules from stage s+1 -> s
        self.up_modules = nn.ModuleList()

        for s in range(num_stages - 1):
            if downsample_at[s]:
                self.up_modules.append(Upsample(in_channels=stage_channels[s + 1], out_channels=stage_channels[s]))
            else:
                if stage_channels[s + 1] == stage_channels[s]:
                    self.up_modules.append(nn.Identity())
                else:
                    self.up_modules.append(nn.Conv2d(stage_channels[s + 1], stage_channels[s], 1))


        # Build per-stage decoder residual blocks (processed from top stage downwards in forward)
        for s in range(num_stages):
            blocks = nn.ModuleList()
            # First decoder block takes concatenated features: skip (C_s) + incoming (C_s)
            in_ch = stage_channels[s] * 2
            out_ch = stage_channels[s]
            blocks.append(ResidualBlock(in_channels=in_ch, out_channels=out_ch, embedding_dim=embedding_dim, num_groups=num_groups))
            for _ in range(num_res_blocks_dec[s] - 1):
                blocks.append(ResidualBlock(in_channels=out_ch, out_channels=out_ch, embedding_dim=embedding_dim, num_groups=num_groups))
            self.dec_blocks[s] = blocks

            if dec_attn_stages[s]:
                self.dec_attn_modules[s] = MultiheadedAttention(total_embedding_dim=stage_channels[s], num_heads=dec_heads[s])
            else:
                self.dec_attn_modules[s] = nn.Identity()
            
        # Head
        self.head = nn.Sequential(
            nn.GroupNorm(min(num_groups, stage_channels[0]), stage_channels[0]),
            nn.SiLU(),
            nn.Conv2d(stage_channels[0], out_channels, 3, padding=1)
        )

        self.to(device)

    def forward(self, x, cond):
        """
        Args:
            x: [B, C, H, W]
            cond: [B, embedding_dim]

        Returns:
            [B, C, H, W]
        """
        stage_channels = self.stage_channels
        num_stages = len(stage_channels)

        # Encoder
        skips = []
        h = x
        # print(h.shape)
        for s in range(num_stages):
            blocks = self.enc_blocks[s]
            for i, block in enumerate(blocks, start=1):
                h = block(h, cond)
                # print(f"Encoder stage {s} block {i} output shape: {h.shape}")
                if isinstance(self.enc_attn_modules[s], nn.Identity):
                    pass
                else:
                    if i == self.attn_block_index:
                        h = self.enc_attn_modules[s](h)
                        # print(f"Encoder stage {s} attention output shape: {h.shape}")
            skips.append(h)
            if s < num_stages - 1:
                h = self.enc_transitions[s](h)
                # print(f"Encoder stage {s} transition output shape: {h.shape}")

        # Bottleneck
        h = self.b1(h, cond)
        h = self.b_attn(h)
        h = self.b2(h, cond)
        # print(f"Bottleneck output shape: {h.shape}")

        # Decoder (top stage down to 0)
        for s in reversed(range(num_stages)):
            if s < num_stages - 1:
                h = self.up_modules[s](h)
                # print(f"Decoder stage {s} upsampling output shape: {h.shape}")
            h = torch.cat([h, skips[s]], dim=1)
            blocks = self.dec_blocks[s]
            for i, block in enumerate(blocks, start=1):
                h = block(h, cond)
                # print(f"Decoder stage {s} block {i} output shape: {h.shape}")
                if isinstance(self.dec_attn_modules[s], nn.Identity):
                    pass
                else:
                    if i == self.attn_block_index:
                        h = self.dec_attn_modules[s](h)
                        # print(f"Decoder stage {s} attention output shape: {h.shape}")

        h = self.head(h)
        # print(f"Head output shape: {h.shape}")
        return h