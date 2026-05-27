from torch import nn
import torch
import math

from .layers import *
from .utils import *


class UniversalTransformerPredictor(nn.Module):
    def __init__(
        self,
        num_patches,
        act=False,
        time_emb=False,
        epsilon=None,
        max_steps=4,
        embed_dim=768,
        predictor_embed_dim=384,
        depth=6,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        init_std=0.02,
        **kwargs
    ):
        super().__init__()

        self.act = act
        self.time_emb = time_emb

        self.num_steps = max_steps if act else depth

        if act:
            self.epsilon = epsilon
            self.halting_linear = nn.Linear(predictor_embed_dim, 1)

        self.predictor_embed = nn.Linear(embed_dim, predictor_embed_dim, bias=True) # перескакиваем в пространство меньшей размерности predictor_embed_dim
        self.mask_token = nn.Parameter(torch.zeros(1, 1, predictor_embed_dim)) # [1, 1, E], токены масок которые будем предсказывать
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # Чем глубже - тем чаще дропаутим слои, см class DropPath
        # --
        self.predictor_pos_embed = nn.Parameter(torch.zeros(1, num_patches, predictor_embed_dim),
                                                requires_grad=False)
        predictor_pos_embed = get_2d_sincos_pos_embed(self.predictor_pos_embed.shape[-1],
                                                      int(num_patches**.5),
                                                      cls_token=False)
        self.predictor_pos_embed.data.copy_(torch.from_numpy(predictor_pos_embed).float().unsqueeze(0)) # Кринжово создали пос эмбеддинг
        # --
        
        self.block = Block(predictor_embed_dim, num_heads, qkv_bias=True)
        self.predictor_norm = norm_layer(predictor_embed_dim) # нормализация на чилле
        self.predictor_proj = nn.Linear(predictor_embed_dim, embed_dim, bias=True) # проекция на чилле обратно в пространство большей размерности
        # ------

        self.step_embeddings = nn.Parameter(torch.zeros(self.num_steps, 1, predictor_embed_dim))
        trunc_normal_(self.step_embeddings, std=0.02)

        self.init_std = init_std # для инициализации весов
        trunc_normal_(self.mask_token, std=self.init_std)
        self.apply(self._init_weights)
        self.fix_init_weight()

    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.predictor_blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x, masks_x, masks):
        assert (masks is not None) and (masks_x is not None), 'Cannot run predictor without mask indices'

        if not isinstance(masks_x, list):
            masks_x = [masks_x]

        if not isinstance(masks, list):
            masks = [masks]

        # -- Batch Size
        B = len(x) // len(masks_x)

        # -- map from encoder-dim to pedictor-dim
        x = self.predictor_embed(x)

        # -- add positional embedding to x tokens
        x_pos_embed = self.predictor_pos_embed.repeat(B, 1, 1)
        x += apply_masks(x_pos_embed, masks_x)

        _, N_ctxt, D = x.shape

        # -- concat mask tokens to x
        pos_embs = self.predictor_pos_embed.repeat(B, 1, 1)
        pos_embs = apply_masks(pos_embs, masks)
        pos_embs = repeat_interleave_batch(pos_embs, B, repeat=len(masks_x))
        # --
        pred_tokens = self.mask_token.repeat(pos_embs.size(0), pos_embs.size(1), 1)
        # --
        pred_tokens += pos_embs
        x = x.repeat(len(masks), 1, 1)
        x = torch.cat([x, pred_tokens], dim=1)

        if self.act:
            B, N, _ = x.shape
            halting_prob = torch.zeros(B, N, 1, device=x.device)
            remainders = torch.zeros(B, N, 1, device=x.device)
            n_updates = torch.zeros(B, N, 1, device=x.device)
            output = torch.zeros_like(x)

        for t in range(self.num_steps):
            if self.time_emb:
                x = x + self.step_embeddings[t]
            
            x_new = self.block(x)
            
            if self.act:
                # some act logic
                h = torch.sigmoid(self.halting_linear(x_new))
                still_running = (halting_prob < (1 - self.epsilon)).float()
                
                new_halted = (halting_prob + h * still_running > 1 - self.epsilon).float() * still_running
                still_running_after = (halting_prob + h * still_running <= 1 - self.epsilon).float() * still_running

                remainders += new_halted * (1 - halting_prob)
                halting_prob += h * still_running_after
                halting_prob += remainders * new_halted

                n_updates += still_running

                update_weights = h * still_running_after + remainders * new_halted
                output = output + update_weights * x_new
                
                # updating x for active tokens
                x = x_new * still_running + x * (1 - still_running)
            else:
                x = x_new

        if self.act:
            act_loss = (n_updates + remainders).mean()
            output = self.layernorm(output)
            output = output[:, N_ctxt:]
            output = self.enc_proj(output)

            return {
                "predictions": output, 
                "act_loss": act_loss
            }


        x = self.predictor_norm(x)

        # -- return preds for mask tokens
        x = x[:, N_ctxt:]
        x = self.predictor_proj(x)

        return {
            "predictions": x
        }
