from torch import nn
import torch

from layers import Block, get_2d_sincos_pos_embed
from utils import trunc_normal_

class UniversalTransformerBlock(nn.Module):
    def __init__(self, num_patches, predictor_dim, encoder_dim, num_heads, p_dropout=0.0, num_steps=4, time_emb=False, act=False, max_steps=None, epsilon=None):
        """
        Args:
            predictor_dim (int): size of predictor's hidden dimension
            encoder_dim (int): size of encoder's hidden dimension
            num_heads (int): number of heads to use in Attention
            p_dropout (float): probability of an element to be zeroed
            num_steps (int): number of times to run the Transformer block
            time_emb (bool): whether to use time embedding
            act (bool): whether to use ACT 
            max_steps (int): maximum possible steps a token goes through Transformer block
            epilon (float): ACT epsilon
        """
        super().__init__()
        self.mask_token = nn.Parameter(torch.zeros(1, 1, predictor_dim))

        self.act = act
        self.time_emb = time_emb

        self.num_steps = max_steps if act else num_steps

        if act:
            self.epsilon = epsilon
            self.halting_linear = nn.Linear(predictor_dim, 1)

        self.layernorm = nn.LayerNorm(predictor_dim)
        self.dropout = nn.Dropout(p_dropout)
        self.block = Block(predictor_dim, num_heads, qkv_bias=True)

        self.predictor_pos_embed = nn.Parameter(torch.zeros(1, num_patches, predictor_dim),
                                                requires_grad=False)
        predictor_pos_embed = get_2d_sincos_pos_embed(self.predictor_pos_embed.shape[-1],
                                                      int(num_patches**.5),
                                                      cls_token=False)
        self.predictor_pos_embed.data.copy_(torch.from_numpy(predictor_pos_embed).float().unsqueeze(0))

        self.step_embeddings = nn.Parameter(torch.zeros(self.num_steps, 1, predictor_dim))
        trunc_normal_(self.step_embeddings, std=0.02)

        self.pred_proj = nn.Linear(in_features=encoder_dim, out_features=predictor_dim)
        self.enc_proj = nn.Linear(in_features=predictor_dim, out_features=encoder_dim)

    def forward(self, x, masks_x, masks):
        """
        Args:
            x (Tensor): context embeddings of the images' patches of shape [B, num_patches, enc_dim]
            masks_x (Tensor): patch indices of the context (images' visible parts) of shape [N_ctxt]
            masks (Tensor): patch indices of targets of shape [N_trgt]
        Returns a tensor of predicted target embeddings
        """
        B = len(x) // len(masks_x)
        # Prepating context embeddings
        x = self.pred_proj(x)
        x_pos_embed = self.predictor_pos_embed.repeat(B, 1, 1)
        x += self.apply_masks(x_pos_embed, masks_x)

        N_ctxt = x.size(1)

        # Preparing pos embeddings for targets
        pos_embs = self.predictor_pos_embed.repeat(B, 1, 1)
        pos_embs = self.apply_masks(pos_embs, masks)
        pos_embs = self.repeat_interleave_batch(pos_embs, B, repeat=len(masks_x))

        # Preparing tokens for target (context + pos)
        pred_tokens = self.mask_token.repeat(pos_embs.size(0), pos_embs.size(1), 1)
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

            return output, act_loss

        x = self.layernorm(x)
        x = x[:, N_ctxt:]
        x = self.enc_proj(x)

        return x
    
    def apply_masks(self, x, masks):
        """
        Args:
            x (Tensor): data of shape [B (batch-size), N (num-patches), D (feature-dim)]
            masks (list): list containing indices of patches to keep
        """
        all_x = []
        for m in masks:
            mask_keep = m.unsqueeze(-1).repeat(1, 1, x.size(-1))
            all_x += [torch.gather(x, dim=1, index=mask_keep)]
        return torch.cat(all_x, dim=0)
    
    def repeat_interleave_batch(self, x, B, repeat):
        N = len(x) // B
        x = torch.cat([
            torch.cat([x[i*B:(i+1)*B] for _ in range(repeat)], dim=0)
            for i in range(N)
        ], dim=0)
        return x