from typing import Mapping, Optional, Tuple

import numpy as np
import scipy
import torch
import torch.nn as nn
from torch import Tensor

from multigame_dt_utils import (
    accuracy,
    autoregressive_generate,
    cross_entropy,
    decode_return,
    encode_return,
    encode_reward,
    sample_from_logits,
    variance_scaling_,
)


class MLP(nn.Module):
    r"""A 2-layer MLP which widens then narrows the input."""

    def __init__(
        self,
        in_dim: int,
        init_scale: float,
        widening_factor: int = 4,
    ):
        super().__init__()
        self._init_scale = init_scale
        self._widening_factor = widening_factor

        self.fc1 = nn.Linear(in_dim, self._widening_factor * in_dim)
        self.act = nn.GELU(approximate="tanh")
        self.fc2 = nn.Linear(self._widening_factor * in_dim, in_dim)

        self.reset_parameters()

    def reset_parameters(self):
        variance_scaling_(self.fc1.weight, scale=self._init_scale)
        nn.init.zeros_(self.fc1.bias)
        variance_scaling_(self.fc2.weight, scale=self._init_scale)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        w_init_scale: Optional[float] = None,
        qkv_bias: bool = True,
        proj_bias: bool = True,
    ):
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5
        self.w_init_scale = w_init_scale

        self.qkv = nn.Linear(dim, 3 * dim, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)

        self.reset_parameters()

    def reset_parameters(self):
        variance_scaling_(self.qkv.weight, scale=self.w_init_scale)
        if self.qkv.bias is not None:
            nn.init.zeros_(self.qkv.bias)
        variance_scaling_(self.proj.weight, scale=self.w_init_scale)
        if self.proj.bias is not None:
            nn.init.zeros_(self.proj.bias)

    def forward(self, x, mask: Optional[Tensor] = None) -> Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        if mask is not None:
            mask_value = -torch.finfo(attn.dtype).max  # max_neg_value
            attn = attn.masked_fill(~mask.to(dtype=torch.bool), mask_value)

        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, T, C)
        x = self.proj(x)
        return x


class CausalSelfAttention(Attention):
    r"""Self attention with a causal mask applied."""

    def forward(
        self,
        x: Tensor,
        mask: Optional[Tensor] = None,
        custom_causal_mask: Optional[Tensor] = None,
        prefix_length: Optional[int] = 0,
    ) -> Tensor:
        if x.ndim != 3:
            raise ValueError("Expect queries of shape [B, T, D].")

        seq_len = x.shape[1]
        # If custom_causal_mask is None, the default causality assumption is
        # sequential (a lower triangular causal mask).
        causal_mask = custom_causal_mask
        if causal_mask is None:
            device = x.device
            causal_mask = torch.tril(torch.ones((seq_len, seq_len), dtype=torch.bool, device=device))
        causal_mask = causal_mask[None, None, :, :]

        # Similar to T5, tokens up to prefix_length can all attend to each other.
        causal_mask[:, :, :, :prefix_length] = 1
        mask = mask * causal_mask if mask is not None else causal_mask

        return super().forward(x, mask)


class Block(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, init_scale: float, dropout_rate: float):
        super().__init__()
        self.ln_1 = nn.LayerNorm(embed_dim)
        self.attn = CausalSelfAttention(embed_dim, num_heads=num_heads, w_init_scale=init_scale)
        self.dropout_1 = nn.Dropout(dropout_rate)

        self.ln_2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, init_scale)
        self.dropout_2 = nn.Dropout(dropout_rate)

    def forward(self, x, **kwargs):
        x = x + self.dropout_1(self.attn(self.ln_1(x), **kwargs))
        x = x + self.dropout_2(self.mlp(self.ln_2(x)))
        return x


class Transformer(nn.Module):
    r"""A transformer stack."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_layers: int,
        dropout_rate: float,
    ):
        super().__init__()
        self._num_layers = num_layers
        self._num_heads = num_heads
        self._dropout_rate = dropout_rate

        init_scale = 2.0 / self._num_layers
        self.layers = nn.ModuleList([])
        for _ in range(self._num_layers):
            block = Block(embed_dim, num_heads, init_scale, dropout_rate)
            self.layers.append(block)
        self.norm_f = nn.LayerNorm(embed_dim)

    def forward(
        self,
        h: Tensor,
        mask: Optional[Tensor] = None,
        custom_causal_mask: Optional[Tensor] = None,
        prefix_length: Optional[int] = 0,
    ) -> Tensor:
        r"""Connects the transformer.

        Args:
        h: Inputs, [B, T, D].
        mask: Padding mask, [B, T].
        custom_causal_mask: Customized causal mask, [T, T].
        prefix_length: Number of prefix tokens that can all attend to each other.

        Returns:
        Array of shape [B, T, D].
        """
        if mask is not None:
            # Make sure we're not passing any information about masked h.
            h = h * mask[:, :, None]
            mask = mask[:, None, None, :]

        for block in self.layers:
            h = block(
                h,
                mask=mask,
                custom_causal_mask=custom_causal_mask,
                prefix_length=prefix_length,
            )
        h = self.norm_f(h)
        return h


class MultiGameDecisionTransformer(nn.Module):
    def __init__(
        self,
        num_actions: int,
        num_rewards: int,
        return_range: Tuple[int],
        d_model: int,
        num_layers: int,
        dropout_rate: float,
        predict_reward: bool,
        single_return_token: bool,
        conv_dim: int,
    ):
        super().__init__()

        # Expected by the transformer model.
        if d_model % 64 != 0:
            raise ValueError(f"Model size {d_model} must be divisible by 64")

        self.num_actions = num_actions
        self.num_rewards = num_rewards
        self.num_returns = return_range[1] - return_range[0]
        self.return_range = return_range
        self.d_model = d_model
        self.predict_reward = predict_reward
        self.conv_dim = conv_dim
        self.single_return_token = single_return_token
        self.spatial_tokens = True

        self.transformer = Transformer(
            embed_dim=self.d_model,
            num_heads=self.d_model // 64,
            num_layers=num_layers,
            dropout_rate=dropout_rate,
        )

        img_size = (84, 84)
        patch_size = (14, 14)
        patch_height, patch_width = patch_size[0], patch_size[1]
        # If patch_size is (14, 14) for example, P = 84 / 14 = 6
        self.image_emb = nn.Conv2d(
            in_channels=1,
            out_channels=self.d_model,
            kernel_size=(patch_height, patch_width),
            stride=(patch_height, patch_width),
            padding="valid",
        )  # image_emb is now [BT x D x P x P].
        patch_grid = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        num_patches = patch_grid[0] * patch_grid[1]
        self.image_pos_enc = nn.Parameter(torch.randn(1, 1, num_patches, self.d_model))

        self.ret_emb = nn.Embedding(self.num_returns, self.d_model)
        self.act_emb = nn.Embedding(self.num_actions, self.d_model)
        if self.predict_reward:
            self.rew_emb = nn.Embedding(self.num_rewards, self.d_model)

        num_steps = 4
        num_obs_tokens = num_patches if self.spatial_tokens else 1
        if self.predict_reward:
            tokens_per_step = num_obs_tokens + 3
        else:
            tokens_per_step = num_obs_tokens + 2
        self.positional_embedding = nn.Parameter(torch.randn(tokens_per_step * num_steps, self.d_model))

        self.ret_linear = nn.Linear(self.d_model, self.num_returns)
        self.act_linear = nn.Linear(self.d_model, self.num_actions)
        if self.predict_reward:
            self.rew_linear = nn.Linear(self.d_model, self.num_rewards)

    def reset_parameters(self):
        nn.init.trunc_normal_(self.image_emb.weight, std=0.02)
        nn.init.zeros_(self.image_emb.bias)
        nn.init.normal_(self.image_pos_enc, std=0.02)

        nn.init.trunc_normal_(self.ret_emb.weight, std=0.02)
        nn.init.trunc_normal_(self.act_emb.weight, std=0.02)
        if self.predict_reward:
            nn.init.trunc_normal_(self.rew_emb.weight, std=0.02)

        nn.init.trunc_normal_(self.positional_embedding, std=0.02)

        variance_scaling_(self.ret_linear.weight)
        nn.init.zeros_(self.ret_linear.bias)
        variance_scaling_(self.act_linear.weight)
        nn.init.zeros_(self.act_linear.bias)
        if self.predict_reward:
            variance_scaling_(self.rew_linear.weight)
            nn.init.zeros_(self.rew_linear.bias)

    def _image_embedding(self, image: Tensor):
        r"""Embed [B x T x C x W x H] images to tokens [B x T x output_dim] tokens.

        Args:
            image: [B x T x C x W x H] image to embed.

        Returns:
            Image embedding of shape [B x T x output_dim] or [B x T x _ x output_dim].
        """
        assert len(image.shape) == 5

        image_dims = image.shape[-3:]
        batch_dims = image.shape[:2]

        # Reshape to [BT x C x H x W].
        image = torch.reshape(image, (-1,) + image_dims)
        # Perform any-image specific processing.
        image = image.to(dtype=torch.float32) / 255.0

        image_emb = self.image_emb(image)  # [BT x D x P x P]
        # haiku.Conv2D is channel-last, so permute before reshape below for consistency
        image_emb = image_emb.permute(0, 2, 3, 1)  # [BT x P x P x D]

        # Reshape to [B x T x P*P x D].
        image_emb = torch.reshape(image_emb, batch_dims + (-1, self.d_model))
        image_emb = image_emb + self.image_pos_enc
        return image_emb

    def _embed_inputs(self, obs: Tensor, ret: Tensor, act: Tensor, rew: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        # Embed only prefix_frames first observations.
        # obs are [B x T x C x H x W].
        obs_emb = self._image_embedding(obs)
        # Embed returns and actions
        # Encode returns.
        ret = encode_return(ret, self.return_range)
        rew = encode_reward(rew)
        ret_emb = self.ret_emb(ret)
        act_emb = self.act_emb(act)
        if self.predict_reward:
            rew_emb = self.rew_emb(rew)
        else:
            rew_emb = None
        return obs_emb, ret_emb, act_emb, rew_emb

    def forward(self, inputs: Mapping[str, Tensor]) -> Mapping[str, Tensor]:
        r"""Process sequence."""
        num_batch = inputs["actions"].shape[0]
        num_steps = inputs["actions"].shape[1]
        # Embed inputs.
        obs_emb, ret_emb, act_emb, rew_emb = self._embed_inputs(
            inputs["observations"],
            inputs["returns-to-go"],
            inputs["actions"],
            inputs["rewards"],
        )
        device = obs_emb.device

        if self.spatial_tokens:
            # obs is [B x T x W x D]
            num_obs_tokens = obs_emb.shape[2]
            obs_emb = torch.reshape(obs_emb, obs_emb.shape[:2] + (-1,))
            # obs is [B x T x W*D]
        else:
            num_obs_tokens = 1
        # Collect sequence.
        # Embeddings are [B x T x D].
        if self.predict_reward:
            token_emb = torch.cat([obs_emb, ret_emb, act_emb, rew_emb], dim=-1)
            tokens_per_step = num_obs_tokens + 3
            # sequence is [obs ret act rew ... obs ret act rew]
        else:
            token_emb = torch.cat([obs_emb, ret_emb, act_emb], dim=-1)
            tokens_per_step = num_obs_tokens + 2
            # sequence is [obs ret act ... obs ret act]
        token_emb = torch.reshape(token_emb, [num_batch, tokens_per_step * num_steps, self.d_model])
        # Create position embeddings.
        token_emb = token_emb + self.positional_embedding
        # Run the transformer over the inputs.

        # Token dropout.
        batch_size = token_emb.shape[0]
        obs_mask = np.ones([batch_size, num_steps, num_obs_tokens], dtype=bool)
        ret_mask = np.ones([batch_size, num_steps, 1], dtype=bool)
        act_mask = np.ones([batch_size, num_steps, 1], dtype=bool)
        rew_mask = np.ones([batch_size, num_steps, 1], dtype=bool)
        if self.single_return_token:
            # Mask out all return tokens expect the first one.
            ret_mask[:, 1:] = 0
        if self.predict_reward:
            mask = [obs_mask, ret_mask, act_mask, rew_mask]
        else:
            mask = [obs_mask, ret_mask, act_mask]
        mask = np.concatenate(mask, axis=-1)
        mask = np.reshape(mask, [batch_size, tokens_per_step * num_steps])
        mask = torch.tensor(mask, dtype=torch.bool, device=device)

        custom_causal_mask = None
        if self.spatial_tokens:
            # Temporal transformer by default assumes sequential causal relation.
            # This makes the transformer causal mask a lower triangular matrix.
            #     P1 P2 R  a  P1 P2 ... (Ps: image patches)
            # P1  1  0* 0  0  0  0
            # P2  1  1  0  0  0  0
            # R   1  1  1  0  0  0
            # a   1  1  1  1  0  0
            # P1  1  1  1  1  1  0*
            # P2  1  1  1  1  1  1
            # ... (0*s should be replaced with 1s in the ideal case)
            # But, when we have multiple tokens for an image (e.g. patch tokens, conv
            # feature map tokens, etc) as inputs to transformer, this assumption does
            # not hold, because there is no sequential dependencies between tokens.
            # Therefore, the ideal causal mask should not mask out tokens that belong
            # to the same images from each others.
            seq_len = token_emb.shape[1]
            sequential_causal_mask = np.tril(np.ones((seq_len, seq_len)))
            num_timesteps = seq_len // tokens_per_step
            num_non_obs_tokens = tokens_per_step - num_obs_tokens
            diag = [
                np.ones((num_obs_tokens, num_obs_tokens)) if i % 2 == 0 else np.zeros((num_non_obs_tokens, num_non_obs_tokens))
                for i in range(num_timesteps * 2)
            ]
            block_diag = scipy.linalg.block_diag(*diag)
            custom_causal_mask = np.logical_or(sequential_causal_mask, block_diag)
            custom_causal_mask = torch.tensor(custom_causal_mask, dtype=torch.bool, device=device)

        output_emb = self.transformer(token_emb, mask, custom_causal_mask)

        # Output_embeddings are [B x 3T x D].
        # Next token predictions (tokens one before their actual place).
        ret_pred = output_emb[:, (num_obs_tokens - 1) :: tokens_per_step, :]
        act_pred = output_emb[:, (num_obs_tokens - 0) :: tokens_per_step, :]
        embeds = torch.cat([ret_pred, act_pred], dim=-1)
        # Project to appropriate dimensionality.
        ret_pred = self.ret_linear(ret_pred)
        act_pred = self.act_linear(act_pred)
        # Return logits as well as pre-logits embedding.
        result_dict = {
            "embeds": embeds,
            "action_logits": act_pred,
            "return_logits": ret_pred,
        }
        if self.predict_reward:
            rew_pred = output_emb[:, (num_obs_tokens + 1) :: tokens_per_step, :]
            rew_pred = self.rew_linear(rew_pred)
            result_dict["reward_logits"] = rew_pred
        # Return evaluation metrics.
        result_dict["loss"] = self.sequence_loss(inputs, result_dict)
        result_dict["accuracy"] = self.sequence_accuracy(inputs, result_dict)
        return result_dict

    def _objective_pairs(self, inputs: Mapping[str, Tensor], model_outputs: Mapping[str, Tensor]) -> Tensor:
        r"""Get logit-target pairs for the model objective terms."""
        act_target = inputs["actions"]
        ret_target = encode_return(inputs["returns-to-go"], self.return_range)
        act_logits = model_outputs["action_logits"]
        ret_logits = model_outputs["return_logits"]
        if self.single_return_token:
            ret_target = ret_target[:, :1]
            ret_logits = ret_logits[:, :1, :]
        obj_pairs = [(act_logits, act_target), (ret_logits, ret_target)]
        if self.predict_reward:
            rew_target = encode_reward(inputs["rewards"])
            rew_logits = model_outputs["reward_logits"]
            obj_pairs.append((rew_logits, rew_target))
        return obj_pairs

    def sequence_loss(self, inputs: Mapping[str, Tensor], model_outputs: Mapping[str, Tensor]) -> Tensor:
        r"""Compute the loss on data wrt model outputs."""
        obj_pairs = self._objective_pairs(inputs, model_outputs)
        obj = [cross_entropy(logits, target) for logits, target in obj_pairs]
        return sum(obj) / len(obj)

    def sequence_accuracy(self, inputs: Mapping[str, Tensor], model_outputs: Mapping[str, Tensor]) -> Tensor:
        r"""Compute the accuracy on data wrt model outputs."""
        obj_pairs = self._objective_pairs(inputs, model_outputs)
        obj = [accuracy(logits, target) for logits, target in obj_pairs]
        return sum(obj) / len(obj)

    def optimal_action(
        self,
        inputs: Mapping[str, Tensor],
        return_range: Tuple[int] = (-100, 100),
        single_return_token: bool = False,
        opt_weight: Optional[float] = 0.0,
        num_samples: Optional[int] = 128,
        action_temperature: Optional[float] = 1.0,
        return_temperature: Optional[float] = 1.0,
        action_top_percentile: Optional[float] = None,
        return_top_percentile: Optional[float] = None,
        rng: Optional[torch.Generator] = None,
        deterministic: bool = False,
    ):
        r"""Calculate optimal action for the given sequence model."""
        logits_fn = self.forward
        obs, act, rew = inputs["observations"], inputs["actions"], inputs["rewards"]
        assert len(obs.shape) == 5
        assert len(act.shape) == 2
        inputs = {
            "observations": obs,
            "actions": act,
            "rewards": rew,
            "returns-to-go": torch.zeros_like(act),
        }
        sequence_length = obs.shape[1]
        # Use samples from the last timestep.
        timestep = -1
        # A biased sampling function that prefers sampling larger returns.
        def ret_sample_fn(rng, logits):
            assert len(logits.shape) == 2
            # Add optimality bias.
            if opt_weight > 0.0:
                # Calculate log of P(optimality=1|return) := exp(return) / Z.
                logits_opt = torch.linspace(0.0, 1.0, logits.shape[1])
                logits_opt = torch.repeat_interleave(logits_opt[None, :], logits.shape[0], dim=0)
                # Sample from log[P(optimality=1|return)*P(return)].
                logits = logits + opt_weight * logits_opt
            logits = torch.repeat_interleave(logits[None, ...], num_samples, dim=0)
            ret_sample = sample_from_logits(
                logits,
                generator=rng,
                deterministic=deterministic,
                temperature=return_temperature,
                top_percentile=return_top_percentile,
            )
            # Pick the highest return sample.
            ret_sample, _ = torch.max(ret_sample, dim=0)
            # Convert return tokens into return values.
            ret_sample = decode_return(ret_sample, return_range)
            return ret_sample

        # Set returns-to-go with an (optimistic) autoregressive sample.
        if single_return_token:
            # Since only first return is used by the model, only sample that (faster).
            ret_logits = logits_fn(inputs)["return_logits"][:, 0, :]
            ret_sample = ret_sample_fn(rng, ret_logits)
            inputs["returns-to-go"][:, 0] = ret_sample
        else:
            # Auto-regressively regenerate all return tokens in a sequence.
            ret_logits_fn = lambda input: logits_fn(input)["return_logits"]
            ret_sample = autoregressive_generate(
                ret_logits_fn,
                inputs,
                "returns-to-go",
                sequence_length,
                generator=rng,
                deterministic=deterministic,
                sample_fn=ret_sample_fn,
            )
            inputs["returns-to-go"] = ret_sample

        # Generate a sample from action logits.
        act_logits = logits_fn(inputs)["action_logits"][:, timestep, :]
        act_sample = sample_from_logits(
            act_logits,
            generator=rng,
            deterministic=deterministic,
            temperature=action_temperature,
            top_percentile=action_top_percentile,
        )
        return act_sample
