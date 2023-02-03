from typing import Optional, Tuple

import numpy as np
import torch
import torch.distributions as D
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# https://github.com/rwightman/pytorch-image-models/blob/29fda20e6d428bf636090ab207bbcf60617570ca/timm/layers/weight_init.py#L99
def variance_scaling_(tensor: Tensor, scale=1.0, mode="fan_in", distribution="trunc_normal") -> Tensor:
    fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(tensor)
    if mode == "fan_in":
        scale /= max(1.0, fan_in)
    elif mode == "fan_out":
        scale /= max(1.0, fan_out)
    elif mode == "fan_avg":
        scale /= max(1.0, (fan_in + fan_out) / 2.0)

    if distribution == "trunc_normal":
        stddev = np.sqrt(scale)
        # Adjust stddev for truncation.
        # Constant from scipy.stats.truncnorm.std(a=-2, b=2, loc=0., scale=1.)
        stddev = stddev / 0.87962566103423978
        return nn.init.trunc_normal_(tensor, std=stddev)
    elif distribution == "normal":
        stddev = np.sqrt(scale)
        return nn.init.normal_(tensor, std=stddev)
    elif distribution == "uniform":
        limit = np.sqrt(3.0 * scale)
        return nn.init.uniform_(tensor, -limit, limit)
    else:
        raise ValueError(f"Invalid distribution: {distribution}")


def cross_entropy(logits, labels):
    r"""Applies sparse cross entropy loss between logits and target labels."""
    labels = F.one_hot(labels.long(), logits.shape[-1]).to(dtype=logits.dtype)
    loss = -labels * F.log_softmax(logits, dim=-1)
    return torch.mean(loss)


def accuracy(logits, labels):
    r"""Compute accuracy between predicted labels from logits and target labels."""
    predicted_label = torch.argmax(logits, dim=-1)
    acc = torch.eq(predicted_label, labels).to(dtype=torch.float32)
    return torch.mean(acc)


def sample_from_logits(
    logits: Tensor,
    generator: Optional[torch.Generator] = None,
    deterministic: Optional[bool] = False,
    temperature: Optional[float] = 1e0,
    top_k: Optional[int] = None,
    top_percentile: Optional[float] = None,
) -> Tuple[Tensor, Tensor]:
    r"""Generate a categorical sample from given logits."""
    if deterministic:
        sample = torch.argmax(logits, dim=-1)
    else:
        if top_percentile is not None:
            # percentile: 0 to 100, quantile: 0 to 1
            percentile = torch.quantile(logits, top_percentile / 100, dim=-1)
            logits = torch.where(logits > percentile[..., None], logits, -np.inf)
        if top_k is not None:
            logits, top_indices = torch.topk(logits, top_k)
        sample = D.Categorical(logits=temperature * logits).sample()
        # probs = F.softmax(temperature * logits, dim=-1)
        # sample = torch.multinomial(probs, num_samples=1, generator=generator)
        if top_k is not None:
            sample_shape = sample.shape
            # Flatten top-k indices and samples for easy indexing.
            top_indices = torch.reshape(top_indices, [-1, top_k])
            sample = sample.flatten()
            sample = top_indices[torch.arange(len(sample)), sample]
            # Reshape samples back to original dimensions.
            sample = torch.reshape(sample, sample_shape)
    return sample


def autoregressive_generate():
    pass


def encode_reward(rew: Tensor) -> Tensor:
    r"""Encode reward values into values expected by the model."""
    # 0: no reward   1: positive reward   2: terminal reward   3: negative reward
    rew = (rew > 0) * 1 + (rew < 0) * 3
    return rew.to(dtype=torch.int32)


def encode_return(ret: Tensor, ret_range: Tuple[int]) -> Tensor:
    r"""Encode (possibly negative) return values into discrete return tokens."""
    ret = ret.to(dtype=torch.int32)
    ret = torch.clip(ret, ret_range[0], ret_range[1])
    ret = ret - ret_range[0]
    return ret


def decode_return(ret: Tensor, ret_range: Tuple[int]) -> Tensor:
    r"""Decode discrete return tokens into return values."""
    ret = ret.to(dtype=torch.int32)
    ret = ret + ret_range[0]
    return ret
