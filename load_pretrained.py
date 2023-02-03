import pickle

import numpy as np
import torch
import torch.nn as nn


def load_jax_weights(model, model_params):
    def load_ln(m, k):
        m.weight.data = torch.from_numpy(model_params[k]["scale"])
        m.bias.data = torch.from_numpy(model_params[k]["offset"])

    def load_linear(m, k):
        m.weight.data = torch.from_numpy(model_params[k]["w"]).t()
        m.bias.data = torch.from_numpy(model_params[k]["b"])

    def load_attn(attn, k):
        qkv_w = np.concatenate(
            [
                model_params[k + "/query"]["w"],
                model_params[k + "/key"]["w"],
                model_params[k + "/value"]["w"],
            ],
            axis=-1,
        )
        attn.qkv.weight.data = torch.from_numpy(qkv_w).t()

        qkv_b = np.concatenate(
            [
                model_params[k + "/query"]["b"],
                model_params[k + "/key"]["b"],
                model_params[k + "/value"]["b"],
            ]
        )
        attn.qkv.bias.data = torch.from_numpy(qkv_b)

        load_linear(attn.proj, k + "/linear")

    def load_mlp(mlp, k):
        load_linear(mlp.fc1, k + "/linear")
        load_linear(mlp.fc2, k + "/linear_1")

    def load_transformer(transformer):
        prefix = "decision_transformer/~/sequence"
        for i in range(transformer._num_layers):
            block = transformer.layers[i]

            load_ln(block.ln_1, f"{prefix}/h{i}_ln_1")
            load_attn(block.attn, f"{prefix}/h{i}_attn")

            load_ln(block.ln_2, f"{prefix}/h{i}_ln_2")
            load_mlp(block.mlp, f"{prefix}/h{i}_mlp")
        load_ln(transformer.norm_f, f"{prefix}/ln_f")

    def load_embedding(m, k):
        m.weight.data = torch.from_numpy(model_params[k]["embeddings"])

    def load_image_emb(m, k):
        # [H x W x Cin x Cout] -> [Cout, Cin, H, W]
        m.weight.data = torch.from_numpy(model_params[k]["w"]).permute(3, 2, 0, 1)
        m.bias.data = torch.from_numpy(model_params[k]["b"])

    # --- Load transformer

    load_transformer(model.transformer)

    # --- Load model

    load_linear(model.act_linear, "decision_transformer/act_linear")
    load_linear(model.ret_linear, "decision_transformer/ret_linear")
    if model.predict_reward:
        load_linear(model.rew_linear, "decision_transformer/rew_linear")

    model.image_pos_enc = nn.Parameter(torch.tensor(model_params["decision_transformer"]["image_pos_enc"]))
    model.positional_embedding = nn.Parameter(torch.tensor(model_params["decision_transformer"]["positional_embeddings"]))

    load_image_emb(model.image_emb, "decision_transformer/~_embed_inputs/image_emb")

    load_embedding(model.ret_emb, "decision_transformer/~_embed_inputs/embed")
    load_embedding(model.act_emb, "decision_transformer/~_embed_inputs/embed_1")
    if model.predict_reward:
        load_embedding(model.rew_emb, "decision_transformer/~_embed_inputs/embed_2")
