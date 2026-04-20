from __future__ import annotations

from . import mpl_config as _mpl_config  # noqa: F401

import matplotlib.pyplot as plt
import numpy as np

from .plotting import BLUE, CHAPTER_GRAY, ORANGE, book_subplots, latex_escape, save_figure


def _attention_modules():
    from bertviz import head_view, model_view
    from bertviz.neuron_view import show
    from bertviz.transformers_neuron_view import BertModel as NeuronBertModel
    from bertviz.transformers_neuron_view import BertTokenizer as NeuronBertTokenizer
    from transformers import BertModel, BertTokenizer

    return {
        "head_view": head_view,
        "model_view": model_view,
        "show": show,
        "BertModel": BertModel,
        "BertTokenizer": BertTokenizer,
        "NeuronBertModel": NeuronBertModel,
        "NeuronBertTokenizer": NeuronBertTokenizer,
    }


def load_attention_data(
    sentence_a: str,
    sentence_b: str | None = None,
    *,
    model_name: str = "bert-base-uncased",
):
    modules = _attention_modules()
    tokenizer = modules["BertTokenizer"].from_pretrained(model_name)
    model = modules["BertModel"].from_pretrained(
        model_name,
        output_attentions=True,
        return_dict=True,
    )

    if sentence_b is None:
        inputs = tokenizer(sentence_a, return_tensors="pt")
        outputs = model(**inputs)
        input_ids = inputs["input_ids"][0]
        tokens = tokenizer.convert_ids_to_tokens(input_ids)
        sentence_b_start = None
    else:
        inputs = tokenizer.encode_plus(sentence_a, sentence_b, return_tensors="pt")
        outputs = model(inputs["input_ids"], token_type_ids=inputs["token_type_ids"])
        input_ids = inputs["input_ids"][0]
        tokens = tokenizer.convert_ids_to_tokens(input_ids)
        sentence_b_start = inputs["token_type_ids"][0].tolist().index(1)

    return {
        "tokenizer": tokenizer,
        "model": model,
        "inputs": inputs,
        "outputs": outputs,
        "tokens": tokens,
        "sentence_b_start": sentence_b_start,
    }


def create_head_view(
    sentence_a: str,
    sentence_b: str,
    *,
    model_name: str = "bert-base-uncased",
):
    modules = _attention_modules()
    data = load_attention_data(sentence_a, sentence_b, model_name=model_name)
    return modules["head_view"](
        data["outputs"].attentions,
        data["tokens"],
        data["sentence_b_start"],
    )


def create_model_view(
    sentence_a: str,
    sentence_b: str,
    *,
    model_name: str = "bert-base-uncased",
):
    modules = _attention_modules()
    data = load_attention_data(sentence_a, sentence_b, model_name=model_name)
    return modules["model_view"](
        data["outputs"].attentions,
        data["tokens"],
        data["sentence_b_start"],
    )


def create_neuron_view(
    sentence_a: str,
    sentence_b: str,
    *,
    model_name: str = "bert-base-uncased",
    layer: int = 4,
    head: int = 3,
):
    modules = _attention_modules()
    model = modules["NeuronBertModel"].from_pretrained(model_name, output_attentions=True)
    tokenizer = modules["NeuronBertTokenizer"].from_pretrained(model_name, do_lower_case=True)
    return modules["show"](model, "bert", tokenizer, sentence_a, sentence_b, layer=layer, head=head)


def save_attention_heatmaps_pdf(
    text: str,
    output_path,
    *,
    model_name: str = "bert-base-uncased",
):
    data = load_attention_data(text, model_name=model_name)
    tokenizer = data["tokenizer"]
    inputs = data["inputs"]
    attentions = data["outputs"].attentions
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

    n_layers = len(attentions)
    n_heads = attentions[0].shape[1]
    seq_len = len(tokens)

    fig, axes = book_subplots(n_layers, n_heads, size="grid", squeeze=False, extra_height=-0.8)

    for layer_idx in range(n_layers):
        layer_attention = attentions[layer_idx][0].detach().cpu().numpy()
        for head_idx in range(n_heads):
            axis = axes[layer_idx][head_idx]
            axis.imshow(layer_attention[head_idx], aspect="auto")
            if layer_idx == n_layers - 1:
                axis.set_xticks(range(seq_len))
                axis.set_xticklabels([latex_escape(token) for token in tokens], rotation=90, fontsize=6)
            else:
                axis.set_xticks([])
            if head_idx == 0:
                axis.set_yticks(range(seq_len))
                axis.set_yticklabels([latex_escape(token) for token in tokens], fontsize=6)
            else:
                axis.set_yticks([])
            axis.set_title(rf"$L_{{{layer_idx + 1}}}\cdot H_{{{head_idx + 1}}}$", fontsize=8)

    fig.tight_layout()
    save_figure(fig, output_path, tight_layout=False)


def plot_self_attention_connections(
    sentence: str,
    *,
    model_name: str = "bert-base-uncased",
    layer_idx: int = 0,
    head_idx: int = 0,
):
    data = load_attention_data(sentence, model_name=model_name)
    tokenizer = data["tokenizer"]
    attentions = data["outputs"].attentions

    attention = attentions[layer_idx][0, head_idx].detach().cpu().numpy()
    tokens = tokenizer.convert_ids_to_tokens(data["inputs"]["input_ids"][0])[1:-1]
    attention = attention[1:-1, 1:-1]
    n_tokens = len(tokens)
    x_positions = np.arange(n_tokens)

    fig, axis = book_subplots(size="single")
    for index, token in enumerate(tokens):
        escaped_token = latex_escape(token)
        axis.text(x_positions[index], 1.05, escaped_token, ha="center", va="bottom", color=CHAPTER_GRAY, fontsize=12)
        axis.text(x_positions[index], -0.05, escaped_token, ha="center", va="top", color=CHAPTER_GRAY, fontsize=12)

    for source_index in range(n_tokens):
        for target_index in range(n_tokens):
            weight = attention[source_index, target_index]
            if weight <= 0.10:
                continue
            color = BLUE if weight > 0.20 else ORANGE
            axis.plot(
                [x_positions[source_index], x_positions[target_index]],
                [1.0, 0.0],
                linewidth=weight * 8,
                alpha=min(weight * 3, 1.0),
                color=color,
            )

    axis.set_xlim(-0.5, n_tokens - 0.5)
    axis.set_ylim(-0.2, 1.2)
    axis.axis("off")
    fig.suptitle(
        f"Self-attention connections (L{layer_idx + 1}, H{head_idx + 1})",
        fontsize=14,
        y=1.05,
    )
    fig.tight_layout()
    return fig
