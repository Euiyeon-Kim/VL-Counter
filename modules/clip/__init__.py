from typing import Any, Union, List
from pkg_resources import packaging

import torch

from .clip_text_encoder import CLIPTextEncoder
from .clip_image_encoder import CLIPVisionEncoder
from .simple_tokenizer import SimpleTokenizer as _Tokenizer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_tokenizer = _Tokenizer()

PROMPTS = [
    'a photo of a {}.',
    'a photo of a small {}.',
    'a photo of a medium {}.',
    'a photo of a large {}.',
    'this is a photo of a small {}.',
    'this is a photo of a medium {}.',
    'this is a photo of a large {}.',
    'this is a photo of a {}.',
    'a {} in the scene.',
    'a photo of a {} in the scene.',
    'there is a {} in the scene.',
    'there is the {} in the scene.',
    'this is a {} in the scene.',
    'this is the {} in the scene.',
    'this is one {} in the scene.',
    # 'a photo of a number of {}.',
    # 'a photo of a number of small {}.',
    # 'a photo of a number of medium {}.',
    # 'a photo of a number of large {}.',
    # 'a photo of several {}.',
    # 'a photo of several small {}.',
    # 'a photo of several medium {}.',
    # 'a photo of several large {}.',
    # 'this is a photo of a number of {}.',
    # 'this is a photo of a number of small {}.',
    # 'this is a photo of a number of medium {}.',
    # 'this is a photo of a number of large {}.',
    # 'this is a photo of several {}.',
    # 'this is a photo of several small {}.',
    # 'this is a photo of several medium {}.',
    # 'this is a photo of several large {}.',
    # 'a number of {} in the scene.',
    # 'a photo of a number of {} in the scene.',
    # 'several {} in the scene.',
    # 'a photo of several {} in the scene.',
    # 'there are a number of {} in the scene.',
    # 'there are several {} in the scene.',
    # 'these are a number of {} in the scene.',
    # 'these are several {} in the scene.',
]


def tokenize(texts: Union[str, List[str]], context_length: int = 77, truncate: bool = False) \
        -> Union[torch.IntTensor, torch.LongTensor]:
    """
    Returns the tokenized representation of given input string(s)

    Parameters
    ----------
    texts : Union[str, List[str]]
        An input string or a list of input strings to tokenize

    context_length : int
        The context length to use; all CLIP models use 77 as the context length

    truncate: bool
        Whether to truncate the text in case its encoding is longer than the context length

    Returns
    -------
    A two-dimensional tensor containing the resulting tokens, shape = [number of input strings, context_length].
    We return LongTensor when torch version is <1.8.0, since older index_select requires indices to be long.
    """
    if isinstance(texts, str):
        texts = [texts]

    sot_token = _tokenizer.encoder["<|startoftext|>"]
    eot_token = _tokenizer.encoder["<|endoftext|>"]
    all_tokens = [[sot_token] + _tokenizer.encode(text) + [eot_token] for text in texts]
    if packaging.version.parse(torch.__version__) < packaging.version.parse("1.8.0"):
        result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)
    else:
        result = torch.zeros(len(all_tokens), context_length, dtype=torch.int)

    for i, tokens in enumerate(all_tokens):
        if len(tokens) > context_length:
            if truncate:
                tokens = tokens[:context_length]
                tokens[-1] = eot_token
            else:
                raise RuntimeError(f"Input {texts[i]} is too long for context length {context_length}")
        result[i, :len(tokens)] = torch.tensor(tokens)

    return result


@torch.no_grad()
def embed_classname(txt_embedder, classnames):
    embeddings = []
    for classname in classnames:
        sentences = [prompt.format(classname) for prompt in PROMPTS]
        tokenized = tokenize(sentences).to(DEVICE)
        embedded = txt_embedder(tokenized)
        embedded = torch.mean(embedded, dim=0)
        embeddings.append(embedded)
    return torch.stack(embeddings, dim=0)


def build_txt_encoder(args):
    clip_txt_encoder = CLIPTextEncoder(pretrained=args.clip_path)
    added_weight = clip_txt_encoder.init_weights()
    if args.fix_txt_encoder:
        for name, p in clip_txt_encoder.named_parameters():
            if name not in added_weight:        # pretrained weight
                p.requires_grad = False
    return clip_txt_encoder


def build_img_encoder(args):
    clip_img_encoder = CLIPVisionEncoder(
        input_resolution=args.input_resolution,
        out_indices=args.out_indices,
        pretrained=args.clip_path
    )
    added_weight = clip_img_encoder.init_weights()
    if args.fix_img_encoder:
        for name, p in clip_img_encoder.named_parameters():
            if name not in added_weight:  # pretrained weight
                p.requires_grad = False
    return clip_img_encoder

