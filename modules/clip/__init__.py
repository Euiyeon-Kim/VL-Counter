from typing import Any, Union, List
from pkg_resources import packaging

import torch

from modules.clip.clip_text_encoder import CLIPTextEncoder
from modules.clip.clip_image_encoder import CLIPVisionEncoder
from modules.clip.clip_surgery_image_encoder import CLIPSurgeryVisionEncoder
from modules.clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_tokenizer = _Tokenizer()

PROMPT_TEMPLATES = [
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
    'a photo of a number of {}.',
    'a photo of a number of small {}.',
    'a photo of a number of medium {}.',
    'a photo of a number of large {}.',
    'a photo of several {}.',
    'a photo of several small {}.',
    'a photo of several medium {}.',
    'a photo of several large {}.',
    'this is a photo of a number of {}.',
    'this is a photo of a number of small {}.',
    'this is a photo of a number of medium {}.',
    'this is a photo of a number of large {}.',
    'this is a photo of several {}.',
    'this is a photo of several small {}.',
    'this is a photo of several medium {}.',
    'this is a photo of several large {}.',
    'a number of {} in the scene.',
    'a photo of a number of {} in the scene.',
    'several {} in the scene.',
    'a photo of several {} in the scene.',
    'there are a number of {} in the scene.',
    'there are several {} in the scene.',
    'these are a number of {} in the scene.',
    'these are several {} in the scene.',
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
        sentences = [template.format(classname) for template in PROMPT_TEMPLATES]
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


def build_s_img_encoder(args):
    clip_s_img_encoder = CLIPSurgeryVisionEncoder(
        input_resolution=args.input_resolution,
        pretrained=args.clip_path
    )
    added_weight = clip_s_img_encoder.init_weights()
    if args.fix_img_encoder:
        for name, p in clip_s_img_encoder.named_parameters():
            if name not in added_weight:  # pretrained weight
                p.requires_grad = False
    return clip_s_img_encoder


if __name__ == '__main__':
    import torch
    from PIL import Image
    from torchvision import transforms
    from modules.clip.clip_surgery_image_encoder import CLIPSurgeryVisionEncoder

    RESOLUTION = 512
    IMAGE = Image.open("../../datasets/FSC147_384_V2/images_384_VarV2/343.jpg").resize((RESOLUTION, RESOLUTION))
    # IMAGE = Image.open("../../kitti.png").resize((RESOLUTION, RESOLUTION))
    NUM = RESOLUTION // 16
    TEXT_CLASSES = ["kiwi", ""]
    img_transform = transforms.Compose([
        # transforms.Resize(RESOLUTION, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])
    SENTENCES = []
    for txt in TEXT_CLASSES:
        SENTENCES.append([template.format(txt.lower()) for template in PROMPT_TEMPLATES])

    text_encoder = CLIPTextEncoder(context_length=77, vocab_size=49408, embed_dim=512,
                                   transformer_width=512, transformer_heads=8, transformer_layers=12)
    text_encoder.init_weights(pretrained='../../pretrained/ViT-B-16.pt')
    text_encoder = text_encoder.to(DEVICE)

    img_encoder = CLIPSurgeryVisionEncoder(input_resolution=RESOLUTION, pretrained='../../pretrained/ViT-B-16.pt')
    img_encoder.init_weights()
    img_encoder.to(DEVICE)

    with torch.no_grad():
        embeds = []
        for s in SENTENCES:
            txt_tokens = tokenize(s).to(DEVICE)
            txt_embedded = text_encoder(txt_tokens)
            txt_embedded = txt_embedded / txt_embedded.norm(dim=-1, keepdim=True)
            txt_embedded = torch.mean(txt_embedded, dim=0)
            txt_embedded = txt_embedded / txt_embedded.norm()
            embeds.append(txt_embedded)
        txt_embedded = embed_classname(text_encoder, TEXT_CLASSES)

        img = img_transform(IMAGE).to(DEVICE).unsqueeze(0)
        cls_token, patch_feat = img_encoder(img)

        cls_token = cls_token / cls_token.norm(dim=-1, keepdim=True)
        patch_feat = patch_feat / patch_feat.norm(dim=-1, keepdim=True)
        patch_feat = patch_feat.reshape(1, NUM, NUM, 512).permute(0, 3, 1, 2).contiguous()

        import torch.nn.functional as F
        outputs = []
        for embed in txt_embedded:
            output = F.conv2d(patch_feat, embed.unsqueeze(0).unsqueeze(-1).unsqueeze(-1))
            outputs.append(output)
        outputs = torch.stack(outputs, dim=1).squeeze(2)

        for i in range(NUM):
            for j in range(NUM):
                # img_v = patch_feat[i, j, :].unsqueeze(0)
                # logits_for_img_v = img_v @ txt_embedded.t()
                # probs_for_img_v = torch.softmax(logits_for_img_v, dim=-1)
                probs_for_img_v = torch.softmax(outputs[:, :, i, j], dim=-1)
                m_idx = torch.argmin(probs_for_img_v, dim=-1)
                log_txt = f"{TEXT_CLASSES[m_idx.item()]} \t h={i}, w={j} \t "
                for prob, class_name in zip(probs_for_img_v[0], TEXT_CLASSES):
                    log_txt = log_txt + f"{class_name}: {prob.item()} \t"
                print(log_txt)

        normed_cls = cls_token / cls_token.norm(dim=-1, keepdim=True)
        logits_per_image = normed_cls @ txt_embedded.t()

