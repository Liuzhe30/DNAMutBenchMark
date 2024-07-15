import os
import pickle

import numpy as np
import pandas as pd

import json
import os
import subprocess
import torch
import time
# import transformers
from transformers import PreTrainedModel
import re
from standalone_hyenadna import HyenaDNAModel
from standalone_hyenadna import CharacterTokenizer
import torch.nn as nn
import gc

def split_sequence(sequence, max_len):
    # 计算中心位置
    center_index = len(sequence) // 2

    # 中间段的长度为max_len + 1
    center_segment = sequence[center_index - max_len // 2 : center_index + max_len // 2 + 1]

    segments = [center_segment]

    # 往左扩展
    left_start = center_index - max_len // 2
    while left_start > 0:
        left_start -= max_len
        left_end = left_start + max_len
        if left_start < 0:
            left_start = 0
        segments.insert(0, sequence[left_start:left_end])

    # 往右扩展
    right_start = center_index + max_len // 2 + 1
    while right_start < len(sequence):
        right_end = right_start + max_len
        segments.append(sequence[right_start:right_end])
        right_start += max_len

    return segments

# helper 1
def inject_substring(orig_str):
    """Hack to handle matching keys between models trained with and without
    gradient checkpointing."""

    # modify for mixer keys
    pattern = r"\.mixer"
    injection = ".mixer.layer"

    modified_string = re.sub(pattern, injection, orig_str)

    # modify for mlp keys
    pattern = r"\.mlp"
    injection = ".mlp.layer"

    modified_string = re.sub(pattern, injection, modified_string)

    return modified_string

# helper 2
def load_weights(scratch_dict, pretrained_dict, checkpointing=False):
    """Loads pretrained (backbone only) weights into the scratch state dict."""

    # loop thru state dict of scratch
    # find the corresponding weights in the loaded model, and set it

    # need to do some state dict "surgery"
    for key, value in scratch_dict.items():
        if 'backbone' in key:
            # the state dicts differ by one prefix, '.model', so we add that
            key_loaded = 'model.' + key
            # breakpoint()
            # need to add an extra ".layer" in key
            if checkpointing:
                key_loaded = inject_substring(key_loaded)
            try:
                scratch_dict[key] = pretrained_dict[key_loaded]
            except:
                raise Exception('key mismatch in the state dicts!')

    # scratch_dict has been updated
    return scratch_dict

class HyenaDNAPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    base_model_prefix = "hyenadna"

    def __init__(self, config):
        pass

    def forward(self, input_ids, **kwargs):
        return self.model(input_ids, **kwargs)

    @classmethod
    def from_pretrained(cls,
                        path,
                        model_name,
                        download=False,
                        config=None,
                        device='cpu',
                        use_head=False,
                        n_classes=2,
                      ):
        # first check if it is a local path
        pretrained_model_name_or_path = os.path.join(path, model_name)
        if os.path.isdir(pretrained_model_name_or_path) and download == False:
            if config is None:
                config = json.load(open(os.path.join(pretrained_model_name_or_path, 'config.json')))
        else:
            hf_url = f'https://huggingface.co/LongSafari/{model_name}'

            subprocess.run(f'rm -rf {pretrained_model_name_or_path}', shell=True)
            command = f'mkdir -p {path} && cd {path} && git lfs install && git clone {hf_url}'
            subprocess.run(command, shell=True)

            if config is None:
                config = json.load(open(os.path.join(pretrained_model_name_or_path, 'config.json')))

        scratch_model = HyenaDNAModel(**config, use_head=use_head, n_classes=n_classes)  # the new model format
        loaded_ckpt = torch.load(
            os.path.join(pretrained_model_name_or_path, 'weights.ckpt'),
            map_location=torch.device(device)
        )

        # need to load weights slightly different if using gradient checkpointing
        if config.get("checkpoint_mixer", False):
            checkpointing = config["checkpoint_mixer"] == True or config["checkpoint_mixer"] == True
        else:
            checkpointing = False

        # grab state dict from both and load weights
        state_dict = load_weights(scratch_model.state_dict(), loaded_ckpt['state_dict'], checkpointing=checkpointing)

        # scratch model has now been updated
        scratch_model.load_state_dict(state_dict)
        print("Loaded pretrained weights ok!")
        return scratch_model




####################################################################################################




"""# Inference (450k to 1M tokens)!

If all you're interested in is getting embeddings on long DNA sequences
(inference), then we can do that right here in Colab!


*   We provide an example how to load the weights from Huggingface.
*   On the free tier, which uses a
T4 GPU w/16GB of memory, we can process 450k tokens / nucleotides.
*   For processing 1M tokens, you'll need an A100, which Colab offers as a paid tier.
*   (Don't forget to run the entire notebook above too)

--

To pretrain or fine-tune the 1M long sequence model (8 layers, d_model=256),
you'll need 8 A100s 80GB, and all that code is in the main repo!
"""

#@title Single example
import json
import os
import subprocess
# import transformers
from transformers import PreTrainedModel

def inference_single(seq_list):

    '''
    this selects which backbone to use, and grabs weights/ config from HF
    4 options:
      'hyenadna-tiny-1k-seqlen'   # fine-tune on colab ok
      'hyenadna-small-32k-seqlen'
      'hyenadna-medium-160k-seqlen'  # inference only on colab
      'hyenadna-medium-450k-seqlen'  # inference only on colab
      'hyenadna-large-1m-seqlen'  # inference only on colab
    '''

    # you only need to select which model to use here, we'll do the rest!
    pretrained_model_name = 'hyenadna-tiny-1k-seqlen'

    max_lengths = {
        'hyenadna-tiny-1k-seqlen': 1024,
        'hyenadna-small-32k-seqlen': 32768,
        'hyenadna-medium-160k-seqlen': 160000,
        'hyenadna-medium-450k-seqlen': 450000,  # T4 up to here
        'hyenadna-large-1m-seqlen': 1_000_000,  # only A100 (paid tier)
    }

    max_length = max_lengths[pretrained_model_name]  # auto selects

    # data settings:
    use_padding = True
    rc_aug = False  # reverse complement augmentation
    add_eos = False  # add end of sentence token

    # we need these for the decoder head, if using
    use_head = False
    n_classes = 2  # not used for embeddings only

    # you can override with your own backbone config here if you want,
    # otherwise we'll load the HF one in None
    backbone_cfg = None

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Using device:", device)

    # instantiate the model (pretrained here)
    if pretrained_model_name in ['hyenadna-tiny-1k-seqlen',
                                 'hyenadna-small-32k-seqlen',
                                 'hyenadna-medium-160k-seqlen',
                                 'hyenadna-medium-450k-seqlen',
                                 'hyenadna-large-1m-seqlen']:
        # use the pretrained Huggingface wrapper instead
        model = HyenaDNAPreTrainedModel.from_pretrained(
            './checkpoints',
            pretrained_model_name,
            download=False,
            config=backbone_cfg,
            device=device,
            use_head=use_head,
            n_classes=n_classes,
        )

    # from scratch
    elif pretrained_model_name is None:
        model = HyenaDNAModel(**backbone_cfg, use_head=use_head, n_classes=n_classes)

    # create tokenizer
    tokenizer = CharacterTokenizer(
        characters=['A', 'C', 'G', 'T', 'N'],  # add DNA characters, N is uncertain
        model_max_length=max_length + 2,  # to account for special tokens, like EOS
        add_special_tokens=False,  # we handle special tokens elsewhere
        padding_side='left', # since HyenaDNA is causal, we pad on the left
    )
    model = nn.DataParallel(model)
    model.to(device)
    model.eval()
    embedding_result = []
    #### Single embedding example ####
    max_len = 500
    co = 0
    for seq_item in seq_list:
        co += 1
        print(co)
        torch.cuda.empty_cache()
        sequence = seq_item
        segments = split_sequence(sequence, max_len)
        segment_embeddings = []
        for segment in segments:
            tok_seq = tokenizer(segment)
            tok_seq = tok_seq["input_ids"]
            # place on device, convert to tensor
            tok_seq = torch.LongTensor(tok_seq).unsqueeze(0)  # unsqueeze for batch dim
            tok_seq = tok_seq.to(device)

            # prep model and forward
            embeddings = model(tok_seq)
            segment_embeddings.append(torch.mean(embeddings, dim=1).cpu().detach().numpy())

            # 计算加权平均
        average_embedding = np.mean(segment_embeddings, axis=0)
        embedding_result.append(average_embedding)

    return embedding_result
# # uncomment to run! (to get embeddings)

for task in ['slope_prediction', 'sign_prediction']:
    for tissue in ['Nerve_Tibial', 'Heart_Left_Ventricle', 'Esophagus_Mucosa']:
        for dirpath, dirnames, filenames in os.walk('/data3/baoyh/benchmark/dataset/benchmark_eqtl_dataset/' + task + '/' + tissue + '/'):
            for file_name in filenames:
                if 'dataset' not in file_name or 'hyena' in file_name or 'NT' in file_name:
                    continue
                data = pd.read_pickle('/data3/baoyh/benchmark/dataset/benchmark_eqtl_dataset/' + task + '/' + tissue + '/' + file_name)
                start_time = time.time()
                data['hyena_before'] = inference_single(data.loc[:, 'seq_before'])
                data['hyena_after'] = inference_single(data.loc[:, 'seq_after'])
                data.to_pickle('/data3/baoyh/benchmark/dataset/benchmark_eqtl_dataset/' + task + '/' + tissue + '/' + 'hyena_' + file_name)
                print(len(data['hyena_before']))
                print(len(data['hyena_after']))
                end_time = time.time()
                elapsed_time = end_time - start_time
                print(file_name + f"运行时间: {elapsed_time} 秒")



