import os
import pandas as pd
import time
import nucleotide_transformer
import haiku as hk
import jax
import jax.numpy as jnp
from nucleotide_transformer.pretrained import get_pretrained_model
import numpy as np



def split_sequence(sequence, max_len):
    center_index = len(sequence) // 2

    center_segment = sequence[center_index - max_len // 2 : center_index + max_len // 2 + 1]

    segments = [center_segment]

    # ????
    left_start = center_index - max_len // 2
    while left_start > 0:
        left_start -= max_len
        left_end = left_start + max_len
        if left_start < 0:
            left_start = 0
        segments.insert(0, sequence[left_start:left_end])

    right_start = center_index + max_len // 2 + 1
    while right_start < len(sequence):
        right_end = right_start + max_len
        segments.append(sequence[right_start:right_end])
        right_start += max_len
    return segments
    # return segments[:len(segments)//2]

def inference_single(seq_list):
    model_name = '500M_human_ref'
    parameters, forward_fn, tokenizer, config = get_pretrained_model(
        model_name=model_name,
        embeddings_layers_to_save=(20,),
        attention_maps_to_save=((1, 4), (7, 18)),
        max_positions=501,
        # If the progress bar gets stuck at the start of the model wieghts download,
        # you can set verbose=False to download without the progress bar.
        verbose=True
    )
    forward_fn = hk.transform(forward_fn)
    embedding_result = []
    max_len = 500
    co = 0
    for seq_item in seq_list:
        co += 1
        print(co)
        seq_item = seq_item.upper()
        sequence = seq_item
        segments = split_sequence(sequence, max_len)
        segment_embeddings = []
        sequences = segments
        tokens_ids = [b[1] for b in tokenizer.batch_tokenize(sequences)]
        tokens_str = [b[0] for b in tokenizer.batch_tokenize(sequences)]
        tokens = jnp.asarray(tokens_ids, dtype=jnp.int32)
        random_key = jax.random.PRNGKey(0)
        outs = forward_fn.apply(parameters, random_key, tokens)
        embeddings = outs["embeddings_20"][:, 1:, :]  # removing CLS token
        padding_mask = jnp.expand_dims(tokens[:, 1:] != tokenizer.pad_token_id, axis=-1)
        masked_embeddings = embeddings * padding_mask  # multiply by 0 pad tokens embeddings
        sequences_lengths = jnp.sum(padding_mask, axis=1)
        for i in range(len(segments)):
            temp_embeddings = jnp.sum(masked_embeddings[i], axis=0) / sequences_lengths[i]
            segment_embeddings.append(np.array(jax.device_get(temp_embeddings)))
        average_embedding = np.mean(segment_embeddings, axis=0)
        # print(average_embedding.shape)
        embedding_result.append(average_embedding)
    print(np.array(embedding_result).shape)
    return embedding_result
# sign_prediction slope_prediction
# Nerve_Tibial Heart_Left_Ventricle Esophagus_Mucosa
for task in ['slope_prediction', 'sign_prediction']:
    for tissue in ['Nerve_Tibial', 'Heart_Left_Ventricle', 'Esophagus_Mucosa']:
        for dirpath, dirnames, filenames in os.walk('/root/autodl-tmp/benchmark_meqtl_dataset/' + task + '/' + tissue + '/'):
            for file_name in filenames:
                if 'dataset' not in file_name or 'NT' in file_name or 'hyena' in file_name or 'large' not in file_name:
                    continue
                data = pd.read_pickle('/root/autodl-tmp/benchmark_meqtl_dataset/' + task + '/' + tissue + '/' + file_name)
                start_time = time.time()
                data['NT_before'] = inference_single(data.loc[:, 'seq_before'])
                data['NT_after'] = inference_single(data.loc[:, 'seq_after'])
                data.to_pickle('/root/autodl-tmp/benchmark_meqtl_dataset/' + task + '/' + tissue + '/NT_' + file_name)
                end_time = time.time()
                elapsed_time = end_time - start_time
                print(file_name + f": {elapsed_time} s")