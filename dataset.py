from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import Dataset

class BilingualDataset(Dataset):
    def __init__(self, ds, tokenizer_src, tokenizer_tgt, lang_src, lang_tgt, seq_len) -> None:
        super().__init__()
        self.ds = ds
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.lang_src = lang_src
        self.lang_tgt = lang_tgt
        self.seq_len = seq_len

        self.sos_token = torch.tensor([tokenizer_src.token_to_id("[SOS]")], dtype=torch.int64)
        self.eos_token = torch.tensor([tokenizer_src.token_to_id("[EOS]")], dtype=torch.int64)
        self.pad_token = torch.tensor([tokenizer_src.token_to_id("[PAD]")], dtype=torch.int64)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index: Any) -> Any:
        src_target_pair = self.ds[index]
        src_text = src_target_pair['translation'][self.lang_src]
        tgt_text = src_target_pair['translation'][self.lang_tgt]

        enc_input_tokens = self.tokenizer_src.encode(src_text).ids
        dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids

        # use [PAD] token to pad the sequences to the same length
        enc_num_padding_tokens = self.seq_len - len(enc_input_tokens) - 2  # -2 for [SOS] and [EOS]
        dec_num_padding_tokens = self.seq_len - len(dec_input_tokens) - 1  # only -1 for [SOS] since we don't add [EOS] to the decoder input

        if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0:
            raise ValueError(f"Sequence length is too short for the input text. Increase seq_len to at least {max(len(enc_input_tokens), len(dec_input_tokens)) + 2}")

        encoder_input = torch.cat([
            self.sos_token,
            torch.tensor(enc_input_tokens, dtype=torch.int64),
            self.eos_token,
            torch.full((enc_num_padding_tokens,), self.pad_token.item(), dtype=torch.int64),
        ])

        decoder_input = torch.cat([
            self.sos_token,
            torch.tensor(dec_input_tokens, dtype=torch.int64),
            torch.full((dec_num_padding_tokens,), self.pad_token.item(), dtype=torch.int64),
        ])

        label = torch.cat([
            torch.tensor(dec_input_tokens, dtype=torch.int64),
            self.eos_token,
            torch.full((dec_num_padding_tokens,), self.pad_token.item(), dtype=torch.int64),
        ])      # the ground truth used to train
        assert label.size(0) == self.seq_len, f"Label length {label.size(0)} does not match seq_len {self.seq_len}"

        assert encoder_input.size(0) == self.seq_len, f"Encoder input length {encoder_input.size(0)} does not match seq_len {self.seq_len}"
        assert decoder_input.size(0) == self.seq_len, f"Decoder input length {decoder_input.size(0)} does not match seq_len {self.seq_len}"


        return {
            'encoder_input': encoder_input, # size = seq_len
            'decoder_input': decoder_input, # size = seq_len
            'encoder_mask': (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(), # size = (1, 1, seq_len)
            'decoder_mask': (decoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int() & causal_mask(decoder_input.size(0)), # (1, seq_len, seq_len)
            # causal mask means that the decoder can only attend to previous tokens and not future tokens
            'label': label, # size = seq_len
            'src_text': src_text,
            'tgt_text': tgt_text    # for visualization purposes, not used for training
        }

def causal_mask(size):
    mask = torch.triu(torch.ones(1, size, size), diagonal=1).type(torch.int32)
    return mask == 0