import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim

from dataset import BilingualDataset, causal_mask
from model import build_transformer
from config import get_config, get_weights_file_path

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

from pathlib import Path
from tqdm import tqdm
import warnings


def greedy_decode(model, source, source_mask, tokenizer_src, tokenizer_tgt, max_len, device):
    sos_idx = tokenizer_tgt.token_to_id("[SOS]")
    eos_idx = tokenizer_tgt.token_to_id("[EOS]")
    # precompute the encoder output and reuse it for every token we get from the decoder
    encoder_output = model.encode(source, source_mask) 
    # initialize the decoder input with the start-of-sequence token
    decoder_input = torch.empty(1,1).fill_(sos_idx).type_as(source).to(device) 

    # keep asking decoder for the next token until we reach the maximum length or the end-of-sequence token
    while True:
        if decoder_input.size(1) == max_len:
            break
        # create a causal mask for the decoder input
        decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).to(device) 
        # calduclate the decoder output
        out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)

        # get the next token 
        prob = model.project(out[:, -1]) # only want the last token in the sequence
        _, next_word = torch.max(prob, dim=1) # get the index of the token with the highest probability
        decoder_input = torch.cat([decoder_input, torch.empty(1,1).type_as(source).fill_(next_word.item()).to(device)], dim=1) # append the next token to the decoder input

        if next_word == eos_idx:
            break
    return decoder_input.squeeze(0) # remove the batch dimension and return the sequence of token ids

def run_validation(model, validation_ds, tokenizer_src, tokenizer_tgt, max_len, device, print_msg, global_state, writer, num_examples=2):
    model.eval()
    count = 0


    # size of control window
    console_width = 80
    with torch.no_grad():   # only validate, don't train
        for batch in validation_ds:
            count += 1
            encoder_input = batch['encoder_input'].to(device) 
            encoder_mask = batch['encoder_mask'].to(device) 

            assert encoder_input.size(0) == 1, "Batch size must be 1 for validation"

            model_out = greedy_decode(model, encoder_input, encoder_mask, tokenizer_src, tokenizer_tgt, max_len, device)

            # compare model_out with label
            source_text = batch['src_text'][0]
            target_text = batch['tgt_text'][0]
            model_out_text = tokenizer_tgt.decode(model_out.detach().cpu().numpy())

            # print it on the console
            print_msg('-' * console_width)
            print_msg(f"Source: {source_text}") # print_msg is used to print when tqdm is on
            print_msg(f"Expected: {target_text}")
            print_msg(f"Predicted: {model_out_text}")

            if count >= num_examples:
                break


def get_all_sentences(ds, lang):
    for item in ds:
        yield item['translation'][lang]

def get_or_build_tokenizer(config, ds, lang):
    # config['tokenizer_file'] = '../tokenizer.json'
    tokenizer_path = Path(config['tokenizer_file'].format(lang))
    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))  # replace unknown tokens with [UNK]
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2)
        tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer)
        tokenizer.save(str(tokenizer_path))

    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))

    return tokenizer

def get_dataset(config):
    ds = load_dataset(config['dataset_name'], f'{config["lang_src"]}-{config["lang_tgt"]}', split='train')

    # build tokenizer
    tokenizer_src = get_or_build_tokenizer(config, ds, config['lang_src'])
    tokenizer_tgt = get_or_build_tokenizer(config, ds, config['lang_tgt'])

    # keep 90% of the dataset for training and 10% for validation
    train_ds_size = int(0.9 * len(ds))
    val_ds_size = len(ds) - train_ds_size
    train_ds, val_ds = torch.utils.data.random_split(ds, [train_ds_size, val_ds_size])

    train_dataset = BilingualDataset(train_ds, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])
    val_dataset = BilingualDataset(val_ds, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])

    max_len_src, max_len_tgt = 0, 0
    for item in ds:
        src_ids = tokenizer_src.encode(item['translation'][config['lang_src']]).ids # provides the token ids for the source text
        tgt_ids = tokenizer_tgt.encode(item['translation'][config['lang_tgt']]).ids # provides the token ids for the target text
        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))

    print(f"Max length of source sentences: {max_len_src}")
    print(f"Max length of target sentences: {max_len_tgt}")

    train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=True)

    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt

def get_model(config, vocab_src_len, vocab_tgt_len):
    return build_transformer(vocab_src_len, vocab_tgt_len, config['seq_len'], config['seq_len'], config['d_model'])

def train_model(config):
    # define the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    Path(config['model_folder']).mkdir(parents=True, exist_ok=True) # make sure the model folder exists

    # load the dataset
    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_dataset(config)
    model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)

    # Tensorboard to visualize the training process
    writer = SummaryWriter(config['experiment_name'])
    optimizer = optim.Adam(model.parameters(), lr=config['lr'], eps=1e-9)

    # loading last saved model if exists
    initial_epoch = 0
    global_step = 0
    if config['preload']:
        model_filename = get_weights_file_path(config, config['preload'])
        print(f'Preloading model {model_filename}')

        state = torch.load(model_filename)
        initial_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']
        loss_fn = state['loss_fn']

    # define the loss function
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_tgt.token_to_id("[PAD]"), label_smoothing=0.1).to(device)
    # ignore the padding token in the target sequence when calculating the loss; label smoothing is a technique to prevent the model from becoming overconfident in its predictions

    # THE TRAINING LOOP
    for epoch in range(initial_epoch, config['num_epochs']):
        model.train()
        batch_iterator = tqdm(train_dataloader, desc=f'processing epoch {epoch:02d}')
        for batch in batch_iterator:

            encoder_input = batch['encoder_input'].to(device) # (batch_size, seq_len)
            decoder_input = batch['decoder_input'].to(device) # (batch_size, seq_len)
            encoder_mask = batch['encoder_mask'].to(device) # (batch_size, 1, 1, seq_len)
            decoder_mask = batch['decoder_mask'].to(device) # (batch_size, 1,1 ,seq_len)

            # run the tensors through the model
            encoder_output = model.encode(encoder_input, encoder_mask) # (batch_size, seq_len, d_model)
            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask)    # (batch_size, seq_len, d_model)
            proj_output = model.project(decoder_output) # (batch_size, seq_len, vocab_tgt_len)

            # extract the label/target tensor from the batch
            labels = batch['labels'].to(device) # (batch_size, seq_len)
            # compute the loss
            loss = loss_fn(proj_output.view(-1, tokenizer_tgt.get_vocab_size()), labels.view(-1))

            # update the progress bar
            batch_iterator.set_postfix({f'loss': f'{loss.item():.4f}'})
            writer.add_scalar('training_loss', loss.item(), global_step)
            writer.flush()

            # backpropagation
            loss.backward()

            # update the weights
            optimizer.step()
            optimizer.zero_grad()

            global_step += 1

        run_validation(model, val_dataloader, tokenizer_src, tokenizer_tgt, config['seq_len'], device, lambda msg: batch_iterator.write(msg), global_step, writer, num_examples=2)
        
        # save the model after each epoch
        model_filename = get_weights_file_path(config, f'{epoch:02d}')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss_fn': loss_fn,
            'global_step': global_step
        }, model_filename)

if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    config = get_config()
    train_model(config)