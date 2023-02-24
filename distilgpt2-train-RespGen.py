# Importing stock libraries
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from sklearn.model_selection import train_test_split
from itertools import chain
import random
import json
from datetime import datetime
# Importing the T5 modules from huggingface/transformers
from transformers import GPT2Tokenizer, GPT2LMHeadModel, get_polynomial_decay_schedule_with_warmup
from torch.utils.tensorboard import SummaryWriter
# from transformers import T5Tokenizer, T5ForConditionalGeneration
# from transformers import T5Tokenizer, AutoConfig, AutoModelForSeq2SeqLM
import csv
from tqdm import tqdm
import pickle
import gzip
import os
import sys
# WandB – Import the wandb library
import wandb
import math

# # Setting up the device for GPU usage
from torch import cuda
device = 'cuda' if cuda.is_available() else 'cpu'
# Creating a custom dataset for reading the dataframe and loading it into the dataloader to pass it to the neural network at a later stage for finetuning the model and to prepare it for predictions

import warnings
warnings.filterwarnings("ignore")

space = 'Ġ'
pre_quote = '’'
end_marks = ['.', ',', '?', '!', '...']
quotes = ['"', '\'']
abbreviations = ['s', 'd', 't', 'm', 're', 'll', 've', 'S', 'D', 'T', 'M', 'Re', 'Ll', 'Ve']

class CustomDataset(Dataset):
    def __init__(self, dialogues, tokenizer, config):       
        self.input_ids = []  # (N, L)
        self.token_type_ids = []  # (N, L)
        self.labels = []  # (N, L)
        self.tokenizer = tokenizer
        
        print(f"Processing data...")
        for dial in tqdm(dialogues):
            hists = []
            # ignore the task title and intro that give the context
            dial = dial.split('[SEP]')
            for u, utter in enumerate(dial):
                tokens = tokenizer.tokenize(utter.strip().replace(pre_quote, quotes[1]))
                # token_list = process_token_list(token_list)
                # text = tokenizer.convert_tokens_to_string(token_list)
                # tokens = tokenizer.tokenize(utter)
                token_ids = tokenizer.convert_tokens_to_ids(tokens)
                if u % 2 == 0:
                    hists.append([0] + token_ids)
                else:
                    hists.append([1] + token_ids)
                    
            for h in range(0, len(hists)):
                if h % 2 == 1:
                    for s in range(h):
                        contexts = hists[s:h+1] # qa
                        input_ids = [config.bos_id] + list(chain.from_iterable(contexts)) + [config.eos_id]
                        if len(input_ids) <= config.MAX_LEN:
                            start_sp_id, next_sp_id = contexts[0][0], contexts[1][0]
                            token_type_ids = [[start_sp_id] * len(ctx) if c % 2 == 0 else [next_sp_id] * len(ctx) for c, ctx in enumerate(contexts)]
                            assert token_type_ids[-1][0] == 1
                            token_type_ids = [start_sp_id] + list(chain.from_iterable(token_type_ids)) + [1]
                            assert len(input_ids) == len(token_type_ids)
                            
                            labels = [[-100] * len(ctx) if c < len(contexts)-1 else [-100] + ctx[1:] for c, ctx in enumerate(contexts)]
                            assert labels[-1][1:] == contexts[-1][1:]
                            
                            labels = [-100] + list(chain.from_iterable(labels)) + [config.eos_id]
                            assert len(input_ids) == len(labels)
                            
                            self.input_ids.append(input_ids)
                            self.token_type_ids.append(token_type_ids)
                            self.labels.append(labels)
                            break
                            
    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, idx):
        return self.input_ids[idx], self.token_type_ids[idx], self.labels[idx]


class PadCollate():
    def __init__(self, eos_id):
        self.eos_id = eos_id
        
    def pad_collate(self, batch):
        input_ids, token_type_ids, labels =[], [], []
        for idx, seqs in enumerate(batch):
            input_ids.append(torch.LongTensor(seqs[0]))
            token_type_ids.append(torch.LongTensor(seqs[1]))
            labels.append(torch.LongTensor(seqs[2]))
            
        input_ids = torch.nn.utils.rnn.pad_sequence(
          input_ids, batch_first=True, padding_value=self.eos_id
        )
        token_type_ids = torch.nn.utils.rnn.pad_sequence(
          token_type_ids, batch_first=True, padding_value=self.eos_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
          labels, batch_first=True, padding_value=-100
        )
    
        return input_ids, token_type_ids, labels
        

def validation(model, valid_loader):
    print("Validation processing...")
    model.eval()

    valid_losses = []
    valid_ppls = []
    with torch.no_grad():
        for i, batch in enumerate(tqdm(valid_loader)):
            input_ids, token_type_ids, labels = batch
            input_ids, token_type_ids, labels = \
                input_ids.to(device), token_type_ids.to(device), labels.to(device)

            outputs = model(
                input_ids=input_ids,
                token_type_ids = token_type_ids,
                labels = labels
            )

            loss, logits = outputs[0], outputs[1]

            valid_losses.append(loss.detach())
            ppl = torch.exp(loss.detach())
            valid_ppls.append(ppl)

        valid_losses = [loss.item() for loss in valid_losses]
        valid_ppls = [ppl.item() if not math.isinf(ppl.item()) else 1e+8 for ppl in valid_ppls]
        valid_loss = np.mean(valid_losses)
        valid_ppl = np.mean(valid_ppls)

        if math.isnan(valid_ppl):
            valid_ppl = 1e+8

    return valid_loss, valid_ppl


def main():
    # WandB – Initialize a new run
    wandb.init(project="TOC-distilgpt")

    # WandB – Config is a variable that holds and saves hyperparameters and inputs
    # Defining some key variables that will be used later on in the training  
    config = wandb.config          # Initialize config
    config.TRAIN_BATCH_SIZE = 8   # input batch size for training (default: 64)
    config.VALID_BATCH_SIZE = 8    # input batch size for testing (default: 1000)
    config.TRAIN_EPOCHS = 10      # number of epochs to train (default: 10)
    config.VAL_EPOCHS = 1 
    config.LEARNING_RATE = 2e-5    # learning rate (default: 2e-5)
    config.SEED = 42               # random seed (default: 42)
    config.MAX_LEN = 1024
    config.OUTPUT_LEN = 200 
    config.warmup_ratio = 0.1
    config.ckpt_dir = './distilgpt2-ckpt/'

    
    # Set random seeds and deterministic pytorch for reproducibility
    torch.manual_seed(config.SEED) # pytorch random seed
    np.random.seed(config.SEED) # numpy random seed
    torch.backends.cudnn.deterministic = True

    random.seed(config.SEED)
    
    # Tokenizer & Vocab
    print("Loading the tokenizer...")
    tokenizer = GPT2Tokenizer.from_pretrained('distilgpt2')
    config.eos_token = tokenizer.eos_token
    config.bos_token = tokenizer.bos_token
    vocab = tokenizer.get_vocab()
    config.vocab_size = len(vocab)
    config.bos_id = vocab[config.bos_token]
    config.eos_id = vocab[config.eos_token]
    model = GPT2LMHeadModel.from_pretrained('distilgpt2')
    model = model.to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)
    model.resize_token_embeddings(config.vocab_size)
    # load from checkpoint
    load_ckpt = False
    if load_ckpt:
        checkpoint = torch.load(config.ckpt_dir + 'best_ckpt_epoch=5_valid_loss=2.4685.ckpt')
        model.load_state_dict(checkpoint['model_state_dict'])
        optim.load_state_dict(checkpoint['optim_state_dict'])
    
    
    print("loading data... ")
    data_categories = ['art', 'car', 'computer', 'education', 'family', 'finance', 'food', 'health', 'hobby', 'holiday', 'home', 'pet', 'philosophy', 'relationship', 'sport', 'style', 'travel', 'work', 'youth']
    dialogues = []
    for category in data_categories:
        data = json.load(open(f'INST2DIAL-Auto/{category}_multiwoz_convqa_flan_t5_large_2qa_3sent.json'))
        keys = list(data.keys())
        for key in keys:
            dialogues.append(data[key])

    train, valid = train_test_split(dialogues, test_size=0.01, random_state=config.SEED)
    train_dataset = CustomDataset(train, tokenizer, config)
    valid_dataset = CustomDataset(valid, tokenizer, config)
    print("train: ", len(train_dataset))
    print("valid: ", len(valid_dataset))
    
    ppd = PadCollate(eos_id=config.eos_id)

    train_loader = DataLoader(train_dataset, 
                              collate_fn=ppd.pad_collate,
                              shuffle=True, 
                              batch_size=config.TRAIN_BATCH_SIZE,  
                              pin_memory=True)
    
    valid_loader = DataLoader(valid_dataset,
                              collate_fn=ppd.pad_collate,
                              batch_size=config.VALID_BATCH_SIZE, 
                              pin_memory=True)
    
    if not os.path.exists(config.ckpt_dir):
        os.makedirs(config.ckpt_dir)
    
    # Calculate total training steps
    num_batches = len(train_loader)
    config.total_train_steps = config.TRAIN_EPOCHS * num_batches
    config.warmup_steps = int(config.warmup_ratio * config.total_train_steps)

    sched = get_polynomial_decay_schedule_with_warmup(
        optim,
        num_warmup_steps=config.warmup_steps,
        num_training_steps=config.total_train_steps,
        power=2
    )
    if load_ckpt:
        sched.load_state_dict(checkpoint['sched_state_dict'])
    
    writer = SummaryWriter()
    
    print("Setting finished.")
    
    print("Training starts.")
    best_loss = sys.float_info.max
    last_epoch = 0
    start_epoch = 1
    for epoch in range(start_epoch, start_epoch + config.TRAIN_EPOCHS):
        model.train()
        print(f"#"*50 + f"Epoch: {epoch}" + "#"*50)
        train_losses = []
        train_ppls = []
        for i, batch in enumerate(tqdm(train_loader)):
            input_ids, token_type_ids, labels = batch
            input_ids, token_type_ids, labels = \
                input_ids.to(device), token_type_ids.to(device), labels.to(device)
            
            outputs = model(
                input_ids=input_ids,
                token_type_ids=token_type_ids,
                labels=labels
            )

            loss, logits = outputs[0], outputs[1]
            
            optim.zero_grad()
            loss.backward()
            optim.step()
            sched.step()

            train_losses.append(loss.detach())
            ppl = torch.exp(loss.detach())
            train_ppls.append(ppl)

        train_losses = [loss.item() for loss in train_losses]
        train_ppls = [ppl.item() if not math.isinf(ppl.item()) else 1e+8 for ppl in train_ppls]
        train_loss = np.mean(train_losses)
        train_ppl = np.mean(train_ppls)
        print(f"Train loss: {train_loss} || Train perplexity: {train_ppl}")

        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("PPL/train", train_ppl, epoch)

        last_epoch += 1

        valid_loss, valid_ppl = validation(model, valid_loader)

        if valid_loss < best_loss:
            best_loss = valid_loss
            state_dict = {
                'model_state_dict': model.state_dict(),
                'optim_state_dict': optim.state_dict(),
                'sched_state_dict': sched.state_dict(),
                'loss': best_loss,
                'epoch': last_epoch
            }

            torch.save(state_dict, f"{config.ckpt_dir}/best_ckpt_epoch={epoch}_valid_loss={round(best_loss, 4)}.ckpt")
            print("*"*10 + "Current best checkpoint is saved." + "*"*10)
            print(f"{config.ckpt_dir}/best_ckpt_epoch={epoch}_valid_loss={round(best_loss, 4)}.ckpt")

        print(f"Best valid loss: {best_loss}")
        print(f"Valid loss: {valid_loss} || Valid perplexity: {valid_ppl}")

        writer.add_scalar("Loss/valid", valid_loss, epoch)
        writer.add_scalar("PPL/valid", valid_ppl, epoch)

        writer.add_scalars("Losses", {
            'train': train_loss, 
            'valid': valid_loss,
        }, epoch)
        writer.add_scalars("PPLs", {
            'train': train_ppl,
            'valid': valid_ppl,
        }, epoch)

if __name__ == '__main__':
    main()