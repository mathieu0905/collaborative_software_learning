from __future__ import absolute_import, division, print_function

import argparse
import glob
import logging
import os
import pickle
import random
import re
import shutil
import json
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler,TensorDataset
from torch.utils.data.distributed import DistributedSampler

try:
    from torch.utils.tensorboard import SummaryWriter
except:
    from tensorboardX import SummaryWriter

from tqdm import tqdm, trange
import multiprocessing
from model import Model

import flwr as fl
import torch
import torch.nn as nn
import torch.optim as optim
from collections import OrderedDict


cpu_cont = 16
DEVICE = torch.device("cuda" if torch.cuda.is_available()  else "cpu")
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                          BertConfig, BertForMaskedLM, BertTokenizer,
                          GPT2Config, GPT2LMHeadModel, GPT2Tokenizer,
                          OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer,
                          RobertaConfig, RobertaModel, RobertaTokenizer,
                          DistilBertConfig, DistilBertForMaskedLM, DistilBertTokenizer)

logging.basicConfig(filename='training_uneven_tags.log', level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)

MODEL_CLASSES = {
    'gpt2': (GPT2Config, GPT2LMHeadModel, GPT2Tokenizer),
    'openai-gpt': (OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer),
    'bert': (BertConfig, BertForMaskedLM, BertTokenizer),
    'roberta': (RobertaConfig, RobertaModel, RobertaTokenizer),
    'distilbert': (DistilBertConfig, DistilBertForMaskedLM, DistilBertTokenizer)
}

def parse_args():
    parser = argparse.ArgumentParser()

    # 添加所有的参数定义
    ## Required parameters
    parser.add_argument("--train_data_dir", default=None, type=str, required=True,
                        help="The input training data file (a text file).")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--eval_data_file", default=None, type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")
    parser.add_argument("--evaluate_during_training", default=True, type=bool)
    parser.add_argument("--test_data_file", default=None, type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")
                    
    parser.add_argument("--model_type", default="bert", type=str,
                        help="The model architecture to be fine-tuned.")
    parser.add_argument("--model_name_or_path", default=None, type=str,
                        help="The model checkpoint for weights initialization.")

    parser.add_argument("--mlm", action='store_true',
                        help="Train with masked-language modeling loss instead of language modeling.")
    parser.add_argument("--mlm_probability", type=float, default=0.15,
                        help="Ratio of tokens to mask for masked language modeling loss")

    parser.add_argument("--config_name", default="", type=str,
                        help="Optional pretrained config name or path if not the same as model_name_or_path")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Optional directory to store the pre-trained models downloaded from s3 (instread of the default one)")
    parser.add_argument("--block_size", default=-1, type=int,
                        help="Optional input sequence length after tokenization."
                             "The training dataset will be truncated in block of this size for training."
                             "Default to the model max input length for single sentence inputs (take into account special tokens).")
    # parser.add_argument("--do_train", action='store_true',
    #                     help="Whether to run training.")
    # parser.add_argument("--do_eval", action='store_true',
    #                     help="Whether to run eval on the dev set.")
    # parser.add_argument("--do_test", action='store_true',
    #                     help="Whether to run eval on the dev set.")    
    # parser.add_argument("--evaluate_during_training", action='store_true',
    #                     help="Run evaluation during training at each logging step.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")

    parser.add_argument("--train_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=1.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")

    parser.add_argument('--logging_steps', type=int, default=50,
                        help="Log every X updates steps.")
    # parser.add_argument('--save_steps', type=int, default=50,
    #                     help="Save checkpoint every X updates steps.")
    # parser.add_argument('--save_total_limit', type=int, default=None,
    #                     help='Limit the total amount of checkpoints, delete the older checkpoints in the output_dir, does not delete by default')
    parser.add_argument("--eval_all_checkpoints", action='store_true',
                        help="Evaluate all checkpoints starting with the same prefix as model_name_or_path ending and ending with step number")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--overwrite_cache', action='store_true',
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--epoch', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument('--server_ip', type=str, default='', help="For distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="For distant debugging.")

    return parser.parse_args()

def get_example(item):
    url1,url2,label,tokenizer,args,cache,url_to_code=item
    if url1 in cache:
        code1=cache[url1].copy()
    else:
        try:
            code=' '.join(url_to_code[url1].split())
        except:
            code=""
        code1=tokenizer.tokenize(code)
    if url2 in cache:
        code2=cache[url2].copy()
    else:
        try:
            code=' '.join(url_to_code[url2].split())
        except:
            code=""
        code2=tokenizer.tokenize(code)
        
    return convert_examples_to_features(code1,code2,label,url1,url2,tokenizer,args,cache)

class InputFeatures(object):
    """A single training/test features for a example."""
    def __init__(self,
                 input_tokens,
                 input_ids,
                 label,
                 url1,
                 url2

    ):
        self.input_tokens = input_tokens
        self.input_ids = input_ids
        self.label=label
        self.url1=url1
        self.url2=url2

def convert_examples_to_features(code1_tokens,code2_tokens,label,url1,url2,tokenizer,args,cache):
    #source
    code1_tokens=code1_tokens[:args.block_size-2]
    code1_tokens =[tokenizer.cls_token]+code1_tokens+[tokenizer.sep_token]
    code2_tokens=code2_tokens[:args.block_size-2]
    code2_tokens =[tokenizer.cls_token]+code2_tokens+[tokenizer.sep_token]  
    
    code1_ids=tokenizer.convert_tokens_to_ids(code1_tokens)
    padding_length = args.block_size - len(code1_ids)
    code1_ids+=[tokenizer.pad_token_id]*padding_length
    
    code2_ids=tokenizer.convert_tokens_to_ids(code2_tokens)
    padding_length = args.block_size - len(code2_ids)
    code2_ids+=[tokenizer.pad_token_id]*padding_length
    
    source_tokens=code1_tokens+code2_tokens
    source_ids=code1_ids+code2_ids
    return InputFeatures(source_tokens,source_ids,label,url1,url2)

class TextDataset(Dataset):
    def __init__(self, tokenizer, args, file_path='train', block_size=512,pool=None):
        postfix=file_path.split('/')[-1].split('.txt')[0]
        self.examples = []
        index_filename=file_path
        logger.info("Creating features from index file at %s ", index_filename)
        url_to_code={}
        with open('/'.join(index_filename.split('/')[:-1])+'/data.jsonl') as f:
            for line in f:
                line=line.strip()
                js=json.loads(line)
                url_to_code[js['idx']]=js['func']

        data=[]
        cache={}
        f=open(index_filename)
        with open(index_filename) as f:
            for line in f:
                line=line.strip()
                url1,url2,label=line.split('\t')
                if url1 not in url_to_code or url2 not in url_to_code:
                    continue
                if label=='0':
                    label=0
                else:
                    label=1
                data.append((url1,url2,label,tokenizer, args,cache,url_to_code))
        
        data=random.sample(data,int(len(data)*0.1))

        self.examples=pool.map(get_example,tqdm(data,total=len(data)))
        if 'split' in postfix:
            for idx, example in enumerate(self.examples[:3]):
                    logger.info("*** Example ***")
                    logger.info("idx: {}".format(idx))
                    logger.info("label: {}".format(example.label))
                    logger.info("input_tokens: {}".format([x.replace('\u0120','_') for x in example.input_tokens]))
                    logger.info("input_ids: {}".format(' '.join(map(str, example.input_ids))))



    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        
        return torch.tensor(self.examples[item].input_ids),torch.tensor(self.examples[item].label)
    
def get_config_and_tokenizer(args):
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path,
                                          cache_dir=args.cache_dir if args.cache_dir else None)
    config.num_labels=2
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name,
                                                do_lower_case=args.do_lower_case,
                                                cache_dir=args.cache_dir if args.cache_dir else None)
    return config, tokenizer

def load_and_cache_examples(args, evaluate=False,test=False,pool=None):
    config, tokenizer = get_config_and_tokenizer(args)
    dataset = TextDataset(tokenizer, args, file_path=args.test_data_file if test else (args.eval_data_file if evaluate else args.train_data_file),block_size=args.block_size,pool=pool)
    return dataset

def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def train(args, model, train_dataset, eval_dataset, client_id):
    """ Train the model """

    logging.basicConfig(filename='training.log', level=logging.INFO)

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)
    args.max_steps=args.epoch*len( train_dataloader)
    args.save_steps=len( train_dataloader)
    args.warmup_steps=len( train_dataloader)
    args.logging_steps=len( train_dataloader)
    args.num_train_epochs=args.epoch
    model.to(args.device)

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                num_training_steps=args.max_steps)
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # # multi-gpu training (should be after apex fp16 initialization)
    # if args.n_gpu > 1:
    #     model = torch.nn.DataParallel(model)

    # # Distributed training (should be after apex fp16 initialization)
    # if args.local_rank != -1:
    #     model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
    #                                                       output_device=args.local_rank,
    #                                                       find_unused_parameters=True)

    checkpoint_last = os.path.join(args.output_dir, 'checkpoint-last')
    scheduler_last = os.path.join(checkpoint_last, 'scheduler.pt')
    optimizer_last = os.path.join(checkpoint_last, 'optimizer.pt')
    if os.path.exists(scheduler_last):
        scheduler.load_state_dict(torch.load(scheduler_last))
    if os.path.exists(optimizer_last):
        optimizer.load_state_dict(torch.load(optimizer_last))
    # Train!
    logging.info("***** Running training *****")
    logging.info("  Num examples = %d", len(train_dataset))
    logging.info("  Num Epochs = %d", args.num_train_epochs)
    logging.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logging.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
              args.train_batch_size * args.gradient_accumulation_steps * (
                  torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logging.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logging.info("  Total optimization steps = %d", args.max_steps)
    
    global_step = args.start_step
    tr_loss, logging_loss,avg_loss,tr_nb,tr_num,train_loss = 0.0, 0.0,0.0,0,0,0
    best_mrr=0.0
    best_f1=0
    # model.resize_token_embeddings(len(tokenizer))
    model.zero_grad()
    set_seed(args.seed)  # Added here for reproducibility (even between python 2 and 3)
 
    for idx in range(args.start_epoch, int(args.num_train_epochs)): 
        logging.info(f"Starting epoch {idx}")
        bar = tqdm(train_dataloader,total=len(train_dataloader))
        tr_num=0
        train_loss=0
        for step, batch in enumerate(bar):
            inputs = batch[0].to(args.device)        
            labels=batch[1].to(args.device) 
            model.train()
            loss,logits = model(inputs,labels)


            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            tr_loss += loss.item()
            tr_num+=1
            train_loss+=loss.item()
            if avg_loss==0:
                avg_loss=tr_loss
            avg_loss=round(train_loss/tr_num,5)
            bar.set_description("epoch {} loss {}".format(idx,avg_loss))

                
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()  
                global_step += 1
                output_flag=True
                avg_loss=round(np.exp((tr_loss - logging_loss) /(global_step- tr_nb)),4)
                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    logging_loss = tr_loss
                    tr_nb=global_step

                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    
                    if args.local_rank == -1 and args.evaluate_during_training:  # Only evaluate when single GPU otherwise metrics may not average well
                        eval_loss, results = evaluate(args, model, eval_dataset)                 
                        # Save model checkpoint

                    if results['f1']>best_f1:
                        best_f1=results['f1']
                        logging.info("  "+"*"*20)  
                        logging.info("  Best f1:%s",round(best_f1,4))
                        logging.info("  "+"*"*20)                          
                        
                        checkpoint_prefix = 'checkpoint-best-f1'
                        output_dir = f'{args.output_dir}/{checkpoint_prefix}/client_{client_id}'
                        # output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix), )                        
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)                        
                        model_to_save = model.module if hasattr(model,'module') else model
                        # output_dir = os.path.join(output_dir, '{}'.format('model.bin')) 
                        # torch.save(model_to_save.state_dict(), output_dir)
                        logging.info("Saving model checkpoint to %s", output_dir)
        
        logging.info(f"Finished epoch {idx}")                
        if args.max_steps > 0 and global_step > args.max_steps:
            # train_iterator.close()
            break

        # model_save_path = os.path.join(args.output_dir, f"model_client_{client_id}_epoch_{idx}.bin")
        # torch.save(model.state_dict(), model_save_path)
        # logging.info(f"Saved model parameters to {model_save_path}")
    #return global_step, tr_loss / global_step

def evaluate(args, model, eval_dataset, prefix="",eval_when_training=False):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_output_dir = args.output_dir
    #eval_dataset = load_and_cache_examples(args, tokenizer, evaluate=True,pool=pool)
    if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(eval_output_dir)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size,num_workers=4,pin_memory=True)

    # multi-gpu evaluate
    if args.n_gpu > 1 and eval_when_training is False:
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()
    logits=[]  
    y_trues=[]
    for batch in eval_dataloader:
        inputs = batch[0].to(args.device)        
        labels=batch[1].to(args.device) 
        with torch.no_grad():
            lm_loss,logit = model(inputs,labels)
            eval_loss += lm_loss.mean().item()
            logits.append(logit.cpu().numpy())
            y_trues.append(labels.cpu().numpy())
        nb_eval_steps += 1
    logits=np.concatenate(logits,0)
    y_trues=np.concatenate(y_trues,0)
    best_threshold=0
    best_f1=0
    for i in range(1,100):
        threshold=i/100
        y_preds=logits[:,1]>threshold
        from sklearn.metrics import recall_score
        recall=recall_score(y_trues, y_preds)
        from sklearn.metrics import precision_score
        precision=precision_score(y_trues, y_preds)   
        from sklearn.metrics import f1_score
        f1=f1_score(y_trues, y_preds) 
        if f1>best_f1:
            best_f1=f1
            best_threshold=threshold

    y_preds=logits[:,1]>best_threshold
    from sklearn.metrics import recall_score
    recall=recall_score(y_trues, y_preds)
    from sklearn.metrics import precision_score
    precision=precision_score(y_trues, y_preds)   
    from sklearn.metrics import f1_score
    f1=f1_score(y_trues, y_preds)             
    # result = {
    #     "eval_recall": float(recall),
    #     "eval_precision": float(precision),
    #     "eval_f1": float(f1),
    #     "eval_threshold":best_threshold,
        
    # }
    
    result = {
        "recall": float(recall),
        "precision": float(precision),
        "f1": float(f1),
        "threshold":best_threshold,
        
    }

    logger.info("***** Eval results {} *****".format(prefix))
    for key in sorted(result.keys()):
        logger.info("  %s = %s", key, str(round(result[key],4)))

    return eval_loss, result

def load_data(cid, args):
    pool = multiprocessing.Pool(cpu_cont)
    args.train_data_file = f'{args.train_data_dir}/split_{cid}.txt'
    trainset = load_and_cache_examples(args, evaluate=False,pool=pool)
    testset = load_and_cache_examples(args, evaluate=True,pool=pool)
    return trainset, testset

def load_model(args):
    args.start_epoch = 0
    args.start_step = 0
    checkpoint_last = os.path.join(args.output_dir, 'checkpoint-last')
    if os.path.exists(checkpoint_last) and os.listdir(checkpoint_last):
        args.model_name_or_path = os.path.join(checkpoint_last, 'pytorch_model.bin')
        args.config_name = os.path.join(checkpoint_last, 'config.json')
        idx_file = os.path.join(checkpoint_last, 'idx_file.txt')
        with open(idx_file, encoding='utf-8') as idxf:
            args.start_epoch = int(idxf.readlines()[0].strip()) + 1

        step_file = os.path.join(checkpoint_last, 'step_file.txt')
        if os.path.exists(step_file):
            with open(step_file, encoding='utf-8') as stepf:
                args.start_step = int(stepf.readlines()[0].strip())

        logger.info("reload model from {}, resume from {} epoch".format(checkpoint_last, args.start_epoch))

    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path,
                                          cache_dir=args.cache_dir if args.cache_dir else None)
    config.num_labels=2
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name,
                                                do_lower_case=args.do_lower_case,
                                                cache_dir=args.cache_dir if args.cache_dir else None)
    if args.block_size <= 0:
        args.block_size = tokenizer.max_len_single_sentence  # Our input block size will be the max possible for the model
    args.block_size = min(args.block_size, tokenizer.max_len_single_sentence)
    if args.model_name_or_path:
        model = model_class.from_pretrained(args.model_name_or_path,
                                            from_tf=bool('.ckpt' in args.model_name_or_path),
                                            config=config,
                                            cache_dir=args.cache_dir if args.cache_dir else None)    
    else:
        model = model_class(config)

    model = Model(model,config,tokenizer,args).to(DEVICE)

    return model, args