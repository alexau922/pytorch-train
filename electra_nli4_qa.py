import warnings
warnings.filterwarnings("ignore")

import json
from progressbar import *
from transformers import (ElectraPreTrainedModel, ElectraModel,ElectraConfig)

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import OrderedDict
from progressbar import *
import time
import datetime
import sys

from torch.utils.data import Dataset, TensorDataset, DataLoader

from cantokenizer import CanTokenizer
from transformers import TrainingArguments
import random

'''
PYTHON=/opt/conda/bin/python3
TOTAL_NUM_UPDATES=2
WARMUP_UPDATES=0.1 
UPDATE_FREQ=1 
BATCH_SIZE=2 # per gpu 
SEQ_LENGTH=96 
LR=1e-4 
MODEL=electra_nli4_orig
MODEL_SIZE=base 
VOCAB_FILE=cantokenizer-vocab.txt 
DATA_DIR=./ 
SAVE_DIR=$MODEL"_"$MODEL_SIZE"_"$BATCH_SIZE"_"$SEQ_LENGTH
TOKENIZERS_PARALLELISM=false $PYTHON train.py \
  $DATA_DIR \
  --eval-dir $DATA_DIR \
  --model $MODEL \
  --restore-file pt_model_d.pt \
  --save-dir "$SAVE_DIR" \
  --log-interval 5  --weight-decay 0.01 \
  --seq-length $SEQ_LENGTH \
  --lr $LR \
  --gpus 8 \
  --warmup-updates $WARMUP_UPDATES \
  --total-num-update $TOTAL_NUM_UPDATES \
  --batch-size $BATCH_SIZE \
  --num-workers=0 \
  --model-size=$MODEL_SIZE \
  --vocab-file=$VOCAB_FILE \
  --shuffle



'''
'''
import json
from cantokenizer import CanTokenizer
tokenizer = CanTokenizer('cantokenizer-vocab.txt', 
                         add_special=True, 
                         add_special_cls='<s>', 
                         add_special_sep='</s>'
                        )
max_seq_length = 350
with open('yuenli-train.json') as f:
    train = json.load(f)
convert([e for e in train if e[0] and e[1]], tokenizer, 350, 'train')
with open('yuenli-test.json') as f:
    test = json.load(f)
convert([e for e in test if e[0] and e[1]], tokenizer, 350, 'test')
'''

def convert(data,tokenizer, max_seq_length, prefix):
    '''
    Convert the data and encoding the batch, and then shuffle the data and finally return the file
    
    Args:e
        data (list of dictionaries): list of dictionaries of tokenize_data and answer_info
        tokenizer (cantokenizer.CanTokenizer object): tokenizer that is used for encoding the text
        max_seq_length (maximum sequence's length): maximum length of the tokenized sequence
        prefix (string): labelling the data classes into training class and testing classes
    '''
    
    # Define the progress bar
    widgets = ['Progress: ',Percentage(), ' ', Bar('#'),' ', Timer(),
               ' ', ETA(), ' ', FileTransferSpeed()]
    pbar = ProgressBar(widgets=widgets, maxval=10*len(data)).start()
    
    # Check if the variable data is in the type of list
    if isinstance(data,list) == False:
        raise TypeError(f'Unexpected type of variable "data". Expect type list.')
    # Encode the data in the list of dictionaries "data"
    encodeds = tokenizer.encode_batch([data_obj['tokenize_data'] for data_obj in data])
    
    
    # Initializing empty lists for holding different data
    # data_original stores the original data
    data_original = []
    # data_attn_mask stores the attention mask of the encoded data
    data_attn_mask = []
    # data_answerable stores whether the data is answerable
    data_answerable = []
    # data_start_pos stores the starting position of the answer
    data_start_pos = []
    # data_end_pos stores the end position of the answer
    data_end_pos = []
    # data_type_ids stores the info of type_ids in the encoded data
    data_type_ids = []
    
    counter = 0
    # For every encoded data (encodeds) and original data (data)
    for i,(e,t) in enumerate(zip(encodeds,data)):
        
        # Now e is the encoded value, t is th training/testing data set
        # Get the ground-truth label (answer_info)
        label_dict = t['answer_info']
        # If the length of encoded id (a list) is longer than the maximum sequence length, skip processing the data
        if len(e.ids)>max_seq_length:
            continue
        e.pad(max_seq_length)
        # Append the data into list to store the values
        data_original.append(e.ids)
        data_attn_mask.append(e.attention_mask)
        data_answerable.append(label_dict['answerable'])
        if label_dict['start_pos'] is not None :
            data_start_pos.append(label_dict['start_pos'])
        else:
            data_start_pos.append(0)
        if label_dict['end_pos'] is not None :
            data_end_pos.append(label_dict['end_pos'])
        else:
            data_end_pos.append(0)
        data_type_ids.append(e.type_ids)
    
        # update the progress bar
        pbar.update(10 * i + 1)
    
    indices = list(range(len(data_original)))
    # Check if the indices list is empty or not
    if indices == []:
        raise ValueError('Error! There is no encoded entity.')
        
    # Randomly shuffle the indices to prevent similar-data aggregation leading to overfitting 
    random.shuffle(indices)
    
    # Overriding the original list using the shuffled list
    data_original = [data_original[i] for i in indices]
    data_attn_mask = [data_attn_mask[i] for i in indices]
    data_answerable = [data_answerable[i] for i in indices]
    data_start_pos = [data_start_pos[i] for i in indices]
    data_end_pos = [data_end_pos[i] for i in indices]
    data_type_ids = [data_type_ids[i] for i in indices]
    
    # Cast the type of the data so that the calculation will be quicker
    ids = np.array(data_original).astype(np.int16)
    attn = np.array(data_attn_mask).astype(np.int8)
    answerable = np.array(data_answerable).astype(np.float32)
    start_pos = np.array(data_start_pos).astype(np.int8)
    end_pos = np.array(data_end_pos).astype(np.int8)
    type_ids = np.array(data_type_ids).astype(np.int8)
    
    
    pbar.finish()
    
    # Write the file into binary format
    with open(prefix+"_ids",'wb') as f:
        f.write(ids.tobytes())
    with open(prefix+'_mask','wb') as f:
        f.write(attn.tobytes())
    with open(prefix+"_type_ids", 'wb') as f:
        f.write(type_ids.tobytes())
    with open(prefix+"_answerable",'wb')as f:
        f.write(answerable.tobytes())
    with open(prefix+"_start_pos",'wb') as f:
        f.write(start_pos)
    with open(prefix+"_end_pos",'wb') as f:
        f.write(end_pos)
#####################################
## 
##             Modelling
## 
#####################################



from transformers import (
    ElectraForSequenceClassification as ElectraForSequenceClassification_, 
    ElectraModel,
    ElectraPreTrainedModel,ElectraConfig
)
from torch.nn import CrossEntropyLoss, MSELoss
from torch import nn

def gelu_fast(x):
    return 0.5 * x * (1.0 + torch.tanh(x * 0.7978845608 * (1.0 + 0.044715 * x * x)))

class ElectraClassificationHead(nn.Module):
    def __init__(self,config,num_outputs):
        '''
        Initialize the classification head for the electra model (ELECTRA model does not have functional
        head)
        '''
        # Initialize the nn.Module class
        super().__init__()
        # randomly zeroes some of the elements of the input tensor with probability config.hidden_dropout_prob 
        # using samples from a Bernoulli distribution. Each channel will be zeroed out independently on each 
        # forward cell
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # change the hidden size (768) of each token to the number of output
        self.out_proj = nn.Linear(config.hidden_size, num_outputs)
    
    '''
    In nn.Module the forward function is defined in the __call__ method, so when executing model(*args) the 
    "forward" method will automatically used.
    
    Example:
        model = LeNet()
        y = model(x) # Same as using model.forward(x)
    '''
    # Forward Propagation
    def forward(self,features, **kwargs):
        x = features[:,0,:] # take <s> token (equiv. to [CLS])
        # Randomly drop out some cells and return to 0
        x = self.dropout(x)
        # Change the last dimension to 1
        x = self.out_proj(x)
        return x

class ElectraForSquad(ElectraPreTrainedModel):
    def __init__(self,config):
        # initialize the pretrained model with the pretrained config
        ElectraPreTrainedModel.__init__(self,config)
        # Set our config to be the config fetched from internet
        self.config = config
        # Initialize the model with the config that can automatically run the forward method calling itself
        self.electra = ElectraModel(config)
        # Expand the logits to start logits and end logits, map out the linear hidden size to 2 outputs
        self.span_logits = nn.Linear(config.hidden_size,2)
        # Initialize a classification head that it can automatically run the forward method calling itself
        self.classification = ElectraClassificationHead(config,1)
        self.init_weights()
    
    def forward(self,input_ids = None, attn_mask = None, type_ids = None, start_positions = None, 
                end_positions = None, is_impossible = None,return_dict = False):
        
        # Run the ElectraModel forward function by passing the input_ids, attn_mask and type_ids
        discriminator_hidden_states = self.electra(input_ids,attn_mask,type_ids)
        # Get the first dimension in the output tensor as the sequence output
        sequence_output = discriminator_hidden_states[0]
        # We can then split the start and end logits using the split function on the last dimension
        start_logits, end_logits = self.span_logits(sequence_output).split(1,dim = -1)
        # We can also know the answerability tensor using the split function
        answerability = self.classification(sequence_output)
        
        try:
          # reduce the last dimension of 1
          start_logits = start_logits.squeeze(-1)
          end_logits = end_logits.squeeze(-1)
          answerability = answerability.squeeze(-1)
          total_loss = 0
        except Except as e:
          exception_type, exception_object, exception_traceback = sys.exc_info()
          filename = exception_traceback.tb_frame.f_code.co_filename
          line_number = exception_traceback.tb_lineno
          print("Exception type: ", exception_type)
          print("File name: ", filename)
          print("Line number: ", line_number)
        
        if start_positions is not None and end_positions is not None:
            # Initialize a cross entropy loss function to enable calculation of cross entropy loss afterwards
            # reduction has 3 values
            # 1. elementwise_mean: get the mean of n samples' loss and return
            # 2. sum: get the sum of loss of n samples and return
            # 3. none: calculate the n losses 
            loss_fct = nn.CrossEntropyLoss()
            # Do cross entropyloss
            start_loss = loss_fct(start_logits,start_positions)
            end_loss = loss_fct(end_logits,end_positions)
            # Calculate the total loss, the average of cross entropy loss of the start position and end position
            total_loss = (start_loss+end_loss)/2
        
        cls_losses = []
        
        if is_impossible is not None:
            loss_fct_cls = nn.BCEWithLogitsLoss()
            print('answerability_loss = ',answerability)
            print('is_impossible = ',is_impossible)
            # Type cast the tensor to float type
            is_impossible = is_impossible.type(torch.float16)
            print('is_impossible.dtype = ',is_impossible.dtype)
            # Use sigmoid to normalize the value and use BCE loss
            answerability_loss = loss_fct_cls(answerability,is_impossible)
            print('answerability_loss = ',answerability_loss)
            cls_losses.append(answerability_loss)
        
        # if we have the [CLS] tag loss
        if len(cls_losses) >0:
            # Add all the values in the cls_losses
            total_loss += torch.stack(cls_losses, dim = 0).sum()
        
        output = (start_logits,end_logits,answerability)
        return ((total_loss,)+output) if total_loss !=0 else output
      
#####################################
## 
##            Data Utils
## 
#####################################

import numpy as np
import os
class textDataset(Dataset):
    def __init__(self, 
                 data_path, 
                 seq_length,
                 batch_size, 
                 eval=False, 
                 eval_num_samples=0):
        self.seq_length = seq_length
        self.batch_size = batch_size
        self.data_path = data_path
        
        self.eval_num_samples = (eval_num_samples // batch_size)
        self.eval = eval

        prefix = 'test' if self.eval else 'train'
        self.length = os.stat(data_path+prefix+"_ids").st_size//(seq_length*2) // batch_size

        self.ids_bin_buffer = None
        self.dataset = None

    def __len__(self):
        return self.length
    
    def __getitem__(self, i):
        if i >= self.length:
            raise StopIteration
        return self.__getbatch__(i*self.batch_size, self.batch_size)

    def __getbatch__(self, i, size):
        seq_length = self.seq_length
        if self.ids_bin_buffer is None:
            data_path = self.data_path
            prefix = 'test' if self.eval else 'train'

            path = data_path+prefix+"_ids" 
            self.ids_bin_buffer_mmap = np.memmap(path, mode='r', order='C')
            self.ids_bin_buffer = self.ids_bin_buffer_mmap

            path = data_path+prefix+"_mask"
            self.attention_mask_bin_buffer_mmap = np.memmap(path, mode='r', order='C')
            self.attention_mask_bin_buffer = self.attention_mask_bin_buffer_mmap 

            path = data_path+prefix+"_type_ids"
            self.type_ids_bin_buffer_mmap = np.memmap(path, mode='r', order='C')
            self.type_ids_bin_buffer = self.type_ids_bin_buffer_mmap 
            
            path = data_path+prefix+"_start_pos"
            self.start_pos_bin_buffer_mmap = np.memmap(path, mode='r', order='C')
            self.start_pos_bin_buffer = self.start_pos_bin_buffer_mmap
             
            path = data_path+prefix+"_end_pos"
            self.end_pos_bin_buffer_mmap = np.memmap(path, mode='r', order='C')
            self.end_pos_bin_buffer = self.end_pos_bin_buffer_mmap
             
            path = data_path+prefix+"_answerable"
            self.answerable_bin_buffer_mmap = np.memmap(path, mode='r', order='C')
            self.answerable_bin_buffer = self.answerable_bin_buffer_mmap
        
        start = seq_length*i*2
        shape = (size,self.seq_length)
        
        ids_buffer = np.frombuffer(self.ids_bin_buffer, dtype=np.int16, count=seq_length*size, offset=start).reshape(shape)
        attention_mask_buffer = np.frombuffer(self.attention_mask_bin_buffer, dtype=np.int8, count=seq_length*size, offset=start // 2).reshape(shape)
        type_ids_buffer = np.frombuffer(self.type_ids_bin_buffer, dtype=np.int8, count=seq_length*size, offset=start // 2).reshape(shape)
        answerable_buffer = np.frombuffer(self.answerable_bin_buffer, dtype=np.float32, count=size, offset=i).reshape((size))
        start_pos_buffer = np.frombuffer(self.start_pos_bin_buffer, dtype=np.int8, count=size, offset=i).reshape((size))
        end_pos_buffer = np.frombuffer(self.end_pos_bin_buffer, dtype=np.int8, count=size, offset=i).reshape((size))
        
        return (
            torch.LongTensor(ids_buffer),
            torch.LongTensor(attention_mask_buffer), 
            torch.LongTensor(type_ids_buffer), 
            torch.LongTensor(answerable_buffer), 
            torch.LongTensor(start_pos_buffer),
            torch.LongTensor(end_pos_buffer)
        )
            
def get_model(args, tokenizer):
    config = ElectraConfig.from_pretrained('google/electra-base-discriminator')
    config.vocab_size = tokenizer.get_vocab_size() if tokenizer else args.vocab_size
    model = ElectraForSquad(config)
    return model

def get_loss(model, sample, args, device, gpus=0, report=False):
    '''
    gpus (int): Number of gpus used, not the device number
    '''
    ids, mask, type_ids, answerable, start_pos, end_pos = sample
    
    # ids.shape          = (N, L)
    # mask.shape         = (N, L)
    # answerable.shape   = (N)
    # start_pos.shape    = (N)

    if gpus:
        ids = ids.to(device)
        mask = mask.to(device)
        type_ids = type_ids.to(device)
        answerable = answerable.to(device)
        start_pos = start_pos.to(device)
        end_pos = end_pos.to(device)
    try: 
      loss, start_logits, end_logits, answerability = model(ids, mask, type_ids,  start_positions = start_pos, 
                  end_positions = end_pos, is_impossible = answerable, return_dict=False)
    except Exception as e:
        exception_type, exception_object, exception_traceback = sys.exc_info()
        filename = exception_traceback.tb_frame.f_code.co_filename
        line_number = exception_traceback.tb_lineno
        print("Exception type: ", exception_type)
        print("File name: ", filename)
        print("Line number: ", line_number)
        print(e)
        
    print('report = ',report)
    log = None
    if report:
        log = OrderedDict()
        # Find the 
        log['start_acc'] = (start_logits.argmax(-1) == start_pos).sum()/torch.LongTensor([start_pos.shape[0]]).sum().cuda()
        log['end_acc'] = (end_logits.argmax(-1) == end_pos).sum()/torch.LongTensor([end_pos.shape[0]]).sum().cuda()
        log['answerability_acc'] = torch.Tensor((answerability>0).numpy() == answerable.numpy()).sum()/torch.LongTensor([answerability.shape[0]]).sum().cuda()
        log['overall_acc'] = torch.mean(torch.stack([log['start_acc'],log['end_acc'],log['answerability_acc']],dim=0))
    return loss, log


def evaluate(model, sample, args, device, record, gpus=0, report=False):
    ids, mask, type_ids, answerable, start_pos, end_pos = sample

    if gpus:
        ids = ids.to(device)
        mask = mask.to(device) 
        type_ids = type_ids.to(device)
        answerable = answerable.to(device)
        start_pos = start_pos.to(device)
        end_pos = end_pos.to(device)
        
    start_logits, end_logits, answerability = model(ids, mask, type_ids,start_positions = start_pos, 
                end_positions = end_pos, is_impossible = answerable, return_dict=False)

    if 'correct_tot' not in record:
        record['correct_tot'] = 0
    if 'correct' not in record:
        record['correct'] = 0
    
    record['correct'] += (start_logits.argmax(-1) == start_pos).sum()
    record['correct_tot'] += torch.LongTensor([start_pos.shape[0]]).sum().cuda()

def post_evaluate(record, args):
    record['accuracy'] = float(record['correct']) / float(record['correct_tot'])
    record['correct'] = record['correct']
    record['correct_tot'] = record['correct_tot']

def log_formatter(log, tb, step_i):
    log['acc'] = float(log['acc'] / log['acc_tot'])

def get_tokenizer(args):
    return None
    return CanTokenizer(vocab_file = args.vocab_file)

def set_parser(parser):
    parser.add_argument('--base-model', help='model file to import')
    parser.add_argument('--model-size', default='small',
                        help='model size, '
                                'e.g., "small", "base", "large" (default: small)')

def get_dataset(args):
    return textDataset(args.data,args.seq_length,args.batch_size,eval_num_samples=0,)

def get_eval_dataset(args):
    return textDataset(args.data,args.seq_length,args.batch_size,eval_num_samples=0,eval = True)
    pass

get_model
get_loss
log_formatter 
get_tokenizer
evaluate
post_evaluate
set_parser
get_dataset
