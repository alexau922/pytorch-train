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

def convert(data,tokenizer, max_seq_length, prefix,train = 0.8):
    '''
    Convert the data and encoding the batch, and then shuffle the data and finally return the file
    
    Args:
        data (list of dictionaries): list of dictionaries of tokenize_data and answer_info
        tokenizer (cantokenizer.CanTokenizer object): tokenizer that is used for encoding the text
        max_seq_length (maximum sequence's length): maximum length of the tokenized sequence
        prefix (string): labelling the data classes into training class and testing classes
        train (int, optional): proportion of the training set in the data.
    '''
    if train > 1 or train <=0:
        raise ValueError('The variable "train" denotes the proportion of train set, so it should be larger than 0 but inclusively smaller than 1!')
    
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
        
        # If the start position is not None, then there is no conversion-type error thus we can append the data
        if label_dict['start_pos'] is not None :
            data_start_pos.append(label_dict['start_pos'])
            # If the starting position is larger than the maximum sequence length (which should not happen as we have selected those overflow cases
            # before), then there is some procedural error before so raise error
            if label_dict['start_pos'] > max_seq_length:
                print(label_dict)
                raise 
        # If the starting position is none, meaning that the question is not answerble so we selected to recognize this pattern as at the position 0 ([CLS] tag)
        else:
            data_start_pos.append(0)
        
        # Same treatment on ending position. For more information please refer to previous block of code.
        if label_dict['end_pos'] is not None :
            data_end_pos.append(label_dict['end_pos'])
            if label_dict['end_pos'] > max_seq_length:
                print(label_dict)
                raise 
        else:
            data_end_pos.append(0)
            
        data_type_ids.append(e.type_ids)
        
        # update the progress bar
        pbar.update(10 * i + 1)
    
    # Get an array of indices for the length of data_original
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
    
    # Find the training data length to split the dataset 
    train_length = int(len(indices)*train)
    
    # Split the randomized data into training and testing data set
    train_data_original = data_original[:train_length]
    test_data_original = data_original[train_length:]
    
    train_data_attn_mask = data_attn_mask[:train_length]
    test_data_attn_mask = data_attn_mask[train_length:]
    
    train_data_answerable = data_answerable[:train_length]
    test_data_answerable = data_answerable[train_length:]
    
    train_data_start_pos = data_start_pos[:train_length]
    test_data_start_pos = data_start_pos[train_length:]
    
    train_data_end_pos = data_end_pos[:train_length]
    test_data_end_pos = data_end_pos[train_length:]
    
    train_data_type_ids = data_type_ids[:train_length]
    test_data_type_ids = data_type_ids[train_length:]
    
    # Cast the type of the data so that the calculation will be quicker
    train_ids = np.array(train_data_original).astype(np.int16)
    test_ids = np.array(test_data_original).astype(np.int16)
    
    train_attn = np.array(train_data_attn_mask).astype(np.int8)
    test_attn = np.array(test_data_attn_mask).astype(np.int8)
    
    train_answerable = np.array(train_data_answerable).astype(np.float32)
    test_answerable = np.array(test_data_answerable).astype(np.float32)
    
    # For starting position and ending position, as the position of the desired token can be quite large (~1000), using int8 will only allow values between
    # -128 and 127 which will cause overflow on the position, thus errors for calculating cross entropy loss
    train_start_pos = np.array(train_data_start_pos).astype(np.int16)
    test_start_pos = np.array(test_data_start_pos).astype(np.int16)
    train_end_pos = np.array(train_data_end_pos).astype(np.int16)
    test_end_pos = np.array(test_data_end_pos).astype(np.int16)
    
    train_type_ids = np.array(train_data_type_ids).astype(np.int8)
    test_type_ids = np.array(test_data_type_ids).astype(np.int8)
    
    # The progress bar ends its job
    pbar.finish()
    
    # Write the file into binary format
    with open(prefix+"train_ids",'wb') as f:
        f.write(train_ids.tobytes())
    with open(prefix+'train_mask','wb') as f:
        f.write(train_attn.tobytes())
    with open(prefix+"train_type_ids", 'wb') as f:
        f.write(train_type_ids.tobytes())
    with open(prefix+"train_answerable",'wb')as f:
        f.write(train_answerable.tobytes())
    with open(prefix+"train_start_pos",'wb') as f:
        f.write(train_start_pos.tobytes())
    with open(prefix+"train_end_pos",'wb') as f:
        f.write(train_end_pos.tobytes())
    
    # Write the file into binary format
    with open(prefix+"test_ids",'wb') as f:
        f.write(test_ids.tobytes())
    with open(prefix+'test_mask','wb') as f:
        f.write(test_attn.tobytes())
    with open(prefix+"test_type_ids", 'wb') as f:
        f.write(test_type_ids.tobytes())
    with open(prefix+"test_answerable",'wb')as f:
        f.write(test_answerable.tobytes())
    with open(prefix+"test_start_pos",'wb') as f:
        f.write(test_start_pos.tobytes())
    with open(prefix+"test_end_pos",'wb') as f:
        f.write(test_end_pos.tobytes())
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
    '''
    Gaussian Error Linear Unit activation (to replace sigmoidal function) with an approximated version in the following form.
    Args:
      x (torch.tensor): The tensor to be passed to the activation function. 
    Returns:
      torch.tensor (same shape of x)
    '''
    return 0.5 * x * (1.0 + torch.tanh(x * 0.7978845608 * (1.0 + 0.044715 * x * x)))

class ElectraClassificationHead(nn.Module):
    def __init__(self,config,num_outputs):
        '''
        Initialize the classification head for the electra model (ELECTRA model does not have functional
        head)
        Args:
          config (ElectraConfig object): configuration file that contains all the necessary item to load the pretrained model.
          num_outputs (int): Number of outputs in the final layer that shows the loss.
        Returns:
          None
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
        '''
        Initialize the attributes for ElectraForSquad class, which inherits the Electra pretrained model class.
        Args:
          config (ElectraConfig object): configuration file that contains all the necessary item to load the pretrained model
        Returns:
          None
        '''
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
        # Initialize the weights for the class model (weights inside the neural network layers)
        self.init_weights()
    
    def forward(self,input_ids = None, attn_mask = None, type_ids = None, start_positions = None, 
                end_positions = None, is_impossible = None,return_dict = False):
        '''
        Do forward propagation to calculate the training loss.
        '''
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
          # If we are not doing evaluation, then total_loss will be re-defined
          total_loss = None
        except Except as e:
          import traceback
          traceback.print_exc()
          raise e
        
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
        
        # Initialize classification loss
        cls_losses = []
        
        if is_impossible is not None:
            # BCEWithLogitsLoss is a combination of Sigmoidal function and BCE Loss, which BCE loss stands for Binary Cross Entropy Loss.
            # With BCEWithLogitsLoss we need not do normalization before applying BCE loss as there is a sigmoidal layer that help us normalize the data
            loss_fct_cls = nn.BCEWithLogitsLoss()
            
            # Type cast the tensor to float type
            is_impossible = is_impossible.type(torch.float16)
            #print('answerability.shape = ',answerability.shape)
            #print('is_impossible.shape = ',is_impossible.shape)
            # Use sigmoid to normalize the value and use BCE loss
            answerability_loss = loss_fct_cls(answerability,is_impossible)
            #print('answerability_loss = ',answerability_loss)
            cls_losses.append(answerability_loss)
        
        # if we have the [CLS] tag loss
        if len(cls_losses) >0:
            # Add all the values in the cls_losses
            total_loss += torch.stack(cls_losses, dim = 0).sum()
            
        # If there is no loss (under evaluation), then we will return "output" only.
        output = (start_logits,end_logits,answerability)
        return ((total_loss,)+output) if total_loss is not None else output
      
#####################################
## 
##            Data Utils
## 
#####################################

import numpy as np
import os
class textDataset(Dataset):
    def __doc__(self):
      print(
        '''
        Retrieve the training and testing dataset and split different components in the dataset to tokenized id (ids_buffer), 
        attention mask (attention_mask_buffer), type of id (type_ids_buffer), answerable or not (answerable_buffer), starting position (start_pos_buffer),
        ending position (end_pos_buffer).
        '''
      )
    def __init__(self, 
                 data_path, 
                 seq_length,
                 batch_size, 
                 eval=False, 
                 eval_num_samples=0):
        '''
        Initialize a textDataset object with properties of sequence length, batch size, path of dataset, number of evaluate samples and whether it is evaluating
        or not.
        Args:
          data_path (string): The path to the dataset
          seq_length (int): Length of tokenizer sequence length
          batch_size (int): Number of samples to be fed into training each batch
          eval (boolean, optional): whether we need to evaluate or not
          eval_num_samples (int, optional): Number of samples to be used as evaluation
        Returns:
          None
        '''
        # Initialize the attributes of sequence length, batch size and path of data
        self.seq_length = seq_length
        self.batch_size = batch_size
        self.data_path = data_path
        
        # Calculate the steps of evaluation
        self.eval_num_samples = (eval_num_samples // batch_size)
        # Set whether it is evaluating or not
        self.eval = eval
        
        # Set the prefix on the data file name of whether it is testing or training. If we are going on evaluation, then it is 'test'
        prefix = 'test' if self.eval else 'train'
        
        # For the rest of the code I have no idea what they are doing. Please refer to https://github.com/ecchochan for more information
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
        start_pos_buffer = np.frombuffer(self.start_pos_bin_buffer, dtype=np.int16, count=size, offset=i).reshape((size))
        end_pos_buffer = np.frombuffer(self.end_pos_bin_buffer, dtype=np.int16, count=size, offset=i).reshape((size))
        
        return (
            torch.LongTensor(ids_buffer),
            torch.LongTensor(attention_mask_buffer), 
            torch.LongTensor(type_ids_buffer), 
            torch.LongTensor(answerable_buffer), 
            torch.LongTensor(start_pos_buffer),
            torch.LongTensor(end_pos_buffer)
        )
            
def get_model(args, tokenizer=None):
  '''
  Get the Electra pretrained model and vocab size of the tokenizer.
  Args:
    args (argparse object): Contain arguments that can be found at file train.py or this file function "set_parser"
    tokenizer (tokenizer/CanTokenizer object, optional): an object that can tokenize the english/chinese sentence on a word-character mixed basis. 
  Returns:
    model (ElectraForSquad object): an object that can do forward propagation to calculate the training loss and backward propagation to update the model weights
  '''
    # The config object contains the necessary information like pretrained-model weights (for a specific language and model size weighting) and need to be
    # combined with the Model class to return the object for forward and backward propagation
    config = ElectraConfig.from_pretrained('google/electra-base-discriminator')
    # Set the attribute of vocab_size for tokenization. If we want to specify the vocab_size by ourselves, we need not to pass the tokenizer but to specify
    # when running the script using argparse
    config.vocab_size = tokenizer.get_vocab_size() if tokenizer else args.vocab_size
    # get the model object of the Model class using the configurations we have set
    model = ElectraForSquad(config)
    return model

def get_loss(model, sample, args, device, gpus=0, report=False):
    '''
    Do a forward propagation (calling the neural network model object with forward parameters) to calculate the loss and retrieve the result.
    Args:
      model (ElectraForSquad object): An object that enables forward propagation to calculate the loss and backward propagation for updating the weights
                                      of the ElectraForSquad model.
      sample (torch.tensor of with outermost dimension 6): each training sample bundle together, with contains the ground-truth label and tokenized ids
      args (argparse object): Contain arguments that can be found at file train.py or this file function "set_parser"
      device (int): The device that is used to train the model, in the form of integer (0 denotes the first cuda gpu)
      gpus (int): Number of gpus used, not the device number
      report (bool, optional): whether we need to report the accuracy (for training we do not need to report the accuracy but output the training loss)
    Returns:
      loss (scalar aka tensor([])): A dimension 0 tensor that stores the value of training loss
      log (OrderedDict): A dictionary (that is ordered and iterable) that contains the starting position, ending position, answerability and overall accuracy of 
                         type float
    '''
    # ids.shape          = (N, L)
    # mask.shape         = (N, L)
    # answerable.shape   = (N)
    # start_pos.shape    = (N)
    ids, mask, type_ids, answerable, start_pos, end_pos = sample
    
    if gpus:
    # If gpu is used, we need to place the tensor to gpu
        ids = ids.to(device)
        mask = mask.to(device)
        type_ids = type_ids.to(device)
        answerable = answerable.to(device)
        start_pos = start_pos.to(device)
        end_pos = end_pos.to(device)
    try: 
      '''
      Do a forward propagation then backward propagation. When we call the object that inherits the nn.Module class it is actually running model.forward(param).
      e.g. obj = Class_Inherit_nn()
           result = obj(param) # Here it is running obj.forward(param) 
      When a forward is established, a backward function is also established automatically.
      '''
      # Model returns 4 values as we have provided start_positions, end_positions and is_impossible keys' values which will return the total loss also.
      # if we pass model(ids,mask,type_ids) it will only return start_logits, end_logits, answerability (see function "forward" in "ElectraForSquad" class
      # for more information)
      loss, start_logits, end_logits, answerability = model(ids, mask, type_ids,  start_positions = start_pos, 
                  end_positions = end_pos, is_impossible = answerable, return_dict=False)
    except Exception as e:
        # Print errors if occurs, and raise the error
        import traceback
        traceback.print_exc()
        raise e
    
    # log is an object for storing the training data accuracy, if we specify we need to report the training
    log = None
    if report:
        # log is of dictionary type
        log = OrderedDict()
        '''
        Find the accuracy of
          1. Starting position accuracy
          2. Ending position accuracy
          3. Answerability accuracy
          4. Overall Accuracy
        '''
        # The .argmax(-1) will return the position of the largest value in the last dimension
        # e.g. t = tensor([[1,2,3,4],[1,2,3,4],[1,2,3,4],[1,2,3,4]]) and t.argmax(-1) will return tensor([3,3,3,3]) as for each of the outermost dimension
        # the value at the position 3 is the largest among that dimension.
        # The denominator is the total length of the outermost dimension, so each accuracy is normalized
        log['start_acc'] = (start_logits.argmax(-1) == start_pos).sum()/torch.LongTensor([start_pos.shape[0]]).sum().cuda()
        log['end_acc'] = (end_logits.argmax(-1) == end_pos).sum()/torch.LongTensor([end_pos.shape[0]]).sum().cuda()
        # Calculate the number of match case between the answerability tensor (training vector) and answerable tensor (ground-truth label tensor)
        log['answerability_acc'] = (torch.Tensor((answerability>0).cpu().numpy() == (answerable > 0.5).cpu().numpy()).sum().cuda()/torch.LongTensor([answerability.shape[0]]).sum().cuda())
        # Overall_acc is the arithemetic mean of the three accuracies
        log['overall_acc'] = torch.mean(torch.stack([log['start_acc'],log['end_acc'],log['answerability_acc']],dim=0)).cuda()
    return loss, log


def evaluate(model, sample, args, device, record, gpus=0, report=False):
  '''
  Do a testing on the testing dataset to check how is the model behaving.
   Args:
      model (ElectraForSquad object): An object that enables forward propagation to calculate the loss and backward propagation for updating the weights
                                      of the ElectraForSquad model.
      sample (torch.tensor of with outermost dimension 6): each training sample bundle together, with contains the ground-truth label and tokenized ids
      args (argparse object): Contain arguments that can be found at file train.py or this file function "set_parser"
      device (int): The device that is used to train the model, in the form of integer (0 denotes the first cuda gpu)
      gpus (int): Number of gpus used, not the device number
      report (bool, optional): whether we need to report the accuracy (for training we do not need to report the accuracy but output the training loss)
    Returns:
      loss (scalar aka tensor([])): A dimension 0 tensor that stores the value of training loss
      log (OrderedDict): A dictionary (that is ordered and iterable) that contains the starting position, ending position, answerability and overall accuracy of 
                         type float
  '''
    # ids.shape          = (N, L)
    # mask.shape         = (N, L)
    # answerable.shape   = (N)
    # start_pos.shape    = (N)
    ids, mask, type_ids, answerable, start_pos, end_pos = sample

    if gpus:
    # If gpu is used, we need to place the tensor to gpu
        ids = ids.to(device)
        mask = mask.to(device) 
        type_ids = type_ids.to(device)
        answerable = answerable.to(device)
        start_pos = start_pos.to(device)
        end_pos = end_pos.to(device)
    
    # Model returns 3 values only as we only pass the tokenized ids, attention mask (boolean, function similar to netmask) and type of ids (boolean) to the model.
    # if we pass model(ids,mask,type_ids) it will only return start_logits, end_logits, answerability (see function "forward" in "ElectraForSquad" class
    # for more information)
    start_logits, end_logits, answerability = model(ids, mask, type_ids, return_dict=False)
    
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
  '''
  A function that is used in the previous version that is not for calculating starting, ending position and answerability accuracy. Due to compatibility
  issue in the train.py (written manually) we leave the function name without doing anything.
  Args:
    log (OrderedDict): a dictionary (ordered thus iterable) that stores starting position, ending position, answerability and overall accuracy.
    tb (i don't know what is this, please refer to https://github.com/ecchochan for more information)
    step_i (i don't know what is this, please refer to https://github.com/ecchochan for more information)
  '''
    pass

def get_tokenizer(args):
    return None
    return CanTokenizer(vocab_file = args.vocab_file)

def set_parser(parser):
  '''
  Set some **more** arguments to pass to the model during training and evaluation.
  Args:
    parser (argparse object): The parser that is originally set in the train.py file. Its functionality is extended for this ElectraForSquad model.
  Returns:
    None
  '''
    # add the argument of the base model file
    parser.add_argument('--base-model', help='model file to import')
    # add the argument of setting the Electra model size, default small size
    parser.add_argument('--model-size', default='small',
                        help='model size, '
                                'e.g., "small", "base", "large" (default: small)')

def get_dataset(args):
  '''
  Get the dataset for training.
  Args:
    args (argparse object): Contain arguments that can be found at file train.py or this file function "set_parser"
  Returns:
    textDataset object 
  '''
    # Return a textDataset object that contains all the attributes important to the data and the methods to train with the data
    return textDataset(args.data,args.seq_length,args.batch_size,eval_num_samples=0,)

def get_eval_dataset(args):
  '''
  Get the dataset for eva;iatopm.
  Args:
    args (argparse object): Contain arguments that can be found at file train.py or this file function "set_parser"
  Returns:
    textDataset object 
  '''
    # Return a textDataset object that contains all the attributes important to the data and the methods to evaluate the test set
    return textDataset(args.data,args.seq_length,args.batch_size,eval_num_samples=0,eval = True)

get_model
get_loss
log_formatter 
get_tokenizer
evaluate
post_evaluate
set_parser
get_dataset
