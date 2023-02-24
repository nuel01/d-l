#!/usr/bin/env python
# coding: utf-8

# In[1]:


#get_ipython().system('git clone https://github.com/nuel01/d-l.git')


# In[2]:


#get_ipython().system('pip install transformers==4.18.0 sentencepiece')
#get_ipython().system('pip install accelerate')


# In[3]:


from datasets import *
from transformers import *
from tokenizers import *
import os
import json
#from huggingface_hub import notebook_login

#notebook_login()


# In[ ]:


# download and prepare cc_news dataset
#dataset = load_dataset("cc_news", split="train")


# In[ ]:


# split the dataset into training (90%) and testing (10%)
#d = dataset.train_test_split(test_size=0.1)
#d["train"], d["test"]


# In[ ]:


#for t in d["train"]["text"][:3]:
#  print(t)
#  print("="*50)


# In[ ]:


# if you have your custom dataset 
# dataset = LineByLineTextDataset(
#     tokenizer=tokenizer,
#     file_path="path/to/data.txt",
#     block_size=64,
# )


# In[4]:


# or if you have huge custom dataset separated into files
# load the splitted files
files_dir = ["/d-l/otc/0", "/d-l/otc/1",
            "/d-l/otc/2", "/d-l/otc/3",
            "/d-l/otc/4", "/d-l/otc/5",
            "/d-l/otc/6", "/d-l/otc/7",
            "/d-l/otc/8", "/d-l/otc/9",
            "/d-l/otc/a", "/d-l/otc/b",
            "/d-l/otc/c", "/d-l/otc/d",
            "/d-l/otc/e", "/d-l/otc/f"] # train3.txt, etc.

for f in files_dir:
    #print(str(f))
    files = [str(f)+'/'+ff for ff in os.listdir(f)]

print(files[0])
dataset = load_dataset("text", data_files=files, split="train")


# split the dataset into training (90%) and testing (10%)
d = dataset.train_test_split(test_size=0.15)
d["train"], d["test"]


# In[5]:


# if you want to train the tokenizer from scratch (especially if you have custom
# dataset loaded as datasets object), then run this cell to save it as files
# but if you already have your custom data as text files, there is no point using this

def dataset_to_text(dataset, output_filename="data.txt"):
  """Utility function to save dataset text to disk,
  useful for using the texts to train the tokenizer 
  (as the tokenizer accepts files)"""
  with open(output_filename, "w") as f:
    for t in dataset["text"]:
      print(t, file=f)

# save the training set to train.txt
dataset_to_text(d["train"], "train.txt")
# save the testing set to test.txt
dataset_to_text(d["test"], "test.txt")


# In[7]:


special_tokens = [
  "[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "<S>", "<T>"
]
# if you want to train the tokenizer on both sets
# files = ["train.txt", "test.txt"]
# training the tokenizer on the training set
files = ["train.txt"]
# 30,522 vocab is BERT's default vocab size, feel free to tweak
vocab_size = 30_522
# maximum sequence length, lowering will result to faster training (when increasing batch size)
max_length = 512
# whether to truncate
truncate_longer_samples = True
TOKENIZER_BATCH_SIZE = 256  # Batch-size to train the tokenizer on


# In[8]:


# initialize the WordPiece tokenizer
#tokenizer = BertWordPieceTokenizer()
#import tokenizers
#print(tokenizers.__version__)
'''
pub_tokenizer = AutoTokenizer.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")

def batch_iterator():
    for i in range(0, len(files), TOKENIZER_BATCH_SIZE):
        yield files[i : i + TOKENIZER_BATCH_SIZE]
# train the tokenizer
#tokenizer = pub_tokenizer.train_new_from_iterator(files=files, vocab_size=vocab_size, special_tokens=special_tokens)
tokenizer = pub_tokenizer.train_new_from_iterator(
    batch_iterator(), vocab_size=vocab_size
)
'''

# initialize the WordPiece tokenizer
tokenizer = BertWordPieceTokenizer()
# train the tokenizer
tokenizer.train(files=files, vocab_size=vocab_size, special_tokens=special_tokens)
# enable truncation up to the maximum 512 tokens
tokenizer.enable_truncation(max_length=max_length)

# enable truncation up to the maximum 512 tokens
#tokenizer.enable_truncation(max_length=max_length)


# In[9]:


#from huggingface_hub import notebook_login

#notebook_login()

#import os

#os.environ["HF_ENDPOINT"] = "https://huggingface.co"
model_path = "/druglabelBert"
# make the directory if not already there
if not os.path.isdir(model_path):
  os.mkdir(model_path)


# In[10]:


# save the tokenizer  
tokenizer.save_model(model_path)


# In[11]:


# dumping some of the tokenizer config to config file, 
# including special tokens, whether to lower case and the maximum sequence length
with open(os.path.join(model_path, "config.json"), "w") as f:
  tokenizer_cfg = {
      "do_lower_case": True,
      "unk_token": "[UNK]",
      "sep_token": "[SEP]",
      "pad_token": "[PAD]",
      "cls_token": "[CLS]",
      "mask_token": "[MASK]",
      "model_max_length": max_length,
      "max_len": max_length,
  }
  json.dump(tokenizer_cfg, f)


# In[12]:


# when the tokenizer is trained and configured, load it as BertTokenizerFast
tokenizer = BertTokenizerFast.from_pretrained(model_path)


# In[13]:


def encode_with_truncation(examples):
  """Mapping function to tokenize the sentences passed with truncation"""
  return tokenizer(examples["text"], truncation=True, padding="max_length",
                   max_length=max_length, return_special_tokens_mask=True)

def encode_without_truncation(examples):
  """Mapping function to tokenize the sentences passed without truncation"""
  return tokenizer(examples["text"], return_special_tokens_mask=True)

# the encode function will depend on the truncate_longer_samples variable
encode = encode_with_truncation if truncate_longer_samples else encode_without_truncation

# tokenizing the train dataset
train_dataset = d["train"].map(encode, batched=True)
# tokenizing the testing dataset
test_dataset = d["test"].map(encode, batched=True)

if truncate_longer_samples:
  # remove other columns and set input_ids and attention_mask as PyTorch tensors
  train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
  test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
else:
  # remove other columns, and remain them as Python lists
  test_dataset.set_format(columns=["input_ids", "attention_mask", "special_tokens_mask"])
  train_dataset.set_format(columns=["input_ids", "attention_mask", "special_tokens_mask"])


# In[14]:


from itertools import chain
# Main data processing function that will concatenate all texts from our dataset and generate chunks of
# max_seq_length.
# grabbed from: https://github.com/huggingface/transformers/blob/main/examples/pytorch/language-modeling/run_mlm.py
def group_texts(examples):
    # Concatenate all texts.
    concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
    # customize this part to your needs.
    if total_length >= max_length:
        total_length = (total_length // max_length) * max_length
    # Split by chunks of max_len.
    result = {
        k: [t[i : i + max_length] for i in range(0, total_length, max_length)]
        for k, t in concatenated_examples.items()
    }
    return result

# Note that with `batched=True`, this map processes 1,000 texts together, so group_texts throws away a
# remainder for each of those groups of 1,000 texts. You can adjust that batch_size here but a higher value
# might be slower to preprocess.
#
# To speed up this part, we use multiprocessing. See the documentation of the map method for more information:
# https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.map
if not truncate_longer_samples:
  train_dataset = train_dataset.map(group_texts, batched=True,
                                    desc=f"Grouping texts in chunks of {max_length}")
  test_dataset = test_dataset.map(group_texts, batched=True,
                                  desc=f"Grouping texts in chunks of {max_length}")
  # convert them from lists to torch tensors
  train_dataset.set_format("torch")
  test_dataset.set_format("torch")


# In[15]:


len(train_dataset), len(test_dataset)


# In[16]:


# initialize the model with the config
model_config = BertConfig(vocab_size=vocab_size, max_position_embeddings=max_length)
model = BertForMaskedLM(config=model_config)


# In[17]:


# initialize the data collator, randomly masking 20% (default is 15%) of the tokens for the Masked Language
# Modeling (MLM) task
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.2
)


# In[19]:


training_args = TrainingArguments(
    output_dir=model_path,          # output directory to where save model checkpoint
    evaluation_strategy="steps",    # evaluate each `logging_steps` steps
    overwrite_output_dir=True,      
    num_train_epochs=2,            # number of training epochs, feel free to tweak
    per_device_train_batch_size=10, # the training batch size, put it as high as your GPU memory fits
    gradient_accumulation_steps=8,  # accumulating the gradients before updating the weights
    per_device_eval_batch_size=64,  # evaluation batch size
    logging_steps=1000,             # evaluate, log and save model checkpoints every 1000 step
    save_steps=1000,
    push_to_hub=True,
    load_best_model_at_end=True,
    auto_find_batch_size=True,
      # whether to load the best model (in terms of loss) at the end of training
    # save_total_limit=3,           # whether you don't have much space so you let only 3 model weights saved in the disk
)


# In[20]:


# initialize the trainer and pass everything to it
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)


# In[21]:


# train the model
trainer.train()


# In[ ]:


# when you load from pretrained
trainer.push_to_hub()
#model = BertForMaskedLM.from_pretrained(os.path.join(model_path, "checkpoint-6000"))
#tokenizer = BertTokenizerFast.from_pretrained(model_path)
# or simply use pipeline
fill_mask = pipeline("fill-mask", model=model, tokenizer=tokenizer)


# In[ ]:


# perform predictions
example = "Orthostatic hypotension may [MASK] and be aggravated by alcohol, barbiturates or narcotics."
for prediction in fill_mask(example):
  print(prediction)


# In[ ]:


# perform predictions
examples = [
  "Furosemide [MASK] should be reduced or therapy withdrawn.",
  "The adverse events most commonly associated with withdrawal in pediatric [MASK] were emotional lability, hostility, and hyperkinesia.",
]
for example in examples:
  for prediction in fill_mask(example):
    print(f"{prediction['sequence']}, confidence: {prediction['score']}")
  print("="*50)


# In[ ]:


#get_ipython().system('nvidia-smi')

