from summarization.cnndm_reader import CNN_DM_Reader
from summarization.copynet import CustomCopyNetSeq2Seq
from allennlp.data.vocabulary import Vocabulary

from allennlp.modules.token_embedders import Embedding
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.seq2seq_encoders import PytorchSeq2SeqWrapper

import torch
import torch.optim as optim

from allennlp.modules.attention import DotProductAttention

from allennlp.data.iterators import BucketIterator

from allennlp.training.trainer import Trainer
# reader
reader = CNN_DM_Reader('/home/lv/dataset/CNN_DM_origin/cnn_stories/cnn/stories','/home/lv/dataset/CNN_DM_origin/dailymail_stories/dailymail/stories',save_copy_fields=True,separate_namespaces=True)

train_dataset = reader.read('/home/lv/PycharmProjects/cnn-dailymail/url_lists/all_test.txt')

valid_dataset = reader.read('/home/lv/PycharmProjects/cnn-dailymail/url_lists/all_val.txt')

# vocab
vocab = Vocabulary.from_instances(train_dataset + valid_dataset)

# embedding
EMBEDDING_DIM = 128
token_embedding = Embedding(num_embeddings = vocab.get_vocab_size('tokens'),embedding_dim = EMBEDDING_DIM,trainable= True)

word_embeddings = BasicTextFieldEmbedder({'tokens':token_embedding})

# lstm
HIDDEN_DIM = 256

lstm = PytorchSeq2SeqWrapper(torch.nn.LSTM(EMBEDDING_DIM,HIDDEN_DIM,bidirectional=True,batch_first=True))

# attention

attention = DotProductAttention()


# model

model = CustomCopyNetSeq2Seq(vocab,word_embeddings,lstm,attention,5,100)

# cuda device

if torch.cuda.is_available():
    cuda_device = 0
    model = model.cuda(cuda_device)
else:
    cuda_device = -1


# optimizer

optimizer = optim.Adam(model.parameters(),lr=5e-3)

# iterator

iterator = BucketIterator(batch_size=5,sorting_keys = [('source_tokens','num_tokens')],padding_noise= 0.0, cache_instances=True)

iterator.index_with(vocab)

# trainer

trainer = Trainer(
    model = model,
    optimizer = optimizer,
    iterator = iterator,
    train_dataset = train_dataset,
    validation_dataset= valid_dataset,
    num_epochs = 70,
    cuda_device = cuda_device,
    grad_norm = 2.0,
    summary_interval = True,
    shuffle = False,
    should_log_parameter_statistics = False
)

# train
trainer.train()


