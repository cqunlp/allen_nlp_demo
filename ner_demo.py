from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer

from allennlp.data import Instance
import itertools

from allennlp.data.tokenizers import Token
from allennlp.data.fields import Field, TextField, SequenceLabelField

from typing import Dict, List, Iterator, Optional

from allennlp.models import Model

from allennlp.data.vocabulary import Vocabulary
from allennlp.modules.text_field_embedders import TextFieldEmbedder,BasicTextFieldEmbedder
from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder, PytorchSeq2SeqWrapper
from allennlp.training.metrics import SpanBasedF1Measure

import torch
import torch.nn as nn
import torch.optim as optim

from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits

from allennlp.common.file_utils import cached_path

from allennlp.modules.token_embedders import Embedding

from allennlp.data.iterators import BucketIterator

from allennlp.training.trainer import Trainer

@DatasetReader.register('conll_03_reader')
class CoNLL03DatasetReader(DatasetReader):
    def __init__(self,
                token_indexers: Dict[str, TokenIndexer] = None,
                lazy: bool = False
    ) -> None:
        super().__init__(lazy)
        self.token_indexers = token_indexers or {'tokens':SingleIdTokenIndexer()}
    
    def _read(
        self,
        file_path: str
    ) -> Iterator[Instance]:
        is_divider = lambda line:line.strip() == ''
        with open(file_path, 'r') as conll_file:
            for divider, lines in itertools.groupby(conll_file,is_divider):
                if not divider:
                    fields = [l.strip().split() for l in lines]
                    fields = [l for l in zip(*fields)]
                    tokens, _, _, ner_tags = fields

                    yield self.text_to_instance(tokens,ner_tags)

    def text_to_instance(
        self,
        words: List[str],
        ner_tags: List[str]
    ) -> Instance:
        fields : Dict[str,Field] = {}
        tokens = TextField([Token(w) for w in words], self.token_indexers)

        fields['tokens'] = tokens
        fields['label'] = SequenceLabelField(labels=ner_tags,sequence_field=tokens)

        return Instance(fields)

@Model.register('ner_lstm')
class NerLSTM(Model):
    def __init__(self,
        vocab: Vocabulary,
        embedder: TextFieldEmbedder,
        encoder: Seq2SeqEncoder
    ) -> None:
        super().__init__(vocab)

        self._embedder = embedder
        self._encoder = encoder
        self._classifier = nn.Linear(
            in_features = encoder.get_output_dim(),
            out_features = vocab.get_vocab_size('labels')
        )

        self.f1 = SpanBasedF1Measure(vocab,'labels')

      
     
    def forward(
        self,
        tokens: Dict[str, torch.Tensor],
        label: Optional[torch.Tensor] = None
        ) -> Dict[str, torch.Tensor]:

        mask = get_text_field_mask(tokens)

        embedded = self._embedder(tokens)
        encoded = self._encoder(embedded,mask)
        classified = self._classifier(encoded)

        self.f1(classified,label,mask)

        output : Dict[str, torch.Tensor] = {}

        if label is not None:
            output['loss'] = sequence_cross_entropy_with_logits(classified,label,mask)
        
        return output
    
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return self.f1.get_metric(reset)


# reader

reader = CoNLL03DatasetReader()
train_dataset = reader.read(cached_path('https://raw.githubusercontent.com/synalp/NER/master/corpus/CoNLL-2003/eng.train'))
validation_dataset = reader.read(cached_path('https://raw.githubusercontent.com/synalp/NER/master/corpus/CoNLL-2003/eng.testa'))

vocab = Vocabulary.from_instances(train_dataset+validation_dataset)

# embedding

EMBEDDING_DIM = 50
# use glove
token_embedding = Embedding(num_embeddings=vocab.get_vocab_size('tokens'), embedding_dim=EMBEDDING_DIM,trainable=False,pretrained_file="(http://nlp.stanford.edu/data/glove.6B.zip)#glove.6B.50d.txt")

word_embeddings = BasicTextFieldEmbedder({'tokens':token_embedding})

# lstm
HIDDEN_DIM = 25

lstm = PytorchSeq2SeqWrapper(torch.nn.LSTM(EMBEDDING_DIM,HIDDEN_DIM,bidirectional=True,batch_first=True))

model = NerLSTM(vocab,word_embeddings,lstm)

# cuda device

if torch.cuda.is_available():
    cuda_device = 0
    model = model.cuda(cuda_device)
else:
    cuda_device = -1

# optimizer 

optimizer = optim.Adam(model.parameters(),lr=1e-3)

# iterator
iterator = BucketIterator(batch_size=10,sorting_keys = [('tokens','num_tokens')])

iterator.index_with(vocab)

# trainer

trainer = Trainer(
    model = model,
    optimizer = optimizer,
    iterator = iterator,
    train_dataset = train_dataset,
    validation_dataset= validation_dataset,
    patience= 3,
    num_epochs=10,
    cuda_device= cuda_device,
    validation_metric='-loss',
    grad_clipping=5.0
)

# train

trainer.train()

# save model

with open('/tmp/model.th','wb') as f:
    torch.save(model.state_dict(),f)

# predictor

# comparison


