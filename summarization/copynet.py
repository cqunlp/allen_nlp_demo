import torch

from allennlp.data.vocabulary import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import Attention, Seq2SeqEncoder,TextFieldEmbedder
from allennlp.models.encoder_decoders import CopyNetSeq2Seq
from allennlp.training.metrics import Metric

class CustomCopyNetSeq2Seq(CopyNetSeq2Seq):
    def __init__(self,
                 vocab: Vocabulary,
                 source_embedder: TextFieldEmbedder,
                 encoder: Seq2SeqEncoder,
                 attention: Attention,
                 beam_size: int,
                 max_decoding_steps: int,
                 target_embedding_dim: int = None,
                 copy_token: str = "@COPY@",
                 source_namespace: str = "source_tokens",
                 target_namespace: str = "target_tokens",
                 tensor_based_metric: Metric = None,
                 token_based_metric: Metric = None,
                 tie_embeddings: bool = False
    ) -> None:
        target_embedding_dim = target_embedding_dim or source_embedder.get_output_dim()
        super().__init__(
            vocab,
            source_embedder,
            encoder,
            attention,
            beam_size,
            max_decoding_steps,
            target_embedding_dim,
            copy_token,
            source_namespace,
            target_namespace,
            tensor_based_metric,
            token_based_metric
        )

        self.tie_embeddings = tie_embeddings
        if self.tie_embeddings:
            assert source_namespace == target_namespace
            assert "token_embedder_tokens" in dict(self._source_embedder.named_children())
            source_token_embedder = dict(self._source_embedder.named_children())['token_embedder_tokens']

            self._target_embedder.weight = source_token_embedder.weight
        if tensor_based_metric is None:
            self._tensor_based_metric = None
