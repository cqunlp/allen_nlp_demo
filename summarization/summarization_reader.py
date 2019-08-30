from typing import Iterable, Dict, Tuple, List


from allennlp.data.dataset_readers import DatasetReader

from allennlp.data.tokenizers import Tokenizer, WordTokenizer
from allennlp.data.token_indexers import TokenIndexer,SingleIdTokenIndexer

from allennlp.data.tokenizers.word_splitter import SimpleWordSplitter

from allennlp.data.instance import Instance

class SummarizationDatasetReader(DatasetReader):
    def __init__(
        self,
        tokenizer: Tokenizer = None,
        source_token_indexers: Dict[str, TokenIndexer] = None,
        target_token_indexers: Dict[str, TokenIndexer] = None,
        source_max_tokens : int = 400,
        target_max_tokens : int = 100,
        separate_namespaces: bool = False, # for what?
        target_namespace: str = 'target_tokens',
        save_copy_fields: bool = False,
        save_pgn_fields: bool = False,
        lazy: bool = False
        ) -> None:
        
        super().__init__(lazy)

        assert save_pgn_fields pr save_copy_fields or (not save_copy_fields and not save_pgn_fields)

        self.source_max_tokens = source_max_tokens
        self.target_max_tokens = target_max_tokens

        self.tokenizer = tokenizer or WordTokenizer(word_splitter=SimpleWordSplitter())

        tokens_indexer = {'tokens':SingleIdTokenIndexer()}

        self.source_token_indexers = source_token_indexers or tokens_indexer
        self.target_token_indexers = target_token_indexers or tokens_indexer

        self.save_copy_fields = save_copy_fields
        self.save_pgn_fields = save_pgn_fields

        self.target_namespace = 'tokens'
        if separate_namespaces:
            self.target_namespace = target_namespace
            second_tokens_indexer = {'tokens':SingleIdTokenIndexer(namespace=target_namespace)}
            self.target_token_indexers = target_token_indexers or second_tokens_indexer


    def _read(self, file_path:str) -> Iterable[Instance]:
        for source, target in self.parse_set(file_path):
            if not source or not target:
                continue
            instance = self.text_to_instance(source,target)
            yield instance

    def text_to_instance(self):
        pass

    def parse_set(self,file_path:str) -> Iterable[Tuple(str,str)]:
        raise NotImplementedError()