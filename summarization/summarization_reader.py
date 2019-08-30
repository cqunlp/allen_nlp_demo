from typing import Iterable, Dict, Tuple, List


from allennlp.data.dataset_readers import DatasetReader

from allennlp.data.tokenizers import Tokenizer, WordTokenizer
from allennlp.data.token_indexers import TokenIndexer,SingleIdTokenIndexer

from allennlp.data.tokenizers.word_splitter import SimpleWordSplitter

from allennlp.data.instance import Instance

from allennlp.data.tokenizers import Token
from allennlp.common.util import START_SYMBOL,END_SYMBOL

from allennlp.data.fields import TextField, ArrayField, NamespaceSwappingField, MetadataField


import numpy as np
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

        assert save_pgn_fields or save_copy_fields or (not save_copy_fields and not save_pgn_fields)

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


    @staticmethod
    def _tokens_to_ids(tokens: List[Token]) -> List[int]:
        ids = dict()
        out = list()
        for token in tokens:
            out.append(ids.setdefault(token.text.lower(), len(ids)))
        return tokens

    def text_to_instance(self,
            source: str,
            target: str = None
        ) -> Instance:
        def prepare_text(text, max_tokens):
            tokens = self.tokenizer.tokenize(text)[0:max_tokens]
            tokens.insert(0,Token(START_SYMBOL))
            tokens.append(Token(END_SYMBOL))

            return tokens

        # tokenize source sequence
        source_tokens = prepare_text(source,self.source_max_tokens)
        source_tokens_indexed = TextField(source_tokens, self.source_token_indexers)

        result = {'source_tokens': source_tokens_indexed}

        # meta_fields

        meta_fields = {}

        # copy

        if self.save_copy_fields:
            source_to_target_field = NamespaceSwappingField(source_tokens[1:-1],self.target_namespace)
            result['source_to_target'] = source_to_target_field
            meta_fields['source_tokens'] = [x.text for x in source_tokens[1:-1]]
        # pointer

        if self.save_pgn_fields:
            source_to_target_field = NamespaceSwappingField(source_tokens, self.target_namespace)
            result['source_to_target'] = source_to_target_field
            meta_fields['source_tokens'] = [x.text for x in source_tokens]

        if target:
            # target_tokens
            target_tokens = prepare_text(target,self.target_max_tokens)
            target_tokens_indexed = TextField(target_tokens,self.target_token_indexers)
            result['target_tokens'] = target_tokens_indexed

            if self.save_copy_fields:
                meta_fields['target_tokens'] = [y.text for y in target_tokens[1:-1]]
                source_and_target_token_ids = self._tokens_to_ids(source_tokens[1:-1] + target_tokens)
                source_token_ids = source_and_target_token_ids[:len(source_tokens)-2]
                result['source_token_ids'] = ArrayField(np.array(source_token_ids,dtype='long'))

                target_token_ids = source_and_target_token_ids[len(source_tokens)-2:]
                result['target_token_ids'] = ArrayField(np.array(target_token_ids,dtype='long'))


            if self.save_pgn_fields:
                meta_fields['target_tokens'] = [y.text for y in target_tokens]
                source_and_target_token_ids = self._tokens_to_ids(source_tokens + target_tokens)
                source_token_ids = source_and_target_token_ids[:len(source_tokens)]
                result['source_token_ids'] = ArrayField(np.array(source_token_ids,dtype='long'))

                target_token_ids = source_and_target_token_ids[len(source_tokens):]
                result['target_token_ids'] = ArrayField(np.array(target_token_ids,dtype='long'))

        elif self.save_copy_fields:
            source_token_ids = self._tokens_to_ids(source_tokens[1:-1])
            result['source_token_ids'] = ArrayField(np.array(source_token_ids))
        elif self.save_pgn_fields:
            source_token_ids = self._tokens_to_ids(source_tokens)
            result['source_token_ids'] = ArrayField(np.array(source_token_ids))

        if self.save_copy_fields or self.save_pgn_fields:
            result['metadata'] = MetadataField(meta_fields)

        return Instance(result)

    def parse_set(self,file_path:str) -> Iterable[Tuple[str,str]]:
        raise NotImplementedError()