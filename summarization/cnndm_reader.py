

from summarization.summarization_reader import SummarizationDatasetReader

from typing import Iterable,Tuple,Dict
from allennlp.data.tokenizers import Tokenizer
from allennlp.data.token_indexers import TokenIndexer

import os
import hashlib

dm_single_close_quote = u'\u2019'
dm_double_close_quote = u'\u201d'
END_TOKENS = ['.', '!', '?', '...', "'", "`", '"', dm_single_close_quote, dm_double_close_quote, ")"]


class CNN_DM_Reader(SummarizationDatasetReader):
    def __init__(
        self,
        cnn_tokenized_dir: str = None,
        dm_tokenized_dir: str = None,
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
        
    ):
        super().__init__(
        tokenizer,
        source_token_indexers,
        target_token_indexers,
        source_max_tokens,
        target_max_tokens,
        separate_namespaces, # for what?
        target_namespace,
        save_copy_fields,
        save_pgn_fields,
        lazy
        )

        self.cnn_tokenized_dir = cnn_tokenized_dir
        self.dm_tokenized_dir = dm_tokenized_dir

    def parse_set(self, urls_path:str) -> Iterable[Tuple[str,str]]:
        file_names = self.get_file_names_by_urls(self.cnn_tokenized_dir,self.dm_tokenized_dir,urls_path)
        for file_name in file_names:
            yield self.get_article_and_abstract(file_name)

    def get_file_names_by_urls(self, cnn_tokenized_dir, dm_tokenized_dir, urls_file_path):
        with open(urls_file_path, 'r', encoding = 'utf-8') as r:
            for url in r:
                url = url.strip()
                file_name = str(self.hashhex(url)) + '.story'
                dirs = (cnn_tokenized_dir,dm_tokenized_dir)
                file_names = [os.path.join(d,file_name) for d in dirs if d is not None]
                file_found = False
                for f in file_names:
                    if os.path.isfile(f):
                        file_found = True
                        yield f
                        break
                assert file_found, "file not found in tokenized dir:" + file_name

    def hashhex(self,s):
        h = hashlib.sha1()
        h.update(s.encode('utf-8'))
        return h.hexdigest()

    def get_article_and_abstract(self, story_file, encoding = 'utf-8', fix_period = True) -> Tuple[str,str]:
        article_lines = []
        abstract = []
        next_is_highlight = False

        with open(story_file, 'r', encoding = encoding) as f:
            for line in f:
                line = line.strip().lower()
                if fix_period:
                    line = self.fix_missing_period(line)
                if not line:
                    continue
                elif line.startswith('@highlight'):
                    next_is_highlight = True
                elif next_is_highlight:
                    abstract.append(line)
                else:
                    article_lines.append(line)
        article = ' '.join(article_lines)
        abstract = ' s_s '.join(abstract)

        return article,abstract
    
    def fix_missing_period(self,line):
        if "@highlight" in line:
            return line
        elif not line:
            return line
        elif line[-1] in END_TOKENS:
            return line
        return line + "."
            