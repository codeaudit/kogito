from typing import Union, Tuple, List
import itertools
from transformers import DebertaV2ForSequenceClassification, DebertaV2Tokenizer
import spacy
import torch

from kogito.core.linker import KnowledgeLinker
from kogito.core.knowledge import KnowledgeGraph
from kogito.core.utils import truncate_sequences_dual, pad_ids
from kogito.core.relation import RELATION_TO_NL

NARRATIVE_SEP_TOKEN = "<n_sep>"
FACT_SEP_TOKEN = "<f_sep>"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DebertaLinker(KnowledgeLinker):
    def __init__(self, model_name_or_path: str = "mismayil/comfact-deberta-v2", language: str = "en_core_web_sm") -> None:
        super().__init__()
        self.tokenizer = DebertaV2Tokenizer.from_pretrained(model_name_or_path)
        self.model = DebertaV2ForSequenceClassification.from_pretrained(model_name_or_path)
        self.narrative_sep_id = self.tokenizer.convert_tokens_to_ids(NARRATIVE_SEP_TOKEN)
        self.fact_sep_id = self.tokenizer.convert_tokens_to_ids(FACT_SEP_TOKEN)
        self.pad_token_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.pad_token)
        self.nlp = spacy.load(language, exclude=["ner"])
        self.max_input_tokens = 512
        self.model.to(device)

    def save_pretrained(self, save_path: str) -> None:
        """Save pretrained model

        Args:
            save_path (str): Directory to save model to
        """
        if save_path:
            self.model.save_pretrained(save_path)
            self.tokenizer.save_pretrained(save_path)

    @classmethod
    def from_pretrained(cls, model_name_or_path: str) -> KnowledgeLinker:
        """Load pretrained linker

        Args:
            model_name_or_path (str): HuggingFace model name or local model path

        Returns:
            KnowledgeModel: Loaded knowledge linker
        """
        return cls(model_name_or_path)
    
    def link(self, input_graph: KnowledgeGraph, context: Union[List[str], str]) -> List[float]:
        if isinstance(context, str):
            doc = self.nlp(context)
            sentences = []

            for sentence in doc.sents:
                sentences.append(sentence.text)
            
            context = sentences
        
        context_ids = [self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(sent)) for sent in context]
        
        input_ids = []
    
        for kg in input_graph:
            head = str(kg.head).strip().lower()
            relation = str(kg.relation).strip()
            if relation not in RELATION_TO_NL:
                raise ValueError(f"Invalid relation found: {relation}")
            relation = RELATION_TO_NL[relation].lower()
            tail = str(kg.tails[0]).strip().lower()
            fact = [head, relation, tail]
            fact_ids = [self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(f)) for f in fact]
            truncated_context_ids = _truncate_context(context_ids, fact_ids, self.max_input_tokens)
            context_ids_with_sep = list(itertools.chain(*[ids+[self.narrative_sep_id] for ids in truncated_context_ids[:-1]], truncated_context_ids[-1]))
            fact_ids_with_sep = list(itertools.chain(*[ids+[self.fact_sep_id] for ids in fact_ids[:-1]], fact_ids[-1]))
            input_ids.append(self.tokenizer.build_inputs_with_special_tokens(context_ids_with_sep, fact_ids_with_sep))
        
        input_ids = torch.tensor(pad_ids(input_ids, self.pad_token_id))

        with torch.no_grad():
            output = self.model(input_ids.to(device))
            probs = torch.softmax(output.logits, dim=1)[:, 1]
        
        return probs.tolist()

def _truncate_context(context_ids: List[List[int]], fact_ids: List[List[int]], max_tokens: int) -> List[List[int]]:
    num_keep_tokens = 1 + len(context_ids) + len(fact_ids)  # [CLS], [SEP], <d_sep> and <s_sep>
    
    for ids in fact_ids:
        num_keep_tokens += len(ids)  # do not truncate statement

    return truncate_sequences_dual(context_ids, max_tokens-num_keep_tokens)