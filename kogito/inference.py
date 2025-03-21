from typing import Union, List, Optional
import warnings

import spacy

from kogito.core.knowledge import Knowledge, KnowledgeGraph
from kogito.core.head import KnowledgeHead
from kogito.core.processors.head import (
    KnowledgeHeadExtractor,
    SentenceHeadExtractor,
    NounPhraseHeadExtractor,
    VerbPhraseHeadExtractor,
)
from kogito.core.relation import KnowledgeRelation
from kogito.core.processors.relation import (
    GraphBasedRelationMatcher,
    KnowledgeRelationMatcher,
    SimpleRelationMatcher,
    BaseRelationMatcher,
)
from kogito.core.model import KnowledgeModel
from kogito.core.linker import KnowledgeLinker
from kogito.models.gpt3.zeroshot import GPT3Zeroshot
from kogito.linkers.deberta import DebertaLinker


class CommonsenseInference:
    """Main interface for commonsense inference"""

    def __init__(self, language: str = "en_core_web_sm") -> None:
        """Initialize a commonsense inference module

        Args:
            language (str, optional): Spacy language pipeline to use. Defaults to "en_core_web_sm".
        """
        self.language = language
        self.nlp = spacy.load(language, exclude=["ner"])

        self._head_processors = {
            "sentence_extractor": SentenceHeadExtractor("sentence_extractor", self.nlp),
            "noun_phrase_extractor": NounPhraseHeadExtractor(
                "noun_phrase_extractor", self.nlp
            ),
            "verb_phrase_extractor": VerbPhraseHeadExtractor(
                "verb_phrase_extractor", self.nlp
            ),
        }
        self._relation_processors = {
            "simple_relation_matcher": SimpleRelationMatcher(
                "simple_matcher", self.nlp
            ),
            "graph_relation_matcher": GraphBasedRelationMatcher(
                "graph_matcher", self.nlp
            ),
        }

    @property
    def processors(self) -> dict:
        """List all processors

        Returns:
            dict: List of head and relation processors
        """
        return {
            "head": list(self._head_processors.keys()),
            "relation": list(self._relation_processors.keys()),
        }

    def infer(
        self,
        text: Optional[str] = None,
        model: Optional[KnowledgeModel] = None,
        heads: Optional[List[str]] = None,
        model_args: Optional[dict] = None,
        extract_heads: bool = True,
        match_relations: bool = True,
        relations: Optional[List[KnowledgeRelation]] = None,
        dry_run: bool = False,
        sample_graph: Optional[KnowledgeGraph] = None,
        context: Optional[Union[List[str], str]] = None,
        linker: Optional[KnowledgeLinker] = None,
        threshold: float = 0.5,
    ) -> KnowledgeGraph:
        """Make commonsense inferences.

        Args:
            text (Optional[str], optional): Text to use to extract commonsense inferences from.
                If omitted, no head extraction will be performed even if ``extract_heads`` is True.
                If provided and ``extract_heads`` is False, text will be used as a head as is.
                Defaults to None.
            model (Optional[KnowledgeModel], optional): Knowledge model to use for inference.
                If omitted, behaviour is equivalent to dry-run mode, i.e. no inference will be performed
                and incomplete input graph will be returned.
                Defaults to None.
            heads (Optional[List[str]], optional): List of custom heads to use for inference. Defaults to None.
            model_args (Optional[dict], optional): Custom arguments to pass to ``KnowledgeModel.generate()`` method.
                Defaults to None.
            extract_heads (bool, optional): Whether to extract heads from given text if any. Defaults to True.
            match_relations (bool, optional): Whether to do smart relation matching. Defaults to True.
            relations (Optional[List[KnowledgeRelation]], optional): Subset of relations to use for direct matching.
                If ``match_relations`` is true, intersection of matched and given relations will be used.
                Defaults to None.
            dry_run (bool, optional): Whether to skip actual inference and return incomplete input graph.
                Defaults to False.
            sample_graph (Optional[KnowledgeGraph], optional): A knowledge graph containing examples.
                It can be used to provide examples for GPT-3 inference. If omitted and GPT-3 model is used,
                warning will be raised.
                Defaults to None.
            context (Optional[Union[List[str], str]], optional): Context text. Can be either given as a list of
                                            sentences or as a string, in which case, it will be
                                            split into sentences using spacy engine. Defaults to None.
            threshold (float, optional): Relevance probability used for filtering. Defaults to 0.5.
            linker (Optional[KnowledgeLinker], optional): Knowledge linker model used for linking to given context.
                                                            Defaults to Deberta-based linker.

        Raises:
            ValueError: if relations argument is not of type list

        Returns:
            KnowledgeGraph: Inferred knowledge graph.
        """
        kg_heads = []
        head_relations = set()
        head_texts = set()
        model_args = model_args or {}

        if relations is not None and not isinstance(relations, list):
            raise ValueError("Relation subset should be a list")

        if text is not None:
            if not isinstance(text, str):
                raise ValueError("Text should be a string")
            text = text.strip()

        if heads is not None:
            if not isinstance(heads, list):
                raise ValueError("Heads should be a list")

        if not text and not heads:
            warnings.warn("Skipping inference, no text or head provided")
            return None

        if heads:
            for head in heads:
                if isinstance(head, str) and head.strip():
                    head_texts.add(head)
                    kg_heads.append(KnowledgeHead(text=head))

        if extract_heads:
            if text.strip():
                print("Extracting knowledge heads...")
                for head_proc in self._head_processors.values():
                    extracted_heads = head_proc.extract(text)
                    for head in extracted_heads:
                        head_text = head.text.strip().lower()
                        # Check for duplication
                        if head_text not in head_texts:
                            kg_heads.append(head)
                            head_texts.add(head_text)
        else:
            if text and text.strip() not in head_texts:
                head_texts.add(text)
                kg_heads.append(KnowledgeHead(text=text))

        if not kg_heads:
            warnings.warn("Skipping inference, no heads found.")
            return None

        base_relation_matcher = BaseRelationMatcher("base-relation-matcher")

        if match_relations:
            print("Matching knowledge heads with relations...")
            for relation_proc in self._relation_processors.values():
                head_relations = head_relations.union(
                    set(
                        relation_proc.match(
                            kg_heads, relations, sample_graph=sample_graph
                        )
                    )
                )
        else:
            print("Pairing knowledge heads with all relations...")
            head_relations = head_relations.union(
                set(
                    base_relation_matcher.match(
                        kg_heads, relations, sample_graph=sample_graph
                    )
                )
            )

        kg_list = []

        for head_relation in head_relations:
            head, relation = head_relation
            kg_list.append(Knowledge(head=head, relation=relation))

        input_graph = KnowledgeGraph(kg_list)

        if sample_graph:
            input_graph = input_graph + sample_graph
        else:
            if isinstance(model, GPT3Zeroshot):
                warnings.warn(
                    "Sample graph not found, but recommended for good performance with GPT-3 based inference."
                )

        if dry_run or not model:
            return input_graph.sort()

        print("Generating knowledge graph...")
        output_graph = model.generate(input_graph, **model_args)

        if context:
            if not linker:
                print("Loading default Deberta linker for filtering...")
                linker = DebertaLinker()

            print("Filtering knowledge graph based on the context...")
            output_graph = linker.filter(output_graph, context, threshold=threshold)

        output_graph.clean()
        output_graph.sort()

        return output_graph

    def add_processor(
        self, processor: Union[KnowledgeHeadExtractor, KnowledgeRelationMatcher]
    ) -> None:
        """Add a new head or relation processor to the module

        Args:
            processor (Union[KnowledgeHeadExtractor, KnowledgeRelationMatcher]): Head or relation processor.

        Raises:
            ValueError: When processor type is not recognized.
        """
        if isinstance(processor, KnowledgeHeadExtractor):
            self._head_processors[processor.name] = processor
            processor.lang = self.nlp
        elif isinstance(processor, KnowledgeRelationMatcher):
            self._relation_processors[processor.name] = processor
            processor.lang = self.nlp
        else:
            raise ValueError("Unknown processor")

    def remove_processor(self, processor_name: str) -> None:
        """Remove a processor from the module

        Args:
            processor_name (str): Name of the processor to remove
        """
        if processor_name in self._head_processors:
            del self._head_processors[processor_name]
        elif processor_name in self._relation_processors:
            del self._relation_processors[processor_name]
