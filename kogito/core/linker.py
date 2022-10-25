from typing import List, Union, Tuple
from abc import ABC, abstractmethod, abstractclassmethod

from kogito.core.knowledge import KnowledgeGraph

class KnowledgeLinker(ABC):
    """Base Knowledge Linker"""

    @abstractmethod
    def save_pretrained(self, save_path: str) -> None:
        """Save linker as a pretrained model

        Args:
            save_path (str): Directory to save the linker to.

        Raises:
            NotImplementedError: This method has to be implemented by
                                 concrete subclasses.
        """
        raise NotImplementedError

    @abstractclassmethod
    def from_pretrained(cls, model_name_or_path: str) -> "KnowledgeLinker":
        """Load model from a pretrained model path
        This method can load linkers either from HuggingFace by model name
        or from disk by model path.

        Args:
            model_name_or_path (str): HuggingFace model name or local model path to load from.

        Raises:
            NotImplementedError: This method has to be implemented by
                                 concrete subclasses.

        Returns:
            KnowledgeLinker: Loaded knowledge linker.
        """
        raise NotImplementedError
    
    @abstractmethod
    def link(self, input_graph: KnowledgeGraph, context: Union[List[str], str]) -> List[float]:
        """Link given knowledge graph with the context. 
        This method computes a relevance probability for each knowledge in the graph
        with respect to the given context and returns these probabilities in a list
        in the same order as the knowledge tuples are in the given graph.

        Args:
            input_graph (KnowledgeGraph): Input graph to link.
            context (Union[List[str], str]): Context text. Can be either given as a list of
                                            sentences or as a string, in which case, it will be
                                            split into sentences using spacy engine.

        Returns:
            List[float]: List of relevance probabilities
        """
        raise NotImplementedError
    
    def filter(self, input_graph: KnowledgeGraph, context: Union[List[str], str], threshold: float = 0.5) -> KnowledgeGraph:
        """Filter given graph based on context relevancy. 
        This method under the hood links the graph to the context and then filters knowledge tuples from the graph
        which have a relevance probability lower than the given threshold. Filtered knowledge tuples
        are returned as a new knowledge graph.

        Args:
            input_graph (KnowledgeGraph): Input graph to filter.
            context (Union[List[str], str]): Context text. Can be either given as a list of
                                            sentences or as a string, in which case, it will be
                                            split into sentences using spacy engine.
            threshold (float, optional): Relevance probability used for filtering. Defaults to 0.5.

        Returns:
            KnowledgeGraph: Filtered knowledge graph based on the relevancy scores.
        """
        probs = self.link(input_graph, context)
        filtered_kgs = []

        for kg, prob in zip(input_graph, probs):
            if prob >= threshold:
                filtered_kgs.append(kg)
        
        return KnowledgeGraph(filtered_kgs)
    
