from typing import List, Union, Tuple
from abc import ABC, abstractmethod, abstractclassmethod

from kogito.core.knowledge import KnowledgeGraph

class KnowledgeLinker(ABC):
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
        raise NotImplementedError
    
    def filter(self, input_graph: KnowledgeGraph, context: Union[List[str], str], threshold: float = 0.5) -> KnowledgeGraph:
        probs = self.link(input_graph, context)
        filtered_kgs = []

        for kg, prob in zip(input_graph, probs):
            if prob >= threshold:
                filtered_kgs.append(kg)
        
        return KnowledgeGraph(filtered_kgs)
    
