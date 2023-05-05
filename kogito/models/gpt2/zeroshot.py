import numpy as np
import torch
from torch import cuda
from transformers import GPT2Tokenizer, GPT2LMHeadModel

from kogito.core.model import KnowledgeModel
from kogito.core.knowledge import KnowledgeGraph

device = "cuda" if cuda.is_available() else "cpu"


class GPT2Zeroshot(KnowledgeModel):
    """Zeroshot knowledge model based on GPT-2"""

    def __init__(self, gpt2_model: str = "gpt2") -> None:
        """Initialize GPT-2 model
        Args:
            gpt2_model (str, optional): HuggingFace model name for gpt2. Defaults to "gpt2".
        """
        self.tokenizer = GPT2Tokenizer.from_pretrained(gpt2_model)
        self.model = GPT2LMHeadModel.from_pretrained(gpt2_model)
        self.model.to(device)

    def train(self):
        raise ValueError("GPT-2 Zeroshot model is not trainable")

    def save_pretrained(self, save_path):
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)

    @classmethod
    def from_pretrained(cls, model_name_or_path: str = "gpt2"):
        return cls(model_name_or_path)

    def generate(
        self, input_graph: KnowledgeGraph, seed: int = 42, **kwargs
    ) -> KnowledgeGraph:
        """Generate inferences from GPT2 model
        Args:
            input_graph (KnowledgeGraph): Input dataset
            seed (int, optional): Random seed. Defaults to 42.
            kwargs: Additional arguments to pass to the model.generate() function
        Returns:
            KnowledgeGraph: Completed knowledge graph
        """
        torch.manual_seed(seed)
        np.random.seed(seed)
        torch.backends.cudnn.deterministic = True

        if "top_k" not in kwargs:
            kwargs["top_k"] = 1

        if "top_p" not in kwargs:
            kwargs["top_p"] = 0.9

        if "num_return_sequences" not in kwargs:
            kwargs["num_return_sequences"] = 3

        if "num_beams" not in kwargs:
            kwargs["num_beams"] = 3

        if "temperature" not in kwargs:
            kwargs["temperature"] = 0.7

        if "repetition_penalty" not in kwargs:
            kwargs["repetition_penalty"] = 1.2

        if "max_length" not in kwargs:
            kwargs["max_length"] = 32

        if "do_sample" not in kwargs:
            kwargs["do_sample"] = True

        outputs = []
        for input_kg in input_graph:
            prompt = input_kg.to_prompt()
            input_ids = self.tokenizer.encode(
                prompt, add_special_tokens=False, return_tensors="pt"
            )
            input_length = input_ids.size(1)
            generations = self.model.generate(
                input_ids=input_ids.to(device),
                max_length=input_length + kwargs["max_length"],
                eos_token_id=198,
                **kwargs
            )

            if len(generations.shape) > 2:
                generations.squeeze_()

            text_generations = []
            for gen in generations:
                gen = gen.tolist()
                text = self.tokenizer.decode(
                    gen[input_length:], clean_up_tokenization_spaces=True
                )
                text_generations.append(text.strip())

            output_kg = input_kg.copy()
            output_kg.tails = text_generations
            outputs.append(output_kg)

        return KnowledgeGraph(outputs)
