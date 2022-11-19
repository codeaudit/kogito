import numpy as np
import torch
from torch import cuda
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from tqdm import tqdm

from kogito.core.model import KnowledgeModel
from kogito.core.knowledge import KnowledgeGraph
from kogito.core.utils import chunks, trim_batch

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
        self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.model.resize_token_embeddings(len(self.tokenizer))

    def train(self):
        raise ValueError("GPT-2 Zeroshot model is not trainable")

    def save_pretrained(self, save_path):
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)

    @classmethod
    def from_pretrained(cls, model_name_or_path: str = "gpt2"):
        return cls(model_name_or_path)

    def generate(
        self,
        input_graph: KnowledgeGraph,
        seed: int = 42,
        top_k: int = 1,
        top_p: float = 0.9,
        num_sequences: int = 3,
        num_beams: int = 3,
        temperature: float = 0.7,
        repetition_penalty: float = 1.2,
        max_length: int = 32,
        batch_size: int = 4
    ) -> KnowledgeGraph:
        """Generate inferences from GPT2 model

        Args:
            input_graph (KnowledgeGraph): Input dataset
            seed (int, optional): Random seed. Defaults to 42.
            top_k (int, optional): GPT-2 top k parameter. Defaults to 1.
            top_p (float, optional): GPT-2 top p parameter. Defaults to 0.9.
            num_sequences (int, optional): GPT-2 num_return_sequences parameter. Defaults to 3.
            num_beams (int, optional): GPT-2 num_beams parameter. Defaults to 3.
            temperature (float, optional): GPT-2 temperature parameter. Defaults to 0.7.
            repetition_penalty (float, optional): GPT-2 repetition_penalty parameter. Defaults to 1.2.
            max_length (int, optional): Max length of generated tokens. Defaults to 32.
            batch_size (int, optional): Batch size to use. Defaults to 64.

        Returns:
            KnowledgeGraph: Completed knowledge graph
        """
        torch.manual_seed(seed)
        np.random.seed(seed)
        torch.backends.cudnn.deterministic = True

        outputs = []
        for kg_batch in list(chunks(input_graph, batch_size)):
            prompts = []

            for input_kg in kg_batch:
                prompts.append(input_kg.to_prompt())
            
            tokenized_batch = self.tokenizer(
                prompts, truncation=True, padding="longest", return_tensors="pt"
            )

            input_ids, attention_mask = trim_batch(
                **tokenized_batch, pad_token_id=self.tokenizer.pad_token_id
            )

            generations = self.model.generate(
                input_ids=input_ids.to(device),
                attention_mask=attention_mask.to(device),
                max_new_tokens=max_length,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                eos_token_id=198,
                pad_token_id=self.tokenizer.pad_token_id,
                repetition_penalty=repetition_penalty,
                do_sample=True,
                num_return_sequences=num_sequences,
                num_beams=num_beams,
            )

            output = self.tokenizer.batch_decode(
                generations,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )

            for input_kg, generations in zip(
                kg_batch, list(chunks(output, num_sequences))
            ):
                output_kg = input_kg.copy()
                output_kg.tails = generations
                outputs.append(output_kg)

        return KnowledgeGraph(outputs)
