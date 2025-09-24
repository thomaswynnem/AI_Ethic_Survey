import torch
from acl_anthology import Anthology
from transformers import AutoTokenizer
from adapters import AutoAdapterModel
from sklearn.metrics.pairwise import euclidean_distances
from typing import List

class Specter2:
    def __init__(self, model_name='allenai/specter2_base'):
        # Ensure GPU is available
        if not torch.cuda.is_available():
            raise RuntimeError("❌ No GPU available. Please run on a CUDA-enabled machine.")

        # Use the first CUDA device
        self.device = torch.device("cuda:0")

        # Load model and tokenizer on GPU
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoAdapterModel.from_pretrained(model_name).to(self.device)
        
        # In __init__ after loading model
        for name, param in self.model.named_parameters():
            if param.device != self.device:
                print(f"⚠️ Parameter {name} on {param.device}, expected {self.device}")

    def embed_input(self, text_batch: List[str]):
        
        # Preprocess input, move tensors to GPU
        inputs = self.tokenizer(
            text_batch, padding=True, truncation=True,
            return_tensors="pt", return_token_type_ids=False, max_length=512
        ).to(self.device)
        
        print({k: v.device for k, v in inputs.items()})

        with torch.no_grad():
            output = self.model(**inputs)
        embeddings = output.last_hidden_state[:, 0, :]

        return embeddings


def embed_in_batches(model, texts, batch_size=32):
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        emb = model.embed_input(batch)   # already on GPU
        all_embeddings.append(emb.cpu()) # move to CPU after each batch
        torch.cuda.empty_cache()         # free unused GPU memory
    return torch.cat(all_embeddings, dim=0)
# --------------------------
# Example usage
# --------------------------
anthology = Anthology.from_repo()
specter2 = Specter2()


# Load adapters
specter2.model.load_adapter(
    "allenai/specter2_adhoc_query", source="hf",
    load_as="specter2_adhoc_query", set_active=True
)
specter2.model.to(specter2.device)   # <--- move adapters to GPU



query = ["AI ethics and considerations for the future of mankind"]
query_embedding = specter2.embed_input(query)

specter2.model.load_adapter(
    "allenai/specter2", source="hf",
    load_as="specter2_proximity", set_active=True
)
specter2.model.to(specter2.device)   # <--- move adapters to GPU


# Pull papers
papers = list(anthology.papers())
text_papers_batch = [
    (str(d.title) or "") + specter2.tokenizer.sep_token + (str(d.abstract) or "")
    for d in papers
]
paper_embeddings = embed_in_batches(specter2, text_papers_batch, batch_size=32)

# Calculate L2 distances
l2_distance = euclidean_distances(paper_embeddings.cpu(), query_embedding.cpu()).flatten()

import numpy as np

# Get indices of the 20 smallest distances
topk = 20
top_indices = np.argsort(l2_distance)[:topk]

# Print results
for rank, idx in enumerate(top_indices, start=1):
    paper = papers[idx]
    print(f"{rank}. {paper.title}\n   Distance: {l2_distance[idx]:.4f}\n")