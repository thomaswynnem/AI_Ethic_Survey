import torch
from acl_anthology import Anthology
from transformers import AutoTokenizer
from adapters import AutoAdapterModel
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import vstack
import numpy as np
from typing import List

class HybridRetrievalModel:
    def __init__(self, model_name='allenai/specter2_base'):
        # Ensure GPU is available
        if not torch.cuda.is_available():
            raise RuntimeError("❌ No GPU available. Please run on a CUDA-enabled machine.")

        # Use the first CUDA device
        self.device = torch.device("cuda:0")

        # Load model and tokenizer on GPU
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoAdapterModel.from_pretrained(model_name).to(self.device)

        # 'stop_words="english"' handles the required standard stopword removal
        self.vectorizer = TfidfVectorizer(stop_words='english')

        # In __init__ after loading model
        for name, param in self.model.named_parameters():
            if param.device != self.device:
                print(f"⚠️ Parameter {name} on {param.device}, expected {self.device}")

    def get_z_dense_helper(self, text_batch: List[str]):

        # Preprocess input, move tensors to GPU
        inputs = self.tokenizer(
            text_batch, padding=True, truncation=True,
            return_tensors="pt", return_token_type_ids=False, max_length=512
        ).to(self.device)

        # print({k: v.device for k, v in inputs.items()})

        with torch.no_grad():
            output = self.model(**inputs)
        embeddings = output.last_hidden_state[:, 0, :]

        return embeddings
    
    def get_z_dense(self, query: List[str]):
        # note that if reload the same adapter, the embedding will still change
        self.model.load_adapter(
            "allenai/specter2_adhoc_query", source="hf",
            load_as="specter2_adhoc_query", set_active=True
        )

        self.model.to(self.device)  # Move adapter to GPU

        return self.get_z_dense_helper(query)

    def get_z_dense_in_batches(self, text_batch: List[str], batch_size=32):
        self.model.load_adapter(
            "allenai/specter2", source="hf",
            load_as="specter2_proximity", set_active=True
        )

        self.model.to(self.device)  # Move adapter to GPU

        all_embeddings = []
        for i in range(0, len(text_batch), batch_size):
            batch = text_batch[i:i+batch_size]
            emb = self.get_z_dense_helper(batch)
            all_embeddings.append(emb.cpu())
            torch.cuda.empty_cache()
        return torch.cat(all_embeddings, dim=0)

    # Generates the sparse TF/IDF vector (z_sparse(D)) for a given document.
    def get_z_sparse(self, text_batch: List[str]):
        z_sparse = self.vectorizer.transform(text_batch) # z_sparse is sparse matrix, not tensor
        return z_sparse

    def get_z_sparse_in_batches(self, text_batch: List[str], batch_size=32):
        # Transform in batches to avoid memory issues
        all_z_sparse = []
        for i in range(0, len(text_batch), batch_size):
            batch = text_batch[i:i+batch_size]
            z_sparse = self.vectorizer.transform(batch)
            all_z_sparse.append(z_sparse)
            torch.cuda.empty_cache()  # Clear GPU cache if needed

        # Concatenate sparse matrices
        return vstack(all_z_sparse)

    def get_similarity_score(self, lambda_param: float, query: List[str], document: List[str], 
                            dense_query_embedding=None, dense_document_embeddings=None):
        # Use pre-computed embeddings if available, otherwise compute them
        # since each time reloading the adapter will lead to different embeddings, to compare SPECTOR2 and Hybrid model using the same dense embedding, 
        # we can use pre-computed dense embedding
        if dense_query_embedding is None:
            z_dense_query = self.get_z_dense(query)
        else:
            z_dense_query = dense_query_embedding
            
        if dense_document_embeddings is None:
            z_dense_document = self.get_z_dense_in_batches(document)
        else:
            z_dense_document = dense_document_embeddings
        
        # Fit vectorizer on combined corpus to ensure same vocabulary
        combined_corpus = query + document
        self.vectorizer.fit(combined_corpus)
        
        z_sparse_query = self.get_z_sparse(query)
        z_sparse_document = self.get_z_sparse_in_batches(document)

        print(cosine_similarity(z_sparse_query, z_sparse_document))

        return lambda_param * cosine_similarity(z_dense_query.cpu(), z_dense_document.cpu()) + (1 - lambda_param) * cosine_similarity(z_sparse_query, z_sparse_document)



# --------------------------
# Example usage
# --------------------------
anthology = Anthology.from_repo()
hybrid_model = HybridRetrievalModel()

query = ["AI ethics and considerations for the future of mankind"]
query_embedding = hybrid_model.get_z_dense(query)

# Pull papers
papers = list(anthology.papers())[0:100] # for debugging reason, use first 100
text_papers_batch = [
    (str(d.title) or "") + hybrid_model.tokenizer.sep_token + (str(d.abstract) or "")
    for d in papers
]
paper_embeddings = hybrid_model.get_z_dense_in_batches(text_papers_batch)

topk = 20

# Calculate similarity
similarity_score = cosine_similarity(query_embedding.cpu(), paper_embeddings.cpu()).flatten()
top_indices = np.argsort(-similarity_score)[:topk]

print("Top 20 results for the SPECTOR model")
for rank, idx in enumerate(top_indices, start=1):
    paper = papers[idx]
    print(f"{rank}. {paper.title}\n   Distance: {similarity_score[idx]:.4f}\n")

# use the hybrid model
print("Top 20 results for the hybrid model")
# Pass pre-computed embeddings to ensure consistency with SPECTOR model
similarity_score = hybrid_model.get_similarity_score(
    0.8, query, text_papers_batch, 
    dense_query_embedding=query_embedding, 
    dense_document_embeddings=paper_embeddings
).flatten()
top_indices = np.argsort(-similarity_score)[:topk]

for rank, idx in enumerate(top_indices, start=1):
    paper = papers[idx]
    print(f"{rank}. {paper.title}\n   Distance: {similarity_score[idx]:.4f}\n")