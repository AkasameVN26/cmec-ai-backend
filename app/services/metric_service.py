import time
import torch
import gc
import numpy as np
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModelForCausalLM
from rank_bm25 import BM25Okapi
from underthesea import word_tokenize
from chonkie import RecursiveChunker
from fastapi.concurrency import run_in_threadpool

from app.models.schemas import SourceSegment
from typing import List, Dict, Tuple

class MetricService:
    def __init__(self):
        self.tokenizer = None
        self.embed_model = None
        self.reranker_model = None
        self.reranker_tokenizer = None
        
        self.embedding_model_name = "Qwen/Qwen3-Embedding-0.6B"
        self.reranker_model_name = "Qwen/Qwen3-Reranker-0.6B"
        # We can use the reranker's tokenizer for the reranking task specifically
        
        self.tokenizer_name = "arcee-ai/Arcee-VyLinh" # Keep for metrics if needed
        
        self._is_loaded = False
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.source_chunker = None 
        self.summary_chunker = None

        # Custom Instruction for Medical Retrieval (Qwen Best Practice)
        self.retrieval_instruction = "Given a claim from a medical summary, retrieve relevant clinical evidence, lab results, and doctor notes from the patient record."
        self.query_prompt = f"Instruct: {self.retrieval_instruction}\nQuery: "

    def load_models(self):
        if self._is_loaded:
            return

        print(f"[{self.__class__.__name__}] Loading models into RAM (CPU)...")
        start_t = time.time()

        print(f"Initializing Tokenizer ({self.tokenizer_name})...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
        except Exception as e:
            print(f"Warning: Could not load tokenizer {self.tokenizer_name}: {e}")
            self.tokenizer = None

        # Initialize Chonkie with default rules (Basic Chunker)
        try:
            print("Initializing RecursiveChunkers (Source: 64, Summary: 16)...")
            base_tokenizer = self.tokenizer if self.tokenizer else "gpt2"
            self.source_chunker = RecursiveChunker(
                tokenizer=base_tokenizer,
                chunk_size=128,
                min_characters_per_chunk=12
            )
            self.summary_chunker = RecursiveChunker(
                tokenizer=base_tokenizer,
                chunk_size=32,
                min_characters_per_chunk=12
            )
        except Exception as e:
            print(f"Warning: Could not initialize RecursiveChunkers: {e}")
            self.source_chunker = None
            self.summary_chunker = None

        print(f"Loading embedding model: {self.embedding_model_name}...")
        try:
            # Load to CPU initially
            self.embed_model = SentenceTransformer(
                self.embedding_model_name, 
                trust_remote_code=True, 
                device="cpu",
                model_kwargs={"dtype": torch.float16 if self.device == "cuda" else torch.float32}
            )
        except Exception as e:
            print(f"Warning: Could not load embedding model: {e}")
            self.embed_model = None
            
        print(f"Loading reranker model: {self.reranker_model_name}...")
        try:
            self.reranker_tokenizer = AutoTokenizer.from_pretrained(self.reranker_model_name, padding_side='left', trust_remote_code=True)
            # Load to CPU initially
            self.reranker_model = AutoModelForCausalLM.from_pretrained(
                self.reranker_model_name,
                dtype=torch.float16 if self.device == "cuda" else torch.float32,
                trust_remote_code=True
            ).to("cpu").eval()
        except Exception as e:
            print(f"Warning: Could not load reranker model: {e}")
            self.reranker_model = None
            self.reranker_tokenizer = None
        
        self._is_loaded = True
        print(f"[{self.__class__.__name__}] Models loaded in {time.time() - start_t:.2f}s")

    def _rank_bm25(self, query: str, bm25: BM25Okapi, top_k: int = 20) -> List[int]:
        """
        Retrieve Top-K candidates using BM25.
        Returns a list of indices.
        """
        tokenized_query = word_tokenize(query)
        
        # Get scores
        scores = bm25.get_scores(tokenized_query)
        # Get top_k indices
        top_indices = np.argsort(scores)[::-1][:top_k]
        return top_indices.tolist()

    def _rank_embedding(self, query: str, documents: List[str], top_k: int = 20, doc_embs: torch.Tensor = None) -> List[int]:
        """
        Retrieve Top-K candidates using Cosine Similarity (Qwen-Embedding).
        Returns a list of indices.
        Assumes embed_model is already on the correct device.
        """
        if not self.embed_model:
            return []
            
        # Embed query
        query_emb = self.embed_model.encode(query, prompt=self.query_prompt, convert_to_tensor=True, normalize_embeddings=True)

        # Embed docs if not provided
        if doc_embs is None:
            doc_embs = self.embed_model.encode(documents, convert_to_tensor=True, normalize_embeddings=True)
        
        # Calculate cosine similarity
        scores = util.cos_sim(query_emb, doc_embs)[0]
        
        # Get top_k indices
        top_vals, top_indices = torch.topk(scores, k=min(top_k, len(documents)))
        return top_indices.tolist()

    def _rrf_fusion(self, rank_lists: List[List[int]], k: int = 60, top_k: int = 10) -> List[int]:
        """
        Apply Reciprocal Rank Fusion (RRF).
        """
        rrf_score = {}
        
        for rank_list in rank_lists:
            for rank, doc_idx in enumerate(rank_list):
                if doc_idx not in rrf_score:
                    rrf_score[doc_idx] = 0
                rrf_score[doc_idx] += 1 / (k + rank + 1)
        
        # Sort by RRF score descending
        sorted_docs = sorted(rrf_score.items(), key=lambda x: x[1], reverse=True)
        
        return [doc_idx for doc_idx, score in sorted_docs[:top_k]]

    def _rerank_batch(self, queries: List[str], documents: List[str], all_candidate_indices: List[List[int]]) -> List[Tuple[int, float]]:
        """
        Batch Rerank candidates for multiple queries using Qwen3-Reranker-0.6B.
        Returns a list of (best_idx, best_score) for each query.
        """
        if not self.reranker_model or not self.reranker_tokenizer:
            return [(candidates[0] if candidates else -1, 0.0) for candidates in all_candidate_indices]

        # Prepare formatting constants
        prefix = "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n"
        suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        task = "Given a claim from a medical summary, determine if the Clinical Document provides evidence to support that claim."
        
        prefix_tokens = self.reranker_tokenizer.encode(prefix, add_special_tokens=False)
        suffix_tokens = self.reranker_tokenizer.encode(suffix, add_special_tokens=False)
        
        token_false_id = self.reranker_tokenizer.convert_tokens_to_ids("no")
        token_true_id = self.reranker_tokenizer.convert_tokens_to_ids("yes")
        max_length = 4096

        all_pairs = []
        
        for i, (query, candidates) in enumerate(zip(queries, all_candidate_indices)):
            if not candidates:
                continue
            
            for idx in candidates:
                doc = documents[idx]
                formatted_input = "<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}".format(
                    instruction=task, query=query, doc=doc
                )
                all_pairs.append(formatted_input)

        if not all_pairs:
             return [(candidates[0] if candidates else -1, 0.0) for candidates in all_candidate_indices]

        # Tokenize and Inference in Chunks
        BATCH_SIZE = 12
        all_scores = []
        
        for i in range(0, len(all_pairs), BATCH_SIZE):
            batch_pairs = all_pairs[i : i + BATCH_SIZE]

            # Tokenize
            inputs = self.reranker_tokenizer(
                batch_pairs, padding=False, truncation='longest_first',
                return_attention_mask=False, max_length=max_length - len(prefix_tokens) - len(suffix_tokens)
            )

             # Add prefix/suffix
            for j, ele in enumerate(inputs['input_ids']):
                inputs['input_ids'][j] = prefix_tokens + ele + suffix_tokens

            # Pad dynamically
            inputs = self.reranker_tokenizer.pad(inputs, padding=True, return_tensors="pt")

            # Move to device
            for key in inputs:
                inputs[key] = inputs[key].to(self.device)

            # Inference
            with torch.no_grad():
                logits = self.reranker_model(**inputs).logits[:, -1, :]
                true_vector = logits[:, token_true_id]
                false_vector = logits[:, token_false_id]
                stacked_logits = torch.stack([false_vector, true_vector], dim=1)
                log_probs = torch.nn.functional.log_softmax(stacked_logits, dim=1)
                scores = log_probs[:, 1].exp().tolist()
                all_scores.extend(scores)

        # Reconstruct results
        results = []
        current_pair_idx = 0

        for candidates in all_candidate_indices:
            if not candidates:
                results.append((-1, 0.0))
                continue

            # Get scores for this query
            num_candidates = len(candidates)
            query_scores = all_scores[current_pair_idx : current_pair_idx + num_candidates]
            current_pair_idx += num_candidates

            best_local_idx = np.argmax(query_scores)
            best_global_idx = candidates[best_local_idx]
            best_score = query_scores[best_local_idx]
            results.append((best_global_idx, best_score))

        return results

    def _explain_summary_sync(self, source_segments: List[SourceSegment], summary_text: str):
        # Force cleanup before loading heavy models to GPU
        gc.collect()
        torch.cuda.empty_cache()

        if not self._is_loaded:
            self.load_models()
        
        # 1. Segmentation
        try:
            if not self.summary_chunker or not self.source_chunker:
                # Fallback just in case load_models partially failed but didn't crash
                base_tokenizer = self.tokenizer if self.tokenizer else "gpt2"
                self.summary_chunker = RecursiveChunker(tokenizer=base_tokenizer, chunk_size=32, min_characters_per_chunk=12)
                self.source_chunker = RecursiveChunker(tokenizer=base_tokenizer, chunk_size=128, min_characters_per_chunk=12)

            # Chonkie returns Chunk objects, we need the text
            summary_sents = [c.text for c in self.summary_chunker.chunk(summary_text)] if summary_text.strip() else []
            
            flattened_source_segments = []
            for seg in source_segments:
                seg_sents = [c.text for c in self.source_chunker.chunk(seg.content)] if seg.content.strip() else []
                for s in seg_sents:
                    flattened_source_segments.append(SourceSegment(
                        content=s,
                        source_type=seg.source_type,
                        source_id=seg.source_id
                    ))
        except Exception as e:
            print(f"Error in chunking: {e}")
            return {"error": f"Chunking error: {str(e)}"}
        
        if not flattened_source_segments or not summary_sents:
            return {
                "notes": flattened_source_segments,
                "summary_sentences": summary_sents,
                "matches": [],
                "error": "Source or Summary is empty after tokenization"
            }

        documents = [s.content for s in flattened_source_segments]
        
        result = {
            "notes": flattened_source_segments,
            "summary_sentences": summary_sents,
            "matches": [],
            "low_similarity_matches": []
        }

        LOW_SIMILARITY_THRESHOLD = 0.7 

        # --- Embedding Calculation Batch ---
        # Move Embedding Model to GPU
        print("Moving Embedding Model to GPU...")
        if self.embed_model:
            self.embed_model.to(self.device)
        
        # Pre-calculate document embeddings ONCE
        doc_embs = None
        all_emb_scores = None

        if self.embed_model and documents:
            doc_embs = self.embed_model.encode(documents, convert_to_tensor=True, normalize_embeddings=True)

            # ⚡ Bolt Optimization: Batch encode all summary sentences at once
            if summary_sents:
                query_embs = self.embed_model.encode(summary_sents, prompt=self.query_prompt, convert_to_tensor=True, normalize_embeddings=True)
                # Compute all similarities in one matrix operation [num_queries, num_docs]
                all_emb_scores = util.cos_sim(query_embs, doc_embs)

        # Phase 1: Retrieve Candidates (BM25 + Embedding)

        # Build BM25 index ONCE
        print("Building BM25 index...")
        tokenized_corpus = [word_tokenize(doc) for doc in documents]
        bm25 = BM25Okapi(tokenized_corpus)

        all_top_candidates = []
        
        for i, query_sent in enumerate(summary_sents):
            # A. Hybrid Retrieval (BM25 + Embedding)
            bm25_indices = self._rank_bm25(query_sent, bm25, top_k=20)

            # Optimized Embedding Retrieval
            emb_indices = []
            if all_emb_scores is not None:
                # Extract scores for the current query sentence
                scores = all_emb_scores[i]
                # Get top-k indices efficiently
                top_vals, top_indices = torch.topk(scores, k=min(20, len(documents)))
                emb_indices = top_indices.tolist()
            else:
                # Fallback (should be covered by batching, but keeps safety)
                emb_indices = self._rank_embedding(query_sent, documents, top_k=20, doc_embs=doc_embs)
            
            # B. RRF Fusion
            top_candidates = self._rrf_fusion([bm25_indices, emb_indices], k=60, top_k=10)
            all_top_candidates.append(top_candidates)

        # Move Embedding Model back to CPU
        print("Moving Embedding Model to CPU...")
        if self.embed_model:
            self.embed_model.to("cpu")
        torch.cuda.empty_cache()
        
        # --- Reranking Batch ---
        # Move Reranker Model to GPU
        print("Moving Reranker Model to GPU...")
        if self.reranker_model:
            self.reranker_model.to(self.device)
            
        # Phase 2: Batch Rerank
        batch_results = self._rerank_batch(summary_sents, documents, all_top_candidates)

        for i, (best_idx, best_score) in enumerate(batch_results):
            if best_idx == -1 and not all_top_candidates[i]:
                 # handle empty matches
                 result["matches"].append({
                    "summary_idx": i,
                    "source_indices": [],
                    "scores": []
                })
                 continue
            
            # Construct Result
            match_detail = {
                "summary_idx": i,
                "source_indices": [best_idx],
                "scores": [best_score]
            }
            result["matches"].append(match_detail)

            if best_score < LOW_SIMILARITY_THRESHOLD:
                result["low_similarity_matches"].append(match_detail)

        # Move Reranker Model back to CPU
        print("Moving Reranker Model to CPU...")
        if self.reranker_model:
            self.reranker_model.to("cpu")
        gc.collect()
        torch.cuda.empty_cache()
            
        # Calculate average similarity score
        all_scores = [m["scores"][0] for m in result["matches"] if m["scores"]]
        avg_score = sum(all_scores) / len(all_scores) if all_scores else 0.0
        result["avg_similarity_score"] = avg_score
            
        return result

    async def explain_summary(self, source_segments: List[SourceSegment], summary_text: str):
        return await run_in_threadpool(self._explain_summary_sync, source_segments, summary_text)

    def _calculate_metrics_sync(self, source_text: str, generated_text: str, duration: float = 0):
        # Force cleanup before loading heavy models to GPU
        gc.collect()
        torch.cuda.empty_cache()

        if not self._is_loaded:
            self.load_models()
        
        start_calc = time.time()
        print(f"[{self.__class__.__name__}] Calculating metrics...")

        metrics = {
            "duration": duration,
            "input_tokens": 0,
            "output_tokens": 0,
            "ratio": 0.0,
            "similarity_score": 0.0
        }

        # Token Counts
        if self.tokenizer:
            input_tokens = len(self.tokenizer.encode(source_text))
            output_tokens = len(self.tokenizer.encode(generated_text))
            metrics["input_tokens"] = input_tokens
            metrics["output_tokens"] = output_tokens
            metrics["ratio"] = output_tokens / input_tokens if input_tokens > 0 else 0

        # Similarity Score
        if self.embed_model:
            print("Moving Embedding Model to GPU for metrics...")
            self.embed_model.to(self.device)
            try:
                # Encoding can be heavy
                emb_summary = self.embed_model.encode([generated_text], prompt=self.query_prompt, convert_to_tensor=True, normalize_embeddings=True)
                emb_source = self.embed_model.encode([source_text], convert_to_tensor=True, normalize_embeddings=True)
                similarity_score = util.cos_sim(emb_summary, emb_source).item()
                metrics["similarity_score"] = similarity_score
            finally:
                print("Moving Embedding Model back to CPU...")
                self.embed_model.to("cpu")
                gc.collect()
                torch.cuda.empty_cache()
        
        calc_time = time.time() - start_calc
        print(f"[{self.__class__.__name__}] Metrics calculated in {calc_time:.2f}s")
        return metrics

    async def calculate_metrics(self, source_text: str, generated_text: str, duration: float = 0):
        return await run_in_threadpool(self._calculate_metrics_sync, source_text, generated_text, duration)

metric_service = MetricService()
