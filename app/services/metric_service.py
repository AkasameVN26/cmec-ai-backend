import time
import torch
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer
from underthesea import word_tokenize
import pysbd
from fastapi.concurrency import run_in_threadpool

from app.models.schemas import SourceSegment
from typing import List

class MetricService:
    def __init__(self):
        self.tokenizer = None
        self.embed_model = None
        self.scorer = None
        self.embedding_model_name = "Qwen/Qwen3-Embedding-0.6B"
        self.tokenizer_name = "arcee-ai/Arcee-VyLinh"
        self._is_loaded = False
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.segmenter = pysbd.Segmenter(language="en", clean=False)

    def load_models(self):
        if self._is_loaded:
            return

        print(f"[{self.__class__.__name__}] Loading models on DEVICE: {self.device.upper()}...")
        start_t = time.time()

        print("Initializing ROUGE scorer...")
        self.scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=False)

        print(f"Initializing Tokenizer ({self.tokenizer_name})...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
        except Exception as e:
            print(f"Warning: Could not load tokenizer {self.tokenizer_name}: {e}")
            self.tokenizer = None

        print(f"Loading embedding model: {self.embedding_model_name}...")
        try:
            self.embed_model = SentenceTransformer(
                self.embedding_model_name, 
                trust_remote_code=True, 
                device=self.device,
                model_kwargs={"dtype": torch.float16 if self.device == "cuda" else torch.float32}
            )
        except Exception as e:
            print(f"Warning: Could not load embedding model: {e}")
            self.embed_model = None
        
        self._is_loaded = True
        print(f"[{self.__class__.__name__}] Models loaded in {time.time() - start_t:.2f}s")

    def _explain_summary_sync(self, source_segments: List[SourceSegment], summary_text: str):
        if not self._is_loaded:
            self.load_models()
        
        # 1. Segmentation using pysbd directly
        try:
            # Process Summary
            summary_sents = self.segmenter.segment(summary_text) if summary_text.strip() else []
            
            # Flatten Source Segments into Sentences
            flattened_source_segments = []
            for seg in source_segments:
                seg_sents = self.segmenter.segment(seg.content) if seg.content.strip() else []
                
                for s in seg_sents:
                    flattened_source_segments.append(SourceSegment(
                        content=s,
                        source_type=seg.source_type,
                        source_id=seg.source_id
                    ))

        except Exception as e:
            print(f"Error in sentence tokenization: {e}")
            return {"error": f"Tokenization error: {str(e)}"}
        
        if not flattened_source_segments or not summary_sents:
            return {
                "notes": flattened_source_segments,
                "summary_sentences": summary_sents,
                "matches": [],
                "error": "Source or Summary is empty after tokenization"
            }

        # 2. Embedding
        if not self.embed_model:
             return {"error": "Embedding model not loaded"}

        segment_contents = [s.content for s in flattened_source_segments]
        
        # Encode (Symmetric encoding as per review_interface, no prompt_name="query" for simple comparison)
        source_embeddings = self.embed_model.encode(segment_contents, convert_to_tensor=True, normalize_embeddings=True)
        summary_embeddings = self.embed_model.encode(summary_sents, convert_to_tensor=True, normalize_embeddings=True)

        # 3. Similarity Matrix Calculation
        similarity_matrix = util.cos_sim(summary_embeddings, source_embeddings)
        
        # 4. Context-Aware Retrieval (Best Match + Neighbors)
        result = {
            "notes": flattened_source_segments,
            "summary_sentences": summary_sents,
            "matches": [],
            "low_similarity_matches": []
        }

        LOW_SIMILARITY_THRESHOLD = 0.5 
        CONTEXT_THRESHOLD = 0.5 # Threshold for including neighbors

        for i in range(len(summary_sents)):
            scores = similarity_matrix[i]
            
            # Find Best Match
            best_score = torch.max(scores).item()
            best_idx = torch.argmax(scores).item()

            # Initialize with Best Match
            current_match_indices = [best_idx]
            current_match_scores = [best_score]

            # --- Context Window Logic ---
            # Check PREVIOUS neighbor in flattened list
            if best_idx > 0:
                prev_idx = best_idx - 1
                prev_score = scores[prev_idx].item()
                if prev_score > CONTEXT_THRESHOLD:
                    current_match_indices.insert(0, prev_idx)
                    current_match_scores.insert(0, prev_score)
            
            # Check NEXT neighbor in flattened list
            if best_idx < len(flattened_source_segments) - 1:
                next_idx = best_idx + 1
                next_score = scores[next_idx].item()
                if next_score > CONTEXT_THRESHOLD:
                    current_match_indices.append(next_idx)
                    current_match_scores.append(next_score)

            match_detail = {
                "summary_idx": i,
                "source_indices": current_match_indices,
                "scores": current_match_scores
            }
            result["matches"].append(match_detail)

            if best_score < LOW_SIMILARITY_THRESHOLD:
                result["low_similarity_matches"].append(match_detail)
            
        # Calculate average similarity score
        best_scores_all, _ = torch.max(similarity_matrix, dim=1)
        avg_score = torch.mean(best_scores_all).item()
        result["avg_similarity_score"] = avg_score
            
        return result

    async def explain_summary(self, source_segments: List[SourceSegment], summary_text: str):
        return await run_in_threadpool(self._explain_summary_sync, source_segments, summary_text)

    def _calculate_metrics_sync(self, source_text: str, generated_text: str, duration: float = 0):
        if not self._is_loaded:
            self.load_models()
        
        start_calc = time.time()
        print(f"[{self.__class__.__name__}] Calculating metrics...")

        metrics = {
            "duration": duration,
            "input_tokens": 0,
            "output_tokens": 0,
            "ratio": 0.0,
            "similarity_score": 0.0,
            "rouge1": {"f1": 0.0, "p": 0.0, "r": 0.0},
            "rouge2": {"f1": 0.0, "p": 0.0, "r": 0.0},
            "rougeL": {"f1": 0.0, "p": 0.0, "r": 0.0}
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
            # Encoding can be heavy
            emb_summary = self.embed_model.encode([generated_text], prompt_name="query", convert_to_tensor=True, normalize_embeddings=True)
            emb_source = self.embed_model.encode([source_text], convert_to_tensor=True, normalize_embeddings=True)
            similarity_score = util.cos_sim(emb_summary, emb_source).item()
            metrics["similarity_score"] = similarity_score

        # ROUGE Scores
        if self.scorer:
            ref_tokens = word_tokenize(source_text, format="text")
            hyp_tokens = word_tokenize(generated_text, format="text")
            scores = self.scorer.score(ref_tokens, hyp_tokens)
            
            metrics["rouge1"] = {
                "f1": scores['rouge1'].fmeasure,
                "p": scores['rouge1'].precision,
                "r": scores['rouge1'].recall
            }
            metrics["rouge2"] = {
                "f1": scores['rouge2'].fmeasure,
                "p": scores['rouge2'].precision,
                "r": scores['rouge2'].recall
            }
            metrics["rougeL"] = {
                "f1": scores['rougeL'].fmeasure,
                "p": scores['rougeL'].precision,
                "r": scores['rougeL'].recall
            }
        
        calc_time = time.time() - start_calc
        print(f"[{self.__class__.__name__}] Metrics calculated in {calc_time:.2f}s")
        return metrics

    async def calculate_metrics(self, source_text: str, generated_text: str, duration: float = 0):
        return await run_in_threadpool(self._calculate_metrics_sync, source_text, generated_text, duration)

metric_service = MetricService()
