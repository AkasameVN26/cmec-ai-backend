## 2024-05-22 - Initial Entry
**Learning:** Initialized Bolt's journal.
**Action:** Always check this file for past learnings.

## 2024-05-22 - Optimize BM25 Index Creation
**Learning:** Repeatedly initializing `BM25Okapi` inside a loop is extremely expensive as it re-tokenizes the entire corpus. Lifting it out of the loop provided a ~50x speedup in ranking tasks.
**Action:** Always check for expensive object initializations inside loops, especially those involving text processing or indexing.

## 2024-05-22 - Batch Reranking Inference
**Learning:** Sequential inference calls to a reranker model (even small ones like 0.6B) incur significant overhead due to Python execution and GPU kernel launches. Batching these requests reduced the number of forward passes from N (queries) to ceil(N/BatchSize), improving throughput.
**Action:** Always look for opportunities to batch model inference calls when processing lists of independent items.
