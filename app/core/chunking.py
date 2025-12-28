from chonkie import RecursiveChunker, RecursiveRules, RecursiveLevel
from typing import Union, Callable, Any

def get_medical_chunker(
    chunk_size: int = 128, 
    min_characters_per_chunk: int = 24,
    tokenizer: Union[str, Callable, Any] = "character"
) -> RecursiveChunker:
    
    medical_rules = RecursiveRules(
        levels=[
            # Single level with custom delimiters list
            # Order: longer sequences first to prevent premature splitting
            RecursiveLevel(
                delimiters=["\n\n**", "\n\n", "\n-", ". ", "; ", ";", "\n", " "], 
                include_delim="prev"
            )
        ]
    )

    return RecursiveChunker(
        tokenizer=tokenizer,
        chunk_size=chunk_size,
        rules=medical_rules,
        min_characters_per_chunk=min_characters_per_chunk
    )

