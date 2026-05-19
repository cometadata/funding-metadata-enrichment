from typing import List, Tuple


def build_chunks(*, token_count: int, max_tok: int, stride: int) -> List[Tuple[int, int]]:
    """Reproduce the chunking loop from the reference snippet exactly.

    Short docs (<= max_tok) get a single [0, token_count] chunk; longer docs
    get overlapping (start, start + max_tok) windows stepped by `stride`. The
    loop terminates the first time a window reaches token_count.
    """
    if token_count <= max_tok:
        return [(0, token_count)]

    chunks: List[Tuple[int, int]] = []
    for st in range(0, token_count, stride):
        en = min(st + max_tok, token_count)
        chunks.append((st, en))
        if en == token_count:
            break
    return chunks
