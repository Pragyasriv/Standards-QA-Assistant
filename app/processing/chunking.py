"""Split structural blocks into citation-sized chunks.

Paragraphs stay intact until a block itself exceeds the maximum size.
Overlap is copied only inside the same article or annex.
"""

from __future__ import annotations

import re

from app.documents.models import DocumentRecord
from app.processing.models import ChunkRecord
from app.processing.settings import ProcessingSettings
from app.processing.structure import Block
from app.processing.tokenize import Tokenizer

_SENTENCE = re.compile(r"(?<=[.;:])\s+")


def build_chunks(
    blocks: list[Block],
    record: DocumentRecord,
    settings: ProcessingSettings,
    tokenizer: Tokenizer,
) -> list[ChunkRecord]:
    """Create deterministic chunks that cover every block."""
    pieces = _split_large_blocks(blocks, settings.chunk_max_tokens, tokenizer)
    grouped = _group_pieces(pieces, settings, tokenizer)
    chunks: list[ChunkRecord] = []
    previous_parent: tuple | None = None
    previous_text = ""
    for index, group in enumerate(grouped, start=1):
        parent = _parent_key(group[0])
        body = "\n\n".join(block.text for block in group if block.text)
        text = body
        if previous_parent == parent and previous_text and settings.chunk_overlap_tokens:
            overlap = tokenizer.tail(previous_text, settings.chunk_overlap_tokens)
            if overlap and overlap not in body:
                text = f"{overlap}\n\n{body}"
                while tokenizer.count(text) > settings.chunk_max_tokens and overlap:
                    overlap = tokenizer.tail(overlap, max(tokenizer.count(overlap) // 2, 1))
                    if tokenizer.count(overlap) <= 1:
                        text = body
                        break
                    text = f"{overlap}\n\n{body}"
        if tokenizer.count(text) > settings.chunk_max_tokens:
            text = body
        first = group[0]
        last = group[-1]
        chunks.append(
            ChunkRecord(
                chunk_id=f"{record.document_id}:{index:05d}",
                document_id=record.document_id,
                regulation=record.regulation_name,
                document_title=record.document_title,
                regulation_number=record.regulation_number,
                consolidation_date=record.version_date,
                chapter=first.chapter,
                title=first.title,
                article=first.article,
                paragraph=first.paragraph,
                subparagraph=first.subparagraph,
                annex=first.annex,
                annex_section=first.annex_section,
                page_start=min(block.page_start for block in group),
                page_end=max(block.page_end for block in group),
                source_filename=record.local_filename,
                text=text,
                token_count=tokenizer.count(text),
            )
        )
        previous_parent = parent
        previous_text = body
    return chunks


def _group_pieces(
    blocks: list[Block],
    settings: ProcessingSettings,
    tokenizer: Tokenizer,
) -> list[list[Block]]:
    groups: list[list[Block]] = []
    current: list[Block] = []
    current_tokens = 0
    for block in blocks:
        block_tokens = tokenizer.count(block.text)
        if current and _citation_key(current[0]) != _citation_key(block):
            groups.append(current)
            current = []
            current_tokens = 0
        if current and current_tokens + block_tokens > settings.chunk_max_tokens:
            groups.append(current)
            current = []
            current_tokens = 0
        elif current and current_tokens >= settings.chunk_target_tokens:
            groups.append(current)
            current = []
            current_tokens = 0
        current.append(block)
        current_tokens += block_tokens
    if current:
        groups.append(current)
    return groups


def _split_large_blocks(
    blocks: list[Block],
    max_tokens: int,
    tokenizer: Tokenizer,
) -> list[Block]:
    pieces: list[Block] = []
    for block in blocks:
        if tokenizer.count(block.text) <= max_tokens:
            pieces.append(block)
            continue
        for part in _split_text(block.text, max_tokens, tokenizer):
            pieces.append(
                Block(
                    kind=block.kind,
                    lines=[part],
                    page_start=block.page_start,
                    page_end=block.page_end,
                    title=block.title,
                    chapter=block.chapter,
                    article=block.article,
                    article_heading=block.article_heading,
                    paragraph=block.paragraph,
                    subparagraph=block.subparagraph,
                    annex=block.annex,
                    annex_section=block.annex_section,
                )
            )
    return pieces


def _split_text(text: str, max_tokens: int, tokenizer: Tokenizer) -> list[str]:
    sentences = [part.strip() for part in _SENTENCE.split(text) if part.strip()]
    if len(sentences) <= 1:
        sentences = text.split()
        return _pack_words(sentences, max_tokens, tokenizer) or [text]
    packed: list[str] = []
    current: list[str] = []
    for sentence in sentences:
        if tokenizer.count(sentence) > max_tokens:
            if current:
                packed.append(" ".join(current))
                current = []
            packed.extend(_pack_words(sentence.split(), max_tokens, tokenizer))
            continue
        candidate = " ".join(current + [sentence])
        if current and tokenizer.count(candidate) > max_tokens:
            packed.append(" ".join(current))
            current = [sentence]
        else:
            current.append(sentence)
    if current:
        packed.append(" ".join(current))
    return packed


def _pack_words(words: list[str], max_tokens: int, tokenizer: Tokenizer) -> list[str]:
    packed: list[str] = []
    current: list[str] = []
    for word in words:
        candidate = " ".join(current + [word])
        if current and tokenizer.count(candidate) > max_tokens:
            packed.append(" ".join(current))
            current = [word]
        else:
            current.append(word)
    if current:
        packed.append(" ".join(current))
    return packed


def _parent_key(block: Block) -> tuple:
    return (
        block.annex,
        block.title,
        block.chapter,
        block.article,
    )


def _citation_key(block: Block) -> tuple:
    """Keep a chunk inside one paragraph and one point when those are known."""
    return (
        *_parent_key(block),
        block.paragraph,
        block.subparagraph,
    )
