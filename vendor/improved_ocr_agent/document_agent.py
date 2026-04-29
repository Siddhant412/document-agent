import os
import re
import json
import time
import math
import argparse
import requests
import numpy as np
from pathlib import Path
from typing import Any, List, Dict, Optional, Tuple
from dataclasses import dataclass, field

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError:
    TfidfVectorizer = None
    cosine_similarity = None

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from .hybrid_pdf_extractor import HybridPDFExtractor, PipelineCustomOCRBackend
from .pipeline_custom import make_ocr_args


# =============================================================================
# 0. EMBEDDING HELPERS
# =============================================================================

def try_get_embeddings(
    texts: List[str],
    embedding_type: str = "local",
    local_model: Any = None,
    base_url: str = "",
    api_key: str = "",
    api_model: str = "",
    batch_size: int = 32,
    timeout: float = 60.0,
    max_retries: int = 2,
) -> Optional[np.ndarray]:
    """
    Returns embeddings or None if embedding service/model is unavailable.
    This is intentionally fail-soft for production document search.
    """
    if not texts:
        return np.array([])

    try:
        if embedding_type == "api":
            endpoint = base_url.strip().rstrip("/")
            if not endpoint.endswith("/embeddings"):
                endpoint += "/embeddings" if endpoint.endswith("/v1") else "/v1/embeddings"

            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }

            all_embeddings: List[List[float]] = []
            total_batches = (len(texts) + batch_size - 1) // batch_size

            for batch_idx in range(total_batches):
                batch_start = batch_idx * batch_size
                batch_texts = texts[batch_start: batch_start + batch_size]
                payload = {"input": batch_texts, "model": api_model}

                ok = False
                for attempt in range(1, max_retries + 1):
                    try:
                        resp = requests.post(endpoint, headers=headers, json=payload, timeout=timeout)
                        if resp.status_code == 200:
                            data = sorted(resp.json()["data"], key=lambda x: x.get("index", 0))
                            all_embeddings.extend([item["embedding"] for item in data])
                            ok = True
                            break
                    except Exception:
                        pass

                    if attempt < max_retries:
                        time.sleep(2 ** attempt)

                if not ok:
                    return None

            return np.array(all_embeddings)

        else:
            if local_model is None:
                return None
            return local_model.encode(texts)

    except Exception:
        return None


# =============================================================================
# 1. GLOBAL REGISTRY
# =============================================================================

_active_pipelines: Dict[str, Any] = {}


def register_pipeline(name: str, pipeline: Dict[str, Any]):
    _active_pipelines[name] = pipeline
    print(f"[DocumentAgent] Registered pipeline '{name}'")


def unregister_pipeline(name: str):
    _active_pipelines.pop(name, None)


def get_active_document_agent() -> Optional["DocumentAgent"]:
    for pipeline in _active_pipelines.values():
        agent = pipeline.get("document_agent")
        if agent is not None:
            return agent
    return None


# =============================================================================
# 2. DATA STRUCTURES
# =============================================================================

@dataclass
class SectionNode:
    id: str
    title: str
    level: int
    content: str = ""
    pages: set = field(default_factory=set)
    children: List["SectionNode"] = field(default_factory=list)
    parent_id: Optional[str] = None
    preview: str = ""


@dataclass
class ChunkRecord:
    chunk_id: str
    text: str
    page_start: Optional[int]
    page_end: Optional[int]
    section_id: str
    section_title: str
    metadata: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# 3. EXTRACTOR WRAPPER
# =============================================================================

class MarkdownDocumentExtractor:
    """
    Wraps HybridPDFExtractor and returns markdown.
    """

    def __init__(
        self,
        pdf_path: str,
        ocr_server: Optional[str] = None,
        ocr_model: str = "allenai/olmOCR-2-7B-1025-FP8",
        ocr_workspace: str = "./tmp_ocr",
        use_pdf_page_ocr: bool = False,
    ):
        self.pdf_path = str(Path(pdf_path).resolve())
        self.ocr_server = ocr_server
        self.ocr_model = ocr_model
        self.ocr_workspace = ocr_workspace
        self.use_pdf_page_ocr = use_pdf_page_ocr

    def extract_to_markdown(self) -> str:
        if self.ocr_server:
            ocr_args = make_ocr_args(
                server=self.ocr_server,
                model=self.ocr_model,
                workspace=self.ocr_workspace,
                embed_page_markers=True,
            )
            backend = PipelineCustomOCRBackend(ocr_args)
        else:
            backend = None

        extractor = HybridPDFExtractor(
            pdf_path=self.pdf_path,
            ocr_backend=backend,
            use_pdf_page_ocr=self.use_pdf_page_ocr,
        )
        return extractor.extract_to_markdown()


# =============================================================================
# 4. SECTION INDEX
# =============================================================================

class SectionIndex:
    """
    Builds section tree from markdown.
    Assumes markdown may contain:
      - headings (#, ##, ### ...)
      - page markers like <!-- page 12 mode: hybrid_paper -->
    """

    PAGE_RE = re.compile(r"page\s+(\d+)", re.IGNORECASE)

    def __init__(self, markdown_text: str):
        self.raw_text = markdown_text
        self.sections: Dict[str, SectionNode] = {}
        self.tree: List[SectionNode] = []
        self._build()

    def _extract_page_number(self, line: str) -> Optional[int]:
        m = self.PAGE_RE.search(line)
        if m:
            return int(m.group(1))
        return None

    def _clean_preview_text(self, text: str, max_len: int = 180) -> str:
        text = re.sub(r"\s+", " ", text).strip()
        if not text:
            return ""

        words = text.split()
        filtered = []
        for w in words:
            # filter very short noisy tokens
            if len(w) == 1 and not re.match(r"^[A-Za-z0-9]$", w):
                continue
            filtered.append(w)

        text = " ".join(filtered)
        return text[:max_len] + ("..." if len(text) > max_len else "")

    def _build(self):
        lines = self.raw_text.splitlines()

        root = SectionNode(id="root", title="Document Root", level=0)
        self.sections[root.id] = root
        self.tree.append(root)

        stack: List[SectionNode] = [root]
        current_section = root
        section_counter = 1
        current_page = 1

        for line in lines:
            page_num = self._extract_page_number(line)
            if page_num is not None:
                current_page = page_num
                continue

            heading_match = re.match(r"^(#{1,6})\s+(.*)$", line.strip())

            pseudo_heading = re.match(
                r"^(abstract|introduction|background|related work|method|methods|methodology|approach|experiments|results|discussion|conclusion|conclusions|references|appendix|limitations|acknowledgments|acknowledgements)\.?\s*$",
                line.strip(),
                flags=re.IGNORECASE,
            )

            numbered_heading = re.match(
                r"^(\d+(\.\d+)*|[A-Z]|[IVXLC]+)[\.\)]?\s+[A-Z].*$",
                line.strip()
            )

            if heading_match or pseudo_heading or numbered_heading:
                if heading_match:
                    level = len(heading_match.group(1))
                    title = heading_match.group(2).strip()
                else:
                    level = 2
                    title = line.strip().rstrip(".").strip()

                sec_id = f"sec_{section_counter}"
                section_counter += 1

                node = SectionNode(
                    id=sec_id,
                    title=title,
                    level=level,
                    pages={current_page},
                )

                while stack and stack[-1].level >= level:
                    stack.pop()

                parent = stack[-1] if stack else root
                node.parent_id = parent.id
                parent.children.append(node)
                self.sections[sec_id] = node
                stack.append(node)
                current_section = node
            else:
                text = line.strip()
                if not text or text == "---":
                    continue
                if re.fullmatch(r"\d{1,4}", text):
                    continue
                current_section.content += line + "\n"
                current_section.pages.add(current_page)
                
                
        for sec_id, sec in self.sections.items():
            if sec_id == "root":
                continue
            first_para = self._first_paragraph(sec.content)
            sec.preview = self._clean_preview_text(first_para)

    def _first_paragraph(self, text: str) -> str:
        paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        return paras[0] if paras else text.strip()

    @staticmethod
    def _section_total_chars(node: "SectionNode") -> int:
        """Recursively sum content chars for a node and all its descendants."""
        total = len(node.content)
        for child in node.children:
            total += SectionIndex._section_total_chars(child)
        return total

    def get_overview(self) -> str:
        lines = ["Document Overview:"]
        all_nodes = [n for n in self._flatten_tree() if n.id != "root"]

        # Compute per-section token estimates (content of each leaf shown)
        # and grand total across ALL leaf sections so the model can plan.
        total_chars = sum(len(n.content) for n in all_nodes)
        total_tok = max(1, total_chars // 4)

        for node in all_nodes:
            pages = sorted(node.pages)
            page_str = f"p.{pages[0]}" if len(pages) == 1 else f"p.{pages[0]}-{pages[-1]}"
            indent = "  " * max(node.level - 1, 0)
            preview = f" | Preview: {node.preview}" if node.preview else ""
            # Include descendant content in the size so the model knows how
            # much it will receive when it calls read_document_section.
            sec_chars = self._section_total_chars(node)
            sec_tok = max(1, sec_chars // 4)
            tok_str = f"~{sec_tok/1000:.1f}k tok" if sec_tok >= 1000 else f"~{sec_tok} tok"
            lines.append(
                f"{indent}- [{node.id}] {node.title} ({page_str}, {tok_str}){preview}"
            )

        # Grand-total summary + reading strategy guidance
        grand_str = f"~{total_tok/1000:.1f}k" if total_tok >= 1000 else f"~{total_tok}"
        lines.append("")
        lines.append(f"📊 Total document content: {grand_str} tokens across {len(all_nodes)} sections.")

        if total_tok <= 4000:
            lines.append(
                "✅ Strategy: Document is SHORT. "
                "Use parallel tool calls — include multiple `read_document_section` calls "
                "in a single response to read all sections at once."
            )
        elif total_tok <= 12000:
            lines.append(
                "⚠️  Strategy: Document is MEDIUM. "
                "Group sections into 2-3 batches of parallel `read_document_section` calls. "
                "Avoid reading the entire document in one turn to prevent context overflow."
            )
        else:
            lines.append(
                "🚨 Strategy: Document is LARGE (risk of context overflow). "
                "You MUST use `execute_parallel_branches` to divide reading work across branches. "
                "Each branch should read a subset of sections independently, "
                "then synthesize the combined results into the final output. "
                "Do NOT read sections one-by-one sequentially — "
                "the accumulated context WILL overflow before you finish."
            )

        return "\n".join(lines)

    def _flatten_tree(self) -> List[SectionNode]:
        out: List[SectionNode] = []

        def walk(node: SectionNode):
            out.append(node)
            for child in node.children:
                walk(child)

        for root in self.tree:
            walk(root)
        return out

    def read_details(self, section_id: str) -> str:
        if section_id not in self.sections:
            return f"Error: Section ID not found: {section_id}"

        sec = self.sections[section_id]
        pages = sorted(sec.pages)
        page_str = f"Page {pages[0]}" if len(pages) == 1 else f"Pages {pages[0]}-{pages[-1]}"

        out = [f"--- SECTION: {sec.title} ({sec.id}) | {page_str} ---", ""]
        out.append(sec.content.strip())

        if sec.children:
            out.append("\nSub-sections available:")
            for child in sec.children:
                child_pages = sorted(child.pages)
                child_page_str = f"p.{child_pages[0]}" if len(child_pages) == 1 else f"p.{child_pages[0]}-{child_pages[-1]}"
                preview = f" | Preview: {child.preview}" if child.preview else ""
                out.append(f"- {child.title} ({child.id}, {child_page_str}){preview}")

        return "\n".join(out)


# =============================================================================
# 5. CHUNKING + RAG
# =============================================================================

def simple_chunk_sections(
    section_index: SectionIndex,
    target_chunk_size: int = 1000,
    chunk_overlap: int = 150,
) -> List[ChunkRecord]:
    chunks: List[ChunkRecord] = []

    for sec in section_index.sections.values():
        if sec.id == "root":
            continue

        text = re.sub(r"\s+", " ", sec.content).strip()
        if not text:
            continue

        start = 0
        chunk_num = 0
        while start < len(text):
            end = min(len(text), start + target_chunk_size)
            chunk_text = text[start:end]

            # try not to cut too abruptly
            if end < len(text):
                last_break = chunk_text.rfind(". ")
                if last_break > int(target_chunk_size * 0.5):
                    end = start + last_break + 1
                    chunk_text = text[start:end]

            pages = sorted(sec.pages)
            page_start = pages[0] if pages else None
            page_end = pages[-1] if pages else None

            chunks.append(
                ChunkRecord(
                    chunk_id=f"{sec.id}_chunk_{chunk_num}",
                    text=chunk_text.strip(),
                    page_start=page_start,
                    page_end=page_end,
                    section_id=sec.id,
                    section_title=sec.title,
                    metadata={},
                )
            )
            chunk_num += 1

            if end >= len(text):
                break
            start = max(end - chunk_overlap, start + 1)

    return chunks


class ChunkRAGIndex:
    """
    Hybrid search:
      - keyword TF-IDF always if sklearn available
      - semantic if embeddings available
      - fallback to keyword grep if embeddings unavailable
    """

    def __init__(
        self,
        chunks: List[ChunkRecord],
        embedding_type: str = "local",
        local_model_name: str = "all-MiniLM-L6-v2",
        api_url: str = "",
        api_key: str = "",
        api_model: str = "",
    ):
        self.chunks = chunks
        self.embedding_type = embedding_type
        self.api_url = api_url
        self.api_key = api_key
        self.api_model = api_model

        self.local_model = None
        if embedding_type == "local" and SentenceTransformer is not None:
            try:
                self.local_model = SentenceTransformer(local_model_name)
            except Exception:
                self.local_model = None

        self.texts = [c.text for c in chunks if c.text.strip()]
        self.valid_chunks = [c for c in chunks if c.text.strip()]

        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        self.chunk_embeddings = None
        self.embedding_available = False

        self._build()

    def _build(self):
        if self.valid_chunks and TfidfVectorizer is not None:
            try:
                self.tfidf_vectorizer = TfidfVectorizer(stop_words="english")
                self.tfidf_matrix = self.tfidf_vectorizer.fit_transform([c.text for c in self.valid_chunks])
            except Exception:
                self.tfidf_vectorizer = None
                self.tfidf_matrix = None

        if self.valid_chunks:
            embs = try_get_embeddings(
                [c.text for c in self.valid_chunks],
                embedding_type=self.embedding_type,
                local_model=self.local_model,
                base_url=self.api_url,
                api_key=self.api_key,
                api_model=self.api_model,
            )
            if embs is not None and len(embs) == len(self.valid_chunks):
                self.chunk_embeddings = embs
                self.embedding_available = True

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        if not self.valid_chunks:
            return []

        # keyword-only fallback
        if self.tfidf_matrix is None or self.tfidf_vectorizer is None or cosine_similarity is None:
            return self._grep_search(query, top_k=top_k)

        query_tfidf = self.tfidf_vectorizer.transform([query])
        keyword_scores = cosine_similarity(query_tfidf, self.tfidf_matrix)[0]

        if self.embedding_available and self.chunk_embeddings is not None:
            query_emb = try_get_embeddings(
                [query],
                embedding_type=self.embedding_type,
                local_model=self.local_model,
                base_url=self.api_url,
                api_key=self.api_key,
                api_model=self.api_model,
            )
            if query_emb is not None:
                semantic_scores = cosine_similarity(query_emb, self.chunk_embeddings)[0]
                hybrid_scores = 0.45 * keyword_scores + 0.55 * semantic_scores
            else:
                hybrid_scores = keyword_scores
        else:
            hybrid_scores = keyword_scores

        top_indices = np.argsort(hybrid_scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            chunk = self.valid_chunks[idx]
            results.append({
                "text": chunk.text,
                "score": float(hybrid_scores[idx]),
                "page_start": chunk.page_start,
                "page_end": chunk.page_end,
                "section_id": chunk.section_id,
                "section_title": chunk.section_title,
            })
        return results

    def _grep_search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        query_terms = [t.lower() for t in re.findall(r"\w+", query) if t.strip()]
        scored = []

        for chunk in self.valid_chunks:
            text_lower = chunk.text.lower()
            score = sum(text_lower.count(term) for term in query_terms)
            if score > 0:
                scored.append((score, chunk))

        scored.sort(key=lambda x: x[0], reverse=True)
        results = []
        for score, chunk in scored[:top_k]:
            results.append({
                "text": chunk.text,
                "score": float(score),
                "page_start": chunk.page_start,
                "page_end": chunk.page_end,
                "section_id": chunk.section_id,
                "section_title": chunk.section_title,
            })
        return results

    def search_formatted(self, query: str, top_k: int = 5) -> str:
        results = self.search(query, top_k=top_k)
        if not results:
            return f"No results for query: '{query}'"

        lines = [f"Search Results for: '{query}'", ""]
        for i, r in enumerate(results, 1):
            page_str = (
                f"Page {r['page_start']}"
                if r["page_start"] == r["page_end"]
                else f"Pages {r['page_start']}-{r['page_end']}"
            )
            snippet = r["text"][:350] + ("..." if len(r["text"]) > 350 else "")
            lines.append(
                f"{i}. Section: {r['section_title']} ({r['section_id']}) | {page_str} | Score: {r['score']:.2f}\n"
                f"   {snippet}"
            )
        return "\n\n".join(lines)


# =============================================================================
# 6. DOCUMENT AGENT
# =============================================================================

class DocumentAgent:
    """
    Public tool-facing interface for LLMs.
    Tools:
      - get_overview()
      - read_details(section_id)
      - search(query, top_k=5)
    """

    def __init__(
        self,
        markdown_text: str,
        embedding_type: str = "local",
        local_model_name: str = "all-MiniLM-L6-v2",
        api_url: str = "",
        api_key: str = "",
        api_model: str = "",
        auto_build_rag: bool = True,
    ):
        self.raw_text = markdown_text
        self.section_index = SectionIndex(markdown_text)
        self.chunk_index: Optional[ChunkRAGIndex] = None

        self.embedding_type = embedding_type
        self.local_model_name = local_model_name
        self.api_url = api_url
        self.api_key = api_key
        self.api_model = api_model

        if auto_build_rag:
            self.ensure_rag()

    def ensure_rag(self):
        if self.chunk_index is not None:
            return

        chunks = simple_chunk_sections(self.section_index)
        self.chunk_index = ChunkRAGIndex(
            chunks=chunks,
            embedding_type=self.embedding_type,
            local_model_name=self.local_model_name,
            api_url=self.api_url,
            api_key=self.api_key,
            api_model=self.api_model,
        )

    # -------------------------------
    # LLM tool interface
    # -------------------------------

    def get_overview(self) -> str:
        return self.section_index.get_overview()

    def read_details(self, section_id: str) -> str:
        return self.section_index.read_details(section_id)

    def search(self, query: str, top_k: int = 5) -> str:
        if self.chunk_index is None:
            self.ensure_rag()

        if self.chunk_index is None:
            return f"Search unavailable for query: '{query}'"

        return self.chunk_index.search_formatted(query, top_k=top_k)


# =============================================================================
# 7. PIPELINE WRAPPERS
# =============================================================================

def create_document_agent_from_pdf(
    pdf_filepath: str,
    ocr_server: Optional[str] = None,
    ocr_model: str = "allenai/olmOCR-2-7B-1025-FP8",
    ocr_workspace: str = "./tmp_ocr",
    use_pdf_page_ocr: bool = False,
    embedding_type: str = "local",
    local_model_name: str = "all-MiniLM-L6-v2",
    api_url: str = "",
    api_key: str = "",
    api_model: str = "",
) -> DocumentAgent:
    if not os.path.exists(pdf_filepath):
        raise FileNotFoundError(f"Could not find PDF at: {pdf_filepath}")

    extractor = MarkdownDocumentExtractor(
        pdf_path=pdf_filepath,
        ocr_server=ocr_server,
        ocr_model=ocr_model,
        ocr_workspace=ocr_workspace,
        use_pdf_page_ocr=use_pdf_page_ocr,
    )
    markdown_text = extractor.extract_to_markdown()

    out_md = Path(pdf_filepath).with_suffix(".extracted.md")
    with open(out_md, "w", encoding="utf-8") as f:
        f.write(markdown_text)
    print(f"Saved markdown to {out_md}")

    agent = DocumentAgent(
        markdown_text=markdown_text,
        embedding_type=embedding_type,
        local_model_name=local_model_name,
        api_url=api_url,
        api_key=api_key,
        api_model=api_model,
        auto_build_rag=True,
    )

    register_pipeline(Path(pdf_filepath).stem, {
        "document_agent": agent,
        "markdown": markdown_text,
    })
    return agent


# =============================================================================
# 8. TOOL HELPERS
# =============================================================================

def tool_get_overview() -> str:
    agent = get_active_document_agent()
    if agent is None:
        return "No active document agent."
    return agent.get_overview()


def tool_read_details(section_id: str) -> str:
    agent = get_active_document_agent()
    if agent is None:
        return "No active document agent."
    return agent.read_details(section_id)


def tool_search(query: str, top_k: int = 5) -> str:
    agent = get_active_document_agent()
    if agent is None:
        return "No active document agent."
    return agent.search(query, top_k=top_k)


# =============================================================================
# 9. TESTS / CLI
# =============================================================================

def test_agent(args):
    sample_markdown = """
# Abstract
This paper introduces a novel framework for analyzing data.

<!-- page 1 mode: simple_text -->
# 1. Introduction
Data analysis is hard. We present a new method.

## 1.1 Background
Historically, people used abacuses.

## 1.2 Motivation
We need faster tools because data is growing.

<!-- page 2 mode: hybrid_paper -->
# 2. Methodology
Our method relies on hybrid extraction.

## 2.1 The OpenDataLoader Approach
We use bounding boxes and layout analysis.

## 2.2 The PaperIndex Structure
We build trees out of documents so LLMs can read them.

# 3. Results
It works very well.
"""

    agent = DocumentAgent(
        markdown_text=sample_markdown,
        embedding_type=args.embedding_type,
        local_model_name=args.local_model,
        api_url=args.api_url,
        api_key=args.api_key,
        api_model=args.api_model,
    )

    print("\n--- GET OVERVIEW ---")
    print(agent.get_overview())

    print("\n--- READ DETAILS sec_2 ---")
    print(agent.read_details("sec_2"))

    print("\n--- SEARCH methodology layout bounding boxes ---")
    print(agent.search("methodology layout bounding boxes"))


def test_pdf(args):
    agent = create_document_agent_from_pdf(
        pdf_filepath=args.pdf_path,
        ocr_server=args.ocr_server,
        ocr_model=args.ocr_model,
        ocr_workspace=args.ocr_workspace,
        use_pdf_page_ocr=args.use_pdf_page_ocr,
        embedding_type=args.embedding_type,
        local_model_name=args.local_model,
        api_url=args.api_url,
        api_key=args.api_key,
        api_model=args.api_model,
    )

    print("\n--- GET OVERVIEW ---")
    print(agent.get_overview())

    print("\n--- SEARCH quantization bias stochastic rounding ---")
    print(agent.search("quantization bias stochastic rounding", top_k=5))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Improved Document Agent v2")

    parser.add_argument("--test", choices=["agent", "pdf", "both"], default="both")
    parser.add_argument("--pdf_path", type=str, default="data/pdftest/NVIDIA-Nemotron-3-Super-Technical-Report.pdf")

    parser.add_argument("--ocr_server", type=str, default=None)
    parser.add_argument("--ocr_model", type=str, default="allenai/olmOCR-2-7B-1025-FP8")
    parser.add_argument("--ocr_workspace", type=str, default="./tmp_ocr")
    parser.add_argument("--use_pdf_page_ocr", action="store_true")

    parser.add_argument("--embedding_type", choices=["local", "api"], default="api")
    parser.add_argument("--local_model", type=str, default="all-MiniLM-L6-v2")
    parser.add_argument("--api_url", type=str, default="http://localhost:8081/v1")
    parser.add_argument("--api_key", type=str, default="EMPTY")
    parser.add_argument("--api_model", type=str, default="text-embeddings-inference")

    args = parser.parse_args()

    if args.test in ["agent", "both"]:
        test_agent(args)

    if args.test in ["pdf", "both"]:
        test_pdf(args)

"""
python -m BatchAgent.ocr_agent.document_agent --test agent

python -m BatchAgent.ocr_agent.document_agent \
  --test agent \
  --embedding_type api \
  --api_url http://localhost:8081/v1 \
  --api_key EMPTY \
  --api_model text-embeddings-inference


python -m BatchAgent.ocr_agent.document_agent \
  --test pdf \
  --pdf_path data/pdftest/NVIDIA-Nemotron-3-Super-Technical-Report.pdf

python -m BatchAgent.ocr_agent.document_agent \
  --test pdf \
  --pdf_path data/sote/SP24_CMPE131_01.pdf \
  --ocr_server http://localhost:8002/v1 \
  --ocr_model allenai/olmOCR-2-7B-1025-FP8 \
  --ocr_workspace ./output_old/tmp_ocr

Test PDF with API embedding:
python -m BatchAgent.ocr_agent.document_agent \
  --test pdf \
  --pdf_path data/pdftest/NVIDIA-Nemotron-3-Super-Technical-Report.pdf \
  --ocr_server http://localhost:8002/v1 \
  --embedding_type api \
  --api_url http://100.81.148.35:8003/v1 \
  --api_key EMPTY \
  --api_model BAAI/bge-large-zh-v1.5
"""
