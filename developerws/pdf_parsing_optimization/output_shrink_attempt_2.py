#!/usr/bin/env python3
import os
import re
import json
import base64
import logging
import time
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol
import datetime as _dt
from dateutil import parser as _dtparse 
from itertools import zip_longest
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv
load_dotenv()

import fitz  # PyMuPDF

# ───────── logging ─────────
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

# ───────── config ─────────
LLM_PROVIDER = (os.getenv("PDF_LLM_PROVIDER") or "").lower().strip()
OVERWRITE_OUTPUT = (os.getenv("OVERWRITE_OUTPUT", "0").lower() in ("1","true","yes"))

# Gemini
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

# Bedrock Claude
AWS_REGION = os.getenv("AWS_REGION", "ap-south-1")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID", "")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY", "")
BEDROCK_INFERENCE_PROFILE_ARN = os.getenv("BEDROCK_INFERENCE_PROFILE_ARN", "")
CLAUDE_MODEL_ID = os.getenv("CLAUDE_MODEL_ID", "anthropic.claude-3-7-sonnet-20250219-v1:0")

# Generation knobs
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "12000"))

# I/O
PDF_INPUT_DIR = os.getenv("PDF_INPUT_DIR", "test")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "output_reduce_2")
RAW_JSON_DIR = os.getenv("RAW_JSON_DIR", "output_reduce_rawjson_2")
SENT_IMG_DIR = Path("output_reduce_2/images")
SENT_PDF_DIR = Path("output_reduce_2/pdfs")

SAVE_SENT_IMAGES = (os.getenv("SAVE_SENT_IMAGES", "1").strip() not in ("0", "false", "False", "no"))
SAVE_SENT_PDF    = (os.getenv("SAVE_SENT_PDF", "1").strip() not in ("0", "false", "False", "no"))

# ---- Optional pricing (USD per 1K tokens), override via env ----
LLM_PRICE_IN_PER_1K  = float(os.getenv("LLM_PRICE_IN_PER_1K",  "0"))
LLM_PRICE_OUT_PER_1K = float(os.getenv("LLM_PRICE_OUT_PER_1K", "0"))

# Claude prompt caching toggle + targets
CLAUDE_ENABLE_CACHE = (os.getenv("CLAUDE_ENABLE_CACHE", "1").lower() in ("1","true","yes"))
TARGET_CACHED_PROMPT_TOKENS = int(os.getenv("TARGET_CACHED_PROMPT_TOKENS", "1200"))

from pathlib import Path  # already imported above, ok to leave as-is
APPENDIX_PATH = Path("prompts/prompt_appendix.md")

# Per-page chunk cap (0 or empty = no cap)
PAGE_CHUNK_CAP = int(os.getenv("PAGE_CHUNK_CAP", "0"))  # ← default 0 (no cap)

# Request/output safety
IMAGE_MAX_BYTES = 4_500_000
MAX_MODEL_RETRIES = 3
RETRY_BASE_SLEEP = 2.0

# Chunking / merging
MAX_WORDS = 400
MIN_WORDS = 200
TITLE_MERGE_MAX_WORDS = 12
CHUNK_CHAR_LIMIT = 6000
OVERFLOW_MIN_WORDS = 120
MIN_CHUNK_TOKENS = 10

# --- Quotas (Gemini 2.5 Flash) ---
GEM_RPM = int(os.getenv("GEMINI_RPM", "1000"))
GEM_TPM = int(os.getenv("GEMINI_TPM", "1000000"))
GEM_RPD = int(os.getenv("GEMINI_RPD", "10000"))

# 70% safety margin
SAFE_FRACTION = float(os.getenv("TOKEN_SAFETY_FRACTION", "0.70"))
SAFE_RPM = int(GEM_RPM * SAFE_FRACTION)           # 700
SAFE_TPM = int(GEM_TPM * SAFE_FRACTION)           # 700_000
SAFE_RPD = int(GEM_RPD * SAFE_FRACTION)           # 7_000

# If you ever use batch endpoints
GEM_BATCH_TOKENS = int(os.getenv("GEMINI_BATCH_TOKENS", "3000000"))
SAFE_BATCH_TOKENS = int(GEM_BATCH_TOKENS * SAFE_FRACTION)  # 2_100_000

# -----------------------
# IMAGE SIZE / TOKEN CONFIG (updated)
IMAGE_TARGET_WIDTH_PX  = int(os.getenv("IMAGE_TARGET_WIDTH_PX", "1600"))
IMAGE_TARGET_HEIGHT_PX = int(os.getenv("IMAGE_TARGET_HEIGHT_PX", "2000"))
IMAGE_FORCE_ASPECT     = (os.getenv("IMAGE_FORCE_ASPECT", "1").lower() in ("1","true","yes"))

# Token divisor for your formula: tokens = (width_px * height_px) / IMAGE_TOKEN_DIVISOR
IMAGE_TOKEN_DIVISOR = int(os.getenv("IMAGE_TOKEN_DIVISOR", "750"))

# If true, skip cached/padded prompt creation when sending TABLES_CHARTS_PROMPT
SKIP_PROMPT_CACHING_FOR_TABLES = (os.getenv("SKIP_PROMPT_CACHING_FOR_TABLES", "1").lower() in ("1", "true", "yes"))


# --- New: toggle to only send tables/charts to the LLM (default ON)
SEND_TABLES_ONLY = (os.getenv("SEND_TABLES_ONLY", "1").lower() in ("1", "true", "yes"))

# --- concise prompt that asks model to ONLY output tables & charts as JSON
TABLES_CHARTS_PROMPT = r"""
You will be given an image of a single PDF page. Produce **only** valid JSON (no surrounding text, no explanation, no markdown fences).
The JSON must be an object with exactly two top-level keys: "tables" (list) and "charts" (list).
Each table must use this structure:
  {"type":"table", "title": <string or null>, "headers": [<str>, ...], "rows": [[cell,...], ...], "page_number": <int>}
Each chart must use this structure:
  {"type":"chart", "chart_type": <str or null>, "title": <string or null>, "x_axis": {"label": <str or null>, "values": [..]}, "y_axis": {"label": <str or null>, "values": [..]}, "data_series": [{"name": <str or null>, "values":[..]}, ...], "page_number": <int>}
If no tables or charts are found, return {"tables": [], "charts": []}.
Do NOT include narrative text, headings, footers, or any other content. If you are unsure whether a visual element is a table or a chart, prefer to omit it (return nothing for it).
Return only the JSON object.
"""


# Whether to dump images to disk for debugging
PDF_DUMP_IMAGES = (os.getenv("PDF_DUMP_IMAGES", "1").lower() in ("1","true","yes"))

CONCURRENCY_PAGES = int(os.getenv("CONCURRENCY_PAGES", "8"))  # LLM workers per PDF

# ---- token accounting (fallback) ----
class TokenAccumulator:
    def __init__(self):
        self.input: List[int] = []
        self.output: List[int] = []

    def add(self, inp: Optional[int], out: Optional[int]):
        if inp is not None: self.input.append(int(inp))
        if out is not None: self.output.append(int(out))

    def summary(self) -> Dict[str, Any]:
        def stats(arr: List[int]):
            if not arr:
                return {"total": 0, "min": 0, "max": 0, "avg": 0.0}
            return {"total": sum(arr), "min": min(arr), "max": max(arr), "avg": round(sum(arr)/len(arr),2)}
        return {"input": stats(self.input), "output": stats(self.output)}

TOK = TokenAccumulator()

def _approx_tokens_from_text(s: str) -> int:
    return max(1, len(s)//4) if s else 0

def _bedrock_raw_image_budget() -> int:
    BASE64_CAP = 5 * 1024 * 1024
    RAW_CAP = (BASE64_CAP * 3) // 4
    SAFETY = 12 * 1024
    return max(512 * 1024, RAW_CAP - SAFETY)

# =========================================================
#                          PROMPT
# =========================================================
BASE_PROMPT = r"""
**STRICT CAPTURE — DO NOT SUMMARIZE OR OMIT ANY TOKEN.**
Capture every visible text token, including small footnotes, table notes, and figure captions.

**PDF to JSON Conversion Task**
- Objective: Extract and convert the entire content of the provided PDF document into a structured JSON format, preserving the hierarchical organization of information.
- Bullet lists MUST be captured verbatim as separate bullet lines. Do not summarize bullets. Include box headings as text sections.
- Important: Use only the provided page image(s) and this instruction to build the JSON. Do not rely on any separate OCR or extracted text from the PDF body; none will be sent.

- Instructions
Analyze the PDF document thoroughly to identify all content elements including:
- In first page the if date is present then that is the subtitle unless a proper subtitle is present.
- You do not have to capture 'image' field, only text (if present) in the image should only be captured.

1. Textual content (headings, paragraphs, lists)
    Tables with all rows, columns, cell values, merged cells and any text related to the tables should be captured as description
    Charts and graphs with their titles, labels, data points, and descriptions.
    For any requested field or key-value pair, if no corresponding value is found in the source document, the value in the JSON output must be the literal `null`.
    Do not use an empty string (`""`) or any other placeholder for a field that has no value in the source text.

2. Create a hierarchical JSON structure that:

    Maintains the document's original organization (sections, subsections, title , subtitle, etc. )
    Preserves the relationship between different content elements.
    Captures the complete data from all tables (including headers and all cells) and descriptions if any.
    Represents charts with their visual data and accompanying text as description.
    Follow a consistent nomenclature throughout the json and do not deviate from it.

3. For tables, use the following structure:
    {"type": "table",
    "title": "Table title if available",
    "headers": ["Column1", "Column2", "..."],
    "rows": [
    ["Row1Cell1", "Row1Cell2", "..."],
    ["Row2Cell1", "Row2Cell2", "..."]
    ]}

- If a column header's name is repeated in different parts of the table capture it as a single entry in the headers array, but do not skip or omit it if it appears again. 
- Your goal is to represent the complete structure of the source table, including any repeated column names.

4. For charts and graphs, use the following structure:

    {"type": "chart",
    "chart_type": "bar/line/pie/etc.",
    "title": "Chart title if available",
    "x_axis": {"label": "X-axis label","values": ["Value1", "Value2", "..."]},
    "y_axis": {"label": "Y-axis label","values": [Value1, Value2, ...]},
    "data_series": [{"name": "Series name if available","values": [Value1, Value2, ...]}]
    }

5. For double line charts, capture the data for each line as series with (date,value) pairs if possible.

- Ensure all numerical data in tables and charts is preserved accurately.
- Maintain the reading order of the document in your JSON structure.
- Do not capture 'Page information' and 'date' separately , keep them as a part of footer.

- Output Format: 
Provide the complete JSON representation of the PDF content without any additional explanations. The output should be valid, parsable JSON that accurately represents all information from the original document.
Return only the JSON output without any preamble or additional explanations.
"""

def _normalize_unix_newlines(s: str) -> str:
    return (s or "").replace("\r\n", "\n").replace("\r", "\n")

def build_cached_prompt(base_prompt: str, target_tokens: int = TARGET_CACHED_PROMPT_TOKENS) -> str:
    """Make the cacheable prompt ≥ target_tokens by appending stable text."""
    try:
        appendix = APPENDIX_PATH.read_text(encoding="utf-8")
    except Exception:
        appendix = ""

    base = _normalize_unix_newlines(base_prompt.strip())
    appx = _normalize_unix_newlines(appendix.strip())

    combo = base + "\n\n" + appx + "\n"
    tok = _approx_tokens_from_text(combo)

    if tok < target_tokens:
        filler = "\n# FILLER BLOCK (no-op, duplicate, stable)\n" + \
                 "\n".join(["- filler line"] * 50) + "\n"
        per = _approx_tokens_from_text(filler)
        need = max(0, (target_tokens - tok + per - 1) // per)
        need = min(need, 20)
        combo = combo + (filler * need)

    logger.info("Cached prompt approx tokens=%d (target=%d)", _approx_tokens_from_text(combo), target_tokens)
    return combo

# =========================================================
#        Provider interface & adapters
# =========================================================
class LLMProvider(Protocol):
    def generate_json(
        self, image_bytes: bytes, mime_type: str, base_prompt: str
    ) -> tuple[str, Dict[str, Optional[int]]]: ...
    def label(self) -> str: ...

class GeminiProvider:
    def __init__(self, api_key: str, model: str, temperature: float = 0.0):
        try:
            import google.generativeai as genai
        except Exception as e:
            raise RuntimeError("google-generativeai is required. `pip install google-generativeai`.") from e
        if not api_key:
            raise RuntimeError("Set GEMINI_API_KEY in environment for Gemini runs.")
        genai.configure(api_key=api_key)
        self._genai = genai
        self._model_name = model
        self._model = genai.GenerativeModel(
            model_name=model,
            generation_config={"temperature": temperature, "response_mime_type": "application/json"},
        )

    def generate_json(self, image_bytes: bytes, mime_type: str, base_prompt: str):
        parts = [base_prompt]
        if image_bytes:
            parts.append({"mime_type": mime_type, "data": image_bytes})
        resp = self._model.generate_content(parts)
        usage = getattr(resp, "usage_metadata", None) or {}
        in_tok  = usage.get("prompt_token_count")
        out_tok = usage.get("candidates_token_count")
        text = (getattr(resp, "text", "") or "").strip()
        return text, {"input_tokens": in_tok, "output_tokens": out_tok}

    def label(self) -> str:
        return f"Gemini ({self._model_name})"

class BedrockClaudeProvider:
    def __init__(
        self,
        region: str,
        access_key: str,
        secret_key: str,
        model_id: str,
        inference_profile: str = "",
        temperature: float = 0.0,
        max_tokens: int = 12000,
    ):
        try:
            import boto3
            from botocore.config import Config as BotoConfig
        except Exception as e:
            raise RuntimeError("boto3 is required for Bedrock. `pip install boto3 botocore`.") from e
        self._model_id = inference_profile or model_id
        self._client = boto3.client(
            "bedrock-runtime",
            region_name=region,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            config=BotoConfig(read_timeout=300, connect_timeout=30, retries={"max_attempts": 3}),
        )
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._via = "inference-profile" if inference_profile else "modelId"

    def generate_json(
        self, image_bytes: bytes, mime_type: str, base_prompt: str
    ) -> tuple[str, Dict[str, Optional[int]]]:
        import base64, json
        full_prompt = build_cached_prompt(base_prompt)
        prompt_block: Dict[str, Any] = {"type": "text", "text": full_prompt}
        if CLAUDE_ENABLE_CACHE:
            prompt_block["cache_control"] = {"type": "ephemeral"}
        image_block = {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": mime_type,
                "data": base64.b64encode(image_bytes).decode("utf-8"),
            },
        }
        req = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": self._max_tokens,
            "temperature": self._temperature,
            "messages": [{"role": "user", "content": [prompt_block, image_block]}],
        }
        logger.info("Cache flag on prompt: %s", bool(prompt_block.get("cache_control")))
        resp = self._client.invoke_model(
            modelId=self._model_id,
            body=json.dumps(req).encode("utf-8"),
            contentType="application/json",
            accept="application/json",
        )
        raw = resp["body"].read()
        raw = raw.decode("utf-8") if isinstance(raw, (bytes, bytearray)) else raw
        try:
            data = json.loads(raw)
            usage_dict = data.get("usage") or {}
            logger.info("Bedrock usage raw: %s", json.dumps(usage_dict))
        except Exception:
            return (raw or "").strip(), {"input_tokens": None, "output_tokens": None}

        parts = []
        for c in data.get("content", []):
            if isinstance(c, dict) and c.get("text"):
                parts.append(c["text"])
        usage_dict = data.get("usage") or {}
        usage = {
            "input_tokens": usage_dict.get("input_tokens"),
            "output_tokens": usage_dict.get("output_tokens"),
            "cache_read_input_tokens": usage_dict.get("cache_read_input_tokens"),
            "cache_creation_input_tokens": usage_dict.get("cache_creation_input_tokens"),
        }
        cr = usage.get("cache_read_input_tokens") or 0
        cw = usage.get("cache_creation_input_tokens") or 0
        if cr or cw:
            logger.info("Prompt cache → read=%s, write=%s (tokens)", cr, cw)
        return "\n\n".join(parts).strip(), usage

    def label(self) -> str:
        return f"Claude via Bedrock ({self._via}={self._model_id})"

def get_provider_from_env() -> LLMProvider:
    prov = (os.getenv("PDF_LLM_PROVIDER") or "").lower().strip()
    if prov.startswith("gemini"):
        return GeminiProvider(api_key=GEMINI_API_KEY, model=GEMINI_MODEL_NAME, temperature=LLM_TEMPERATURE)
    if prov.startswith("bedrock"):
        return BedrockClaudeProvider(
            region=AWS_REGION,
            access_key=AWS_ACCESS_KEY_ID,
            secret_key=AWS_SECRET_ACCESS_KEY,
            model_id=CLAUDE_MODEL_ID,
            inference_profile=BEDROCK_INFERENCE_PROFILE_ARN,
            temperature=LLM_TEMPERATURE,
            max_tokens=LLM_MAX_TOKENS,
        )
    raise RuntimeError("Unsupported or missing PDF_LLM_PROVIDER. Use 'gemini' or 'bedrock'.")

PROVIDER: LLMProvider = get_provider_from_env()

def _llm_provider_label() -> str:
    return PROVIDER.label()

def _provider_key() -> str:
    env = (os.getenv("PDF_LLM_PROVIDER") or "").lower()
    try:
        lab = (PROVIDER.label() or "").lower()
    except Exception:
        lab = ""
    if "gemini" in env or "gemini" in lab: return "Gemini"
    if "openai" in env or "openai" in lab: return "OpenAI"
    if "bedrock" in env or "claude" in lab: return "Claude"
    tok = re.sub(r"[^a-zA-Z0-9]+", "", (lab.split()[0] if lab else (env or "LLM")))
    return tok or "LLM"

PROVIDER_KEY = _provider_key()

def select_provider(name: str):
    """Switch global provider at runtime."""
    name = (name or "").lower().strip()
    os.environ["PDF_LLM_PROVIDER"] = name
    global PROVIDER, PROVIDER_KEY
    PROVIDER = get_provider_from_env()
    PROVIDER_KEY = _provider_key()

def _price_for_current_provider():
    key = (PROVIDER_KEY or "").upper()
    in_p  = float(os.getenv(f"{key}_PRICE_IN_PER_1K",  os.getenv("LLM_PRICE_IN_PER_1K",  "0") or "0"))
    out_p = float(os.getenv(f"{key}_PRICE_OUT_PER_1K", os.getenv("LLM_PRICE_OUT_PER_1K", "0") or "0"))
    return in_p, out_p

# =========================================================
#            Page rendering helpers (byte-safe)
# =========================================================
DATE_PAT = re.compile(
    r"\b(?:\d{1,2}\s*(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|"
    r"May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|"
    r"Nov(?:ember)?|Dec(?:ember)?)\s*,?\s*\d{4}"
    r"|(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|"
    r"Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{1,2},?\s+\d{4}"
    r"|\d{4}-\d{2}-\d{2}"
    r"|\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b",
    re.IGNORECASE
)

FOOTER_MONTH_YEAR_PAT = re.compile(
    r"\b(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
    r"Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+(20\d{2})\b",
    re.IGNORECASE
)
DAY_MONTH_NO_YEAR_PAT = re.compile(
    r"\b(\d{1,2})(?:st|nd|rd|th)?\s*[-_/., ]*\s*"
    r"(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
    r"Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\b",
    re.IGNORECASE
)
YEAR_PAT = re.compile(r"(20\d{2})")
# --- add near other helpers ---
_BUL_START = re.compile(r"^\s*(?:[\u2022•▪‣\-–*]+|\d{1,2}[.)])\s+")

def _fallback_scrape_page(pdf_path: Path, page_no: int) -> dict:
    # If we are in tables-only mode, never extract or return narrative text.
    # This prevents any accidental injection of narrative when SEND_TABLES_ONLY is enabled.
    try:
        if SEND_TABLES_ONLY:
            return {}
    except NameError:
        # If SEND_TABLES_ONLY is not defined for some reason, continue normally.
        pass

    try:
        import fitz
        with fitz.open(str(pdf_path)) as d:
            p = d.load_page(page_no - 1)
            dd = p.get_text("dict")
            raw_text = p.get_text("text") or ""
    except Exception:
        return {}

    # If literally no text layer, bail (you need OCR or rely on the LLM image)
    if (not dd or not dd.get("blocks")) and not raw_text.strip():
        return {}

    lines = []
    for b in dd.get("blocks", []):
        for ln in b.get("lines", []):
            spans = ln.get("spans", []) or []
            txt = "".join(s.get("text", "") for s in spans).strip()
            if not txt:
                continue
            size = max((s.get("size", 0) for s in spans), default=0.0)
            bold = any(("Bold" in (s.get("font","") or "")) or (s.get("flags",0) & 2) for s in spans)
            lines.append({"text": txt, "size": float(size), "bold": bool(bold)})

    # Use median+1pt as heading threshold
    sizes = sorted(x["size"] for x in lines) or [0.0]
    head_thresh = sizes[len(sizes)//2] + 1.0

    sections = []
    cur_title, cur_buf = None, []
    def _flush():
        nonlocal cur_title, cur_buf, sections
        body = "\n".join(cur_buf).strip()
        if cur_title or body:
            sections.append((cur_title, body))
        cur_title, cur_buf = None, []

    for x in lines:
        t = x["text"]
        if x["bold"] or x["size"] >= head_thresh:
            _flush()
            cur_title = t
        else:
            cur_buf.append(t)

    _flush()

    # pack into narrative_text
    parts = []
    for title, body in sections:
        if title and body:
            parts.append(f"{title}\n{body}")
        elif title:
            parts.append(title)
        elif body:
            parts.append(body)

    narrative_text = "\n\n".join(parts).strip()
    if not narrative_text:
        return {}
    return {"title": None, "subtitle": None, "narrative_text": narrative_text, "tables": [], "charts": []}

import re
from typing import Set

# small helpers used by the filter
_RE_NON_ALPHANUM = re.compile(r"[^0-9A-Za-z\-\.%]+")
_RE_NUMERIC = re.compile(r"^[\-\d\.\,]+$")

def _norm_token(s: str) -> str:
    if not s:
        return ""
    s2 = s.strip().lower()
    s2 = _RE_NON_ALPHANUM.sub(" ", s2)
    s2 = " ".join(s2.split())
    return s2

def _tokens_from_text(s: str) -> Set[str]:
    return {t for t in (_norm_token(x) for x in re.split(r"\s+", s or "")) if t}

def _is_mostly_numeric(s: str, numeric_set: Set[str]) -> bool:
    # Return True if majority of tokens are numeric and those tokens appear in numeric_set
    toks = [t for t in (x.strip() for x in re.split(r"\s+|\n", s or "")) if t]
    if not toks:
        return False
    num_count = 0
    matched_num = 0
    for t in toks:
        tn = t.replace(",", "")
        if _RE_NUMERIC.match(tn):
            num_count += 1
            if tn in numeric_set or _norm_token(tn) in numeric_set:
                matched_num += 1
    # at least half numeric and most of those appear in numeric_set
    return (num_count / len(toks)) >= 0.5 and (matched_num >= max(1, int(0.6 * num_count)))

def _filter_local_struct_by_llm(page_obj: dict, local_struct: dict) -> dict:
    """
    Remove local sections/lines that overlap strongly with LLM's tables/charts.
    Returns a cleaned local_struct (copy of input) where duplicate table/chart text is removed.
    """
    if not local_struct:
        return local_struct or {}

    # Build token sets from LLM tables and charts
    table_tokens: Set[str] = set()
    numeric_values: Set[str] = set()
    short_tokens: Set[str] = set()

    # Tables: headers + every cell
    for tbl in (page_obj.get("tables") or []):
        for h in (tbl.get("headers") or []):
            nh = _norm_token(str(h))
            if nh:
                table_tokens.add(nh)
                short_tokens.update(nh.split())
        for row in (tbl.get("rows") or []):
            for cell in row:
                cs = str(cell or "")
                nc = _norm_token(cs)
                if not nc:
                    continue
                table_tokens.add(nc)
                # split cell into tokens, identify numeric tokens
                for tok in nc.split():
                    short_tokens.add(tok)
                    if _RE_NUMERIC.match(tok.replace(",", "")):
                        numeric_values.add(tok.replace(",", ""))

    # Charts: x_axis values, y_axis values, labels, series
    for ch in (page_obj.get("charts") or []):
        # x axis
        xvals = (ch.get("x_axis") or {}).get("values") or []
        for v in xvals:
            nv = _norm_token(str(v))
            if nv:
                table_tokens.add(nv)
                short_tokens.update(nv.split())
        # y axis numeric values (may be numbers)
        yvals = (ch.get("y_axis") or {}).get("values") or []
        for v in yvals:
            s = str(v)
            nv = _norm_token(s)
            if nv:
                table_tokens.add(nv)
            if _RE_NUMERIC.match(s.replace(",", "")):
                numeric_values.add(s.replace(",", ""))
        # series values
        for ds in (ch.get("data_series") or []):
            for v in (ds.get("values") or []):
                s = str(v)
                nv = _norm_token(s)
                if nv:
                    table_tokens.add(nv)
                if _RE_NUMERIC.match(s.replace(",", "")):
                    numeric_values.add(s.replace(",", ""))

    # additional tokens from table titles / chart titles
    for tbl in (page_obj.get("tables") or []):
        t = _norm_token(tbl.get("title") or "")
        if t: table_tokens.add(t)
    for ch in (page_obj.get("charts") or []):
        t = _norm_token(ch.get("title") or "")
        if t: table_tokens.add(t)

    # now filter local_struct sections
    cleaned_sections = []
    sections = local_struct.get("sections") or []
    for sec in sections:
        heading = sec.get("heading") or ""
        content = sec.get("content") or ""

        nh = _norm_token(heading)
        # drop heading if it's exactly a table header / axis label
        if nh and (nh in table_tokens or any(tok in table_tokens for tok in nh.split())):
            # skip this section entirely (we assume it is table text)
            continue

        # clean content line-by-line
        kept_lines = []
        for line in (content.split("\n") if content else []):
            line = line.strip()
            if not line:
                continue
            # normalize tokens in the line
            line_tokens = _tokens_from_text(line)
            if not line_tokens:
                continue

            # 1) if whole line normalized is present in table tokens -> drop
            if _norm_token(line) in table_tokens:
                continue

            # 2) if line is mostly numeric and numbers present in numeric_values -> drop
            if _is_mostly_numeric(line, numeric_values):
                continue

            # 3) if high overlap with table_tokens (>=0.6) -> drop
            overlap = sum(1 for t in line_tokens if t in table_tokens)
            if overlap / max(1, len(line_tokens)) >= 0.6:
                continue

            # else keep
            kept_lines.append(line)

        # if heading survived but no content remains, still keep heading only if it's not in table tokens
        if not kept_lines and nh:
            # if heading is short and equals a numeric or a table header-like token, skip.
            if nh in table_tokens or all(tok in short_tokens or _RE_NUMERIC.match(tok) for tok in nh.split()):
                continue
            cleaned_sections.append({"heading": heading, "content": ""})
        elif kept_lines:
            cleaned_sections.append({"heading": heading, "content": "\n".join(kept_lines)})

    # create new local_struct copy
    filtered = dict(local_struct)
    filtered["sections"] = cleaned_sections
    # add debug counts
    filtered.setdefault("debug", {})["orig_sections"] = len(sections)
    filtered["debug"]["filtered_sections"] = len(cleaned_sections)
    return filtered


def _local_extract_text_structured(pdf_path: Path, page_no: int) -> Dict[str, Any]:
    """
    Extract text from the PDF page using the text layer and return structured JSON:
    {
      "title": <str|null>,
      "subtitle": <str|null>,
      "sections": [ {"heading": <str|null>, "content": <str>}, ... ],
      "raw_lines": [ {"text":..., "size":..., "bold":..., "y":..., "x":...}, ... ]
    }
    This intentionally uses only local OCR/text layer (PyMuPDF) and never calls the LLM.
    """
    out: Dict[str, Any] = {"title": None, "subtitle": None, "sections": [], "raw_lines": []}
    try:
        import fitz
        with fitz.open(str(pdf_path)) as doc:
            # fitz pages are 0-indexed; caller uses 1-indexed pages
            p = doc.load_page(page_no - 1)
            dd = p.get_text("dict") or {}
    except Exception as e:
        logger.debug("Local structured text extractor failed to open page %s:%s -> %s", pdf_path, page_no, e)
        return out

    # If no text layer, return empty structure
    blocks = dd.get("blocks") or []
    if not blocks:
        return out

    # Collect lines with approximate y coordinate, size and boldness
    lines = []
    for b in blocks:
        bbox = b.get("bbox", [0, 0, 0, 0])
        for ln in b.get("lines", []):
            spans = ln.get("spans", []) or []
            txt = "".join(s.get("text", "") for s in spans).strip()
            if not txt:
                continue
            # compute approximate font size: max of spans
            size = max((s.get("size", 0) for s in spans), default=0.0)
            bold = any(("Bold" in (s.get("font","") or "")) or (s.get("flags",0) & 2) for s in spans)
            # take x,y from first span bbox if present
            span_bbox = spans[0].get("bbox") if spans and spans[0].get("bbox") else bbox
            x = span_bbox[0] if span_bbox else 0.0
            y = span_bbox[1] if span_bbox else 0.0
            # discard trivial artifact lines
            if _RE_PAGE.match(txt) or _RE_ONLY_BULLETLITE.match(txt):
                continue
            lines.append({"text": txt, "size": float(size), "bold": bool(bold), "x": float(x), "y": float(y)})

    if not lines:
        return out

    # sort by vertical position then x to keep reading order
    lines.sort(key=lambda r: (round(r["y"], 1), round(r["x"], 1)))

    # expose raw lines to debug if needed
    out["raw_lines"] = lines

    # Find unique sizes and determine thresholds for title / heading
    sizes = sorted({round(l["size"], 2) for l in lines if l.get("size") and l.get("size") > 0}, reverse=True)
    # choose title_size and heading_size heuristically
    title_size = sizes[0] if sizes else None
    heading_size = sizes[1] if len(sizes) > 1 else (sizes[0] if sizes else None)

    # Identify a probable title/subtitle near top: look at the first 3 lines
    first_lines = lines[: min(4, len(lines))]
    if first_lines:
        if title_size and first_lines[0]["size"] >= (title_size - 0.01):
            out["title"] = first_lines[0]["text"]
            # subtitle candidate: second line slightly smaller or next bold line
            if len(first_lines) > 1 and first_lines[1]["size"] < first_lines[0]["size"]:
                out["subtitle"] = first_lines[1]["text"]

    # Build sections using heading_size and boldness
    sections: List[Dict[str, Any]] = []
    cur = {"heading": None, "content_lines": []}

    # Threshold: treat as heading if size >= heading_size (if heading_size exists) or bold True
    for ln in lines:
        is_heading = False
        if heading_size and ln["size"] >= (heading_size - 0.01) and ln["text"] != out.get("title"):
            is_heading = True
        if ln["bold"]:
            # if bold but very short, likely heading
            if is_trivial_heading(ln["text"]) or len(ln["text"].split()) <= 6:
                is_heading = True

        if is_heading:
            # flush previous
            if cur and (cur["heading"] or cur["content_lines"]):
                sections.append({"heading": cur["heading"], "content": "\n".join(cur["content_lines"]).strip()})
            cur = {"heading": ln["text"], "content_lines": []}
        else:
            cur["content_lines"].append(ln["text"])

    # flush last
    if cur and (cur["heading"] or cur["content_lines"]):
        sections.append({"heading": cur["heading"], "content": "\n".join(cur["content_lines"]).strip()})

    # Post-process: drop empty sections
    cleaned = []
    for s in sections:
        content = s.get("content") or ""
        heading = s.get("heading")
        # skip tiny artifact-only sections
        if not content and not heading:
            continue
        # compress spaces
        content = _RE_MULTISPACE.sub(" ", content).strip()
        cleaned.append({"heading": heading, "content": content})

    out["sections"] = cleaned
    return out



def _date_from_parent_folders_relaxed(path: Path) -> Optional[_dt.date]:
    for p in path.parents:
        m = re.search(r"\b(20\d{2})(\d{2})(\d{2})\b", p.name)
        if m:
            try: return _dt.date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
            except: pass
        m = re.search(r"\b(20\d{2})[-_](\d{1,2})[-_](\d{1,2})\b", p.name)
        if m:
            try: return _dt.date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
            except: pass
    return None

def _date_from_filename_relaxed(stem: str, context_year: int | None = None) -> _dt.date | None:
    if not stem:
        return None
    s = stem.strip()
    def _parse_with_context(txt: str) -> _dt.date | None:
        return _try_parse_ambiguous_date(txt, context_year=context_year)
    tail24 = s[-24:]
    m_tail = re.fullmatch(r"(\d{4})_(\d{2})_(\d{2})T\d{2}_\d{2}_\d{2}\.\d{3}Z", tail24)
    if m_tail:
        try:
            y, m, d = int(m_tail.group(1)), int(m_tail.group(2)), int(m_tail.group(3))
            return _dt.date(y, m, d)
        except Exception:
            pass
    candidates: list[tuple[int, _dt.date, bool]] = []
    iso_patterns = [
        r"\b(20\d{2})[_.-](\d{1,2})[_.-](\d{1,2})(?:[T _-]\d{2}[_:]\d{2}(?:[_:]\d{2})?(?:\.\d+)?Z?)?\b",
        r"\b(\d{1,2})[_.-](\d{1,2})[_.-](20\d{2})\b",
        r"\b(20\d{2})(\d{2})(\d{2})\b",
        r"\b(\d{2})(\d{2})(20\d{2})\b",
    ]
    for pat in iso_patterns:
        for m in re.finditer(pat, s):
            try:
                g = m.groups()
                txt = " ".join(str(x) for x in g if x is not None)
                dt = _parse_with_context(txt)
                if dt:
                    candidates.append((m.start(), dt, True))
            except Exception:
                continue
    pat_mdY = re.compile(
        r"\b(\d{1,2})(?:st|nd|rd|th)?\s*[-_/., ]*\s*"
        r"(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
        r"Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
        r"(?:\s*[-_/., ]*\s*(\d{4}))?\b",
        re.IGNORECASE,
    )
    pat_MdY = re.compile(
        r"\b(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
        r"Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
        r"\s*[-_/., ]*\s*(\d{1,2})(?:st|nd|rd|th)?"
        r"(?:\s*[-_/., ]*\s*(\d{4}))?\b",
        re.IGNORECASE,
    )
    def _cand_from_match(m: re.Match, txt_builder: callable, has_year_idx: int | None) -> None:
        txt = txt_builder(m)
        dt = _parse_with_context(txt)
        if dt:
            has_year = False
            if has_year_idx is not None:
                y = m.group(has_year_idx)
                has_year = bool(y and y.isdigit() and len(y) == 4)
            candidates.append((m.start(), dt, has_year))
    for m in pat_mdY.finditer(s):
        def _txt(mm):
            d = mm.group(1); mon = mm.group(2); y = mm.group(3) or ""
            return f"{d} {mon} {y}".strip()
        _cand_from_match(m, _txt, has_year_idx=3)
    for m in pat_MdY.finditer(s):
        def _txt(mm):
            mon = mm.group(1); d = mm.group(2); y = mm.group(3) or ""
            return f"{d} {mon} {y}".strip()
        _cand_from_match(m, _txt, has_year_idx=3)
    try:
        for pat in FILENAME_DATE_PATTERNS:
            for m in pat.finditer(s):
                parts = [p for p in m.groups() if p is not None]
                txt = " ".join(parts)
                dt = _parse_with_context(txt)
                if dt:
                    candidates.append((m.start(), dt, True))
    except NameError:
        pass
    if not candidates:
        return None
    with_year = [t for t in candidates if t[2] is True]
    if with_year:
        with_year.sort(key=lambda t: t[0], reverse=True)
        return with_year[0][1]
    return None

def _parse_month_year(s: str) -> _dt.date | None:
    if not s:
        return None
    m = FOOTER_MONTH_YEAR_PAT.search(s)
    if not m:
        return None
    month, year = m.group(1), m.group(2)
    return _try_parse_ambiguous_date(f"1 {month} {year}")

def _find_year_in_parents(path: Path) -> int | None:
    try:
        for p in [path.parent] + list(path.parents):
            name = (p.name or "").strip()
            m = YEAR_PAT.search(name)
            if m:
                return int(m.group(1))
    except Exception:
        pass
    return None

def _parse_day_month_no_year_with_year(s: str, year: int | None) -> _dt.date | None:
    if not s or not year:
        return None
    m = DAY_MONTH_NO_YEAR_PAT.search(s)
    if not m:
        return None
    d, mon = m.group(1), m.group(2)
    return _try_parse_ambiguous_date(f"{d} {mon} {year}")

def _extract_last_full_date(s: str) -> _dt.date | None:
    if not s: return None
    last = None
    for m in DATE_PAT.finditer(s):
        cand = m.group(0)
        dt = _try_parse_ambiguous_date(cand)
        if dt: last = dt
    return last

def _doc_date_from_footer_text(s: str) -> _dt.date | None:
    return _parse_month_year(s) or _extract_last_full_date(s)

def _doc_date_from_titleish_text(s: str) -> _dt.date | None:
    return _parse_month_year(s) or _extract_last_full_date(s)

def _doc_date_from_filename_or_folder(pdf_path: Path) -> _dt.date | None:
    stem = pdf_path.stem
    d = _parse_month_year(stem)
    if d:
        return d
    y = _find_year_in_parents(pdf_path)
    d2 = _parse_day_month_no_year_with_year(stem, y)
    if d2:
        return d2
    d3 = _date_from_filename_relaxed(stem, context_year=None) or _date_from_filename(stem)
    if d3:
        return d3
    for p in pdf_path.parents:
        d4 = _parse_month_year(p.name)
        if d4:
            return d4
    d5 = _date_from_parent_folders_relaxed(pdf_path)
    if d5:
        return d5
    return None

def resolve_doc_date_once(*, pdf_path: Path, page_obj: dict | None, pdf_meta_title: str | None = None) -> tuple[_dt.date | None, str | None, str | None]:
    if isinstance(page_obj, dict):
        foot = page_obj.get("footer")
        for v in ([foot] if isinstance(foot, str) else [foot.get(k) for k in ("date","text","raw")] if isinstance(foot, dict) else []):
            if isinstance(v, str):
                if FOOTER_MONTH_YEAR_PAT.search(v):
                    d = _parse_month_year(v); 
                    if d: return d, "footer", "month"
                d = _extract_last_full_date(v)
                if d: return d, "footer", "day"
    if isinstance(page_obj, dict):
        for key in ("title","subtitle","Title","Subtitle"):
            val = page_obj.get(key)
            if isinstance(val, str) and val.strip():
                if FOOTER_MONTH_YEAR_PAT.search(val):
                    d = _parse_month_year(val)
                    if d: return d, "title/subtitle", "month"
                d = _extract_last_full_date(val)
                if d: return d, "title/subtitle", "day"
    if pdf_meta_title:
        if FOOTER_MONTH_YEAR_PAT.search(pdf_meta_title):
            d = _parse_month_year(pdf_meta_title)
            if d: return d, "title/subtitle", "month"
        d = _extract_last_full_date(pdf_meta_title)
        if d: return d, "title/subtitle", "day"
    if page_obj is not None:
        d_any = _first_full_date_in_obj(page_obj)
        if d_any:
            return d_any, "page", "day"
    for p in [pdf_path.parent] + list(pdf_path.parents):
        d4 = _parse_month_year(p.name)
        if d4: return d4, "folder", "month"
    y = _find_year_in_parents(pdf_path)
    d2 = _parse_day_month_no_year_with_year(pdf_path.stem, y)
    if d2: return d2, "folder", "day"
    d3 = _date_from_filename_relaxed(pdf_path.stem, context_year=None) or _date_from_filename(pdf_path.stem)
    if d3: return d3, "filename", "day"
    d5 = _date_from_parent_folders_relaxed(pdf_path)
    if d5: return d5, "folder", "day"
    return None, None, None

# ─────────────────────────────────────────────────────────
# NEW: Merge JSON candidates
def _merge_page_objs(objs: List[Dict[str, Any]]) -> Dict[str, Any]:
    out = {"title": None, "subtitle": None, "narrative_text": None, "tables": [], "charts": [], "page_number": None}
    texts = []
    for o in objs:
        if not isinstance(o, dict): 
            continue
        out["title"] = out["title"] or o.get("title")
        out["subtitle"] = out["subtitle"] or o.get("subtitle")
        if o.get("narrative_text"):
            texts.append(str(o["narrative_text"]).strip())
        if isinstance(o.get("tables"), list):
            out["tables"].extend(o["tables"])
        if isinstance(o.get("charts"), list):
            out["charts"].extend(o["charts"])
        if o.get("page_number") and not out["page_number"]:
            out["page_number"] = o["page_number"]
        # keep debug if present
        if o.get("debug"):
            out.setdefault("debug", {}).update(o["debug"])
    if texts:
        out["narrative_text"] = "\n\n".join(t for t in texts if t)
    return out


#new_pritam whole function
def render_page_image_bytes(fitz_page, target_max_bytes: int, target_width_px: Optional[int] = None, target_height_px: Optional[int] = None, force_aspect: bool = True) -> (bytes, str, int, int, int):
    """
    Render a PDF page as bytes under target_max_bytes.
    Returns: (image_bytes, mime_type, width_px, height_px, est_image_tokens)
    """
    fallback_scales = [3.5, 3.0, 2.5, 2.0, 1.5, 1.25, 1.0, 0.85, 0.7]
    qualities = [90, 80, 70, 60]  # ← stop at 60 (remove 50)

    candidate_scales = []
    try:
        p_w = float(fitz_page.rect.width)
        p_h = float(fitz_page.rect.height)
    except Exception:
        p_w = None
        p_h = None

    if p_w and p_h and (target_width_px or target_height_px):
        scx = (target_width_px / p_w) if target_width_px else None
        scy = (target_height_px / p_h) if target_height_px else None
        if scx and scy:
            if force_aspect: sc = min(scx, scy); candidate_scales.append((sc, sc))
            else: candidate_scales.append((scx, scy))
        elif scx: candidate_scales.append((scx, scx))
        elif scy: candidate_scales.append((scy, scy))

    candidate_scales.extend([(s, s) for s in fallback_scales])

    # 1) PNG ladder
    for sc in candidate_scales:
        try:
            sx, sy = sc if isinstance(sc, tuple) else (sc, sc)
            pmap = fitz_page.get_pixmap(matrix=fitz.Matrix(sx, sy))
            b = pmap.tobytes("png")
            if len(b) <= target_max_bytes:
                w_px, h_px = pmap.width, pmap.height
                est_tokens = max(0, int(round((w_px * h_px) / IMAGE_TOKEN_DIVISOR)))
                return b, "image/png", int(w_px), int(h_px), est_tokens
        except Exception:
            pass

    # 2) JPEG ladder
    for sc in candidate_scales:
        try:
            sx, sy = sc if isinstance(sc, tuple) else (sc, sc)
            pmap = fitz_page.get_pixmap(matrix=fitz.Matrix(sx, sy))
        except Exception:
            continue
        for q in qualities:
            try:
                b = pmap.tobytes("jpg", quality=q)
                if len(b) <= target_max_bytes:
                    w_px, h_px = pmap.width, pmap.height
                    est_tokens = max(0, int(round((w_px * h_px) / IMAGE_TOKEN_DIVISOR)))
                    return b, "image/jpeg", int(w_px), int(h_px), est_tokens
            except Exception:
                continue

    raise RuntimeError(f"page image exceeds maximum bytes even after downscale/compression (>{target_max_bytes} bytes)")

# =========================================================
#            LLM invocation
# =========================================================

def _dump_page_image(pdf_stem: str, page_num: int, img_bytes: bytes, mime_type: str) -> Optional[str]:
    try:
        if not img_bytes:
            return None
        ext = ".png" if mime_type == "image/png" else ".jpg"
        SENT_IMG_DIR.mkdir(parents=True, exist_ok=True)
        pdf_dir = SENT_IMG_DIR / pdf_stem
        pdf_dir.mkdir(parents=True, exist_ok=True)
        out_path = pdf_dir / f"sent_page_{page_num:04d}{ext}"
        with open(out_path, "wb") as f:
            f.write(img_bytes)
        return str(out_path)
    except Exception:
        return None

def extract_info_from_image(
    png_bytes_unused: bytes,
    page_num: int,
    fitz_page=None,
    mime_type: str = "image/png",
    tables_only: bool = False,
) -> Dict[str, Any]:
    # Bedrock raw-image budget guard
    if "claude" in _llm_provider_label().lower() or "bedrock" in _llm_provider_label().lower():
        max_bytes = min(IMAGE_MAX_BYTES, _bedrock_raw_image_budget())
    else:
        max_bytes = IMAGE_MAX_BYTES

    img_bytes = b""
    media_type = mime_type or "image/png"
    sent_image_path = None
    sent_w = sent_h = 0
    est_image_tokens = 0

    if fitz_page is not None:
        try:
            img_bytes, media_type, sent_w, sent_h, est_image_tokens = render_page_image_bytes(
                fitz_page,
                max_bytes,
                target_width_px=IMAGE_TARGET_WIDTH_PX,
                target_height_px=IMAGE_TARGET_HEIGHT_PX,
                force_aspect=IMAGE_FORCE_ASPECT,
            )
            if PDF_DUMP_IMAGES and img_bytes:
                pdf_stem = None
                try:
                    parent_doc = getattr(fitz_page, "parent", None)
                    if parent_doc and getattr(parent_doc, "name", ""):
                        pdf_stem = Path(parent_doc.name).stem
                except Exception:
                    pdf_stem = None
                try:
                    sent_image_path = _dump_page_image(pdf_stem or "pdf", page_num, img_bytes, media_type)
                    logger.info("  page %d: dumped sent image → %s", page_num, sent_image_path)
                except Exception as _e:
                    logger.warning("  page %d: could not dump sent image: %s", page_num, _e)
        except Exception as e:
            logger.warning("Page %d: render under byte cap failed (%s); proceeding without image.", page_num, e)
            img_bytes = b""
            media_type = "image/png"
            sent_w = sent_h = 0
            est_image_tokens = 0
    else:
        img_bytes = png_bytes_unused or b""
        media_type = mime_type or "image/png"
        sent_w = sent_h = 0
        est_image_tokens = 0

    if not img_bytes:
        return {
            "page_number": page_num,
            "title": None,
            "subtitle": None,
            "narrative_text": None,
            "tables": [],
            "charts": [],
            "notes": None,
            "error": "no_image_bytes",
            "debug": {
                "sent_image_path": sent_image_path,
                "sent_image_mime": media_type,
                "sent_image_bytes": 0,
                "image_px": [int(sent_w) if sent_w else 0, int(sent_h) if sent_h else 0],
                "image_token_est": int(est_image_tokens or 0),
            },
        }

    last_err = None
    for attempt in range(1, MAX_MODEL_RETRIES + 1):
        try:
            t0 = time.time()
            prompt_to_send = TABLES_CHARTS_PROMPT if tables_only else BASE_PROMPT

            raw_text, usage = PROVIDER.generate_json(img_bytes, media_type, prompt_to_send)
            logger.info("LLM call page=%d took %.2fs (tables_only=%s)", page_num, time.time() - t0, tables_only)

            # estimate tokens from usage when available (but we'll recalc output tokens for tables-only)
            inp_tokens_est = usage.get("input_tokens") if usage else None
            out_tokens_est = usage.get("output_tokens") if usage else None
            if inp_tokens_est is None:
                inp_tokens_est = _approx_tokens_from_text(prompt_to_send) + int(est_image_tokens or 0)
            # we'll compute the out_tokens_est below based on the parsed / filtered content

            raw_text = re.sub(r"```(?:json)?|```", "", raw_text or "").strip()

            # ---------- MERGE ALL JSON CANDIDATES ----------
            cands = _extract_json_candidates(raw_text) or [raw_text]
            valid_objs = []
            for cand in cands:
                obj = _try_load_json(cand)
                if obj is None:
                    continue
                data = (
                    obj[0] if (isinstance(obj, list) and len(obj) == 1 and isinstance(obj[0], dict))
                    else (obj if isinstance(obj, dict) else {"value": obj})
                )
                valid_objs.append(data)

            parsed = _merge_page_objs(valid_objs) if valid_objs else None
            # ------------------------------------------------

            if parsed is None:
                # parsing failed; use the estimates we have
                TOK.add(int(inp_tokens_est or 0), int(out_tokens_est or 0))
                return {
                    "page_number": page_num,
                    "title": None,
                    "subtitle": None,
                    "narrative_text": None,
                    "tables": [],
                    "charts": [],
                    "notes": None,
                    "error": "json_parse_failed",
                    "raw_model_text": (raw_text or "")[:1000],
                    "debug": {
                        "sent_image_path": sent_image_path,
                        "sent_image_mime": media_type,
                        "sent_image_bytes": len(img_bytes) if img_bytes else 0,
                        "image_px": [int(sent_w) if sent_w else 0, int(sent_h) if sent_h else 0],
                        "image_token_est": int(est_image_tokens or 0),
                    },
                }

            # Normalize parsed into data dict
            data = (
                parsed[0]
                if (isinstance(parsed, list) and len(parsed) == 1 and isinstance(parsed[0], dict))
                else (parsed if isinstance(parsed, dict) else {"value": parsed})
            )
            data["page_number"] = page_num

            # Extract lists
            tables_list = data.get("tables", []) or []
            charts_list = data.get("charts", []) or []

            # IF tables_only mode, ALWAYS strip narrative/title/subtitle/notes from output
            if tables_only:
                data["narrative_text"] = None
                data["title"] = None
                data["subtitle"] = None
                data["notes"] = None

                # If no tables AND no charts -> treat as NO DETECTION
                if (not tables_list) and (not charts_list):
                    # record input tokens, zero output tokens (we didn't get structured content)
                    TOK.add(int(inp_tokens_est or 0), 0)
                    return {
                        "page_number": page_num,
                        "title": None,
                        "subtitle": None,
                        "narrative_text": None,
                        "tables": [],
                        "charts": [],
                        "notes": None,
                        "error": "no_tables_or_charts",
                        "debug": {
                            "sent_image_path": sent_image_path,
                            "sent_image_mime": media_type,
                            "sent_image_bytes": len(img_bytes) if img_bytes else 0,
                            "image_px": [int(sent_w) if sent_w else 0, int(sent_h) if sent_h else 0],
                            "image_token_est": int(est_image_tokens or 0),
                            "tables_only": True,
                            "raw_model_text_omitted": True,
                        },
                    }

                # Otherwise there are tables and/or charts — compute output token estimate only for the
                # table/chart JSON we'll actually keep.
                try:
                    minimal_json = json.dumps({"tables": tables_list, "charts": charts_list}, ensure_ascii=False)
                    out_tokens_for_struct = _approx_tokens_from_text(minimal_json)
                except Exception:
                    # fallback: use the provider estimate if we have it
                    out_tokens_for_struct = int(out_tokens_est or 0)

                # normalize table rows
                for tbl in tables_list:
                    if isinstance(tbl, dict):
                        normalize_table_rows(tbl)

                # ensure debug metadata
                data.setdefault("debug", {})
                data["debug"].update(
                    {
                        "sent_image_path": sent_image_path,
                        "sent_image_mime": media_type,
                        "sent_image_bytes": len(img_bytes) if img_bytes else 0,
                        "image_px": [int(sent_w) if sent_w else 0, int(sent_h) if sent_h else 0],
                        "image_token_est": int(est_image_tokens or 0),
                        "tables_only": True,
                        "raw_model_text_omitted": True,
                    }
                )

                # finally add token accounting using the minimal JSON estimate
                TOK.add(int(inp_tokens_est or 0), int(out_tokens_for_struct or 0))
                # return the filtered data (no narrative)
                return data

            # --- NON tables_only behavior: normalize tables and record tokens as usual ---
            for tbl in tables_list:
                if isinstance(tbl, dict):
                    normalize_table_rows(tbl)

            data.setdefault("debug", {})
            data["debug"].update(
                {
                    "sent_image_path": sent_image_path,
                    "sent_image_mime": media_type,
                    "sent_image_bytes": len(img_bytes) if img_bytes else 0,
                    "image_px": [int(sent_w) if sent_w else 0, int(sent_h) if sent_h else 0],
                    "image_token_est": int(est_image_tokens or 0),
                    "tables_only": False,
                    "raw_model_text_omitted": False,
                }
            )

            # If the provider didn't return output token counts, estimate from raw_text
            if out_tokens_est is None:
                out_tokens_est = _approx_tokens_from_text(raw_text or "")
            TOK.add(int(inp_tokens_est or 0), int(out_tokens_est or 0))

            return data

        except Exception as e:
            last_err = e
            logger.error(
                "%s call failed on page %d (attempt %d/%d): %s",
                _llm_provider_label(),
                page_num,
                attempt,
                MAX_MODEL_RETRIES,
                e,
            )
            time.sleep(RETRY_BASE_SLEEP * (2 ** (attempt - 1)))

    logger.error("All attempts failed on page %d → emitting minimal fallback", page_num)
    return {
        "page_number": page_num,
        "title": None,
        "subtitle": None,
        "narrative_text": None,
        "tables": [],
        "charts": [],
        "notes": None,
        "error": "invoke_failed",
        "err": str(last_err) if last_err else None,
        "debug": {
            "sent_image_path": sent_image_path,
            "sent_image_mime": media_type,
            "sent_image_bytes": len(img_bytes) if img_bytes else 0,
            "image_px": [int(sent_w) if sent_w else 0, int(sent_h) if sent_h else 0],
            "image_token_est": int(est_image_tokens or 0),
            "tables_only": bool(tables_only),
        },
    }


# =========================================================
#   JSON helpers & table normalization
# =========================================================
def normalize_table_rows(table: Dict[str, Any]) -> Dict[str, Any]:
    headers = table.get("headers") or table.get("columns")
    rows = table.get("rows", []) or table.get("data", [])
    if (not headers) and isinstance(rows, list) and any(isinstance(r, dict) for r in rows):
        key_order = []
        for r in rows:
            if isinstance(r, dict):
                for k in r.keys():
                    if k not in key_order:
                        key_order.append(k)
        headers = key_order
    if not headers:
        ncols = max((len(r) if isinstance(r, list) else len(getattr(r, "keys", lambda: [])()) )
                    for r in rows) if rows else 0
        headers = [f"col_{i+1}" for i in range(ncols)]
    fixed = []
    for row in rows:
        if isinstance(row, dict):
            fixed.append([row.get(h) for h in headers])
        elif isinstance(row, list):
            padded = row + [None] * (len(headers) - len(row))
            fixed.append(padded[:len(headers)])
        else:
            fixed.append([row] + [None] * (len(headers) - 1))
    table["headers"] = headers
    table["rows"] = fixed
    table.pop("data", None)
    table.pop("columns", None)
    return table

def _extract_json_candidates(text: str) -> List[str]:
    cands = []
    if not text: return cands
    n = len(text)
    for i, ch in enumerate(text):
        if ch not in "{[": continue
        stack = []
        for j in range(i, n):
            c = text[j]
            if c in "{[":
                stack.append(c)
            elif c in "}]":
                if not stack: break
                stack.pop()
                if not stack:
                    cands.append(text[i:j+1]); break
        if len(cands) >= 6: break
    return cands

def _try_load_json(s: str):
    try:
        return json.loads(s)
    except Exception:
        pass
    s2 = re.sub(r",\s*([\}\]])", r"\1", s)
    try:
        return json.loads(s2)
    except Exception:
        pass
    s3 = s2.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
    try:
        return json.loads(s3)
    except Exception:
        pass
    import ast
    try:
        return ast.literal_eval(s3)
    except Exception:
        return None

def _pick_best_json(raw_text: str):
    cands = _extract_json_candidates(raw_text) or [raw_text]
    best, best_score = None, -1
    for cand in cands:
        obj = _try_load_json(cand)
        if obj is None:
            continue
        data = (
            obj[0]
            if (isinstance(obj, list) and len(obj) == 1 and isinstance(obj[0], dict))
            else (obj if isinstance(obj, dict) else {"value": obj})
        )
        n_tbl = len(data.get("tables") or [])
        n_chr = len(data.get("charts") or [])
        score = 10 * n_tbl + 10 * n_chr + len(json.dumps(data, ensure_ascii=False)) // 1000
        if score > best_score:
            best, best_score = data, score
    return best

# =========================================================
#        Flattening helpers (incl. always-on letter logic)
# =========================================================
SENT_END_RE = re.compile(r'[\.!\?]["\']?\s*$')

ARTIFACT_WORDS = {
    "conversation","participant","moderator","speaker","speakers","transcript",
    "page","ppt","slide","continued","slide:","slide",
    "section","subsection","title","subtitle","heading","header",
    "content","text","paragraph","bullet_point","bullet"
}
_RE_ONLY_NUMBER = re.compile(r"^\s*\d+\s*$")
_RE_PAGE = re.compile(r"^\s*page\s*\d+\s*$", re.I)
_RE_ONLY_ARTIFACT = re.compile(r"^\s*(?:" + r"|".join(re.escape(w) for w in ARTIFACT_WORDS) + r")\s*$", re.I)
_RE_MULTI_NL = re.compile(r"\n{1,}")
_RE_SPACE_BEFORE_PUNC = re.compile(r"\s+([.,;:!?])")
_RE_MULTISPACE = re.compile(r"\s{2,}")
_RE_REPEATED_LINES = re.compile(r"(?m)^(?P<line>.+)\n(?P=line)\n+")
_PREFIX_AT_START = re.compile(r"^\s*.+?>.*?\n\n", flags=re.S)
_RE_ONLY_BULLETLITE = re.compile(r"^\s*(?:bullet_point|bullet|•|◦|▪|‣|–|-|—)\s*$", re.I)

TITLE_KEYS = ["title","page_title","pageTitle","heading","section_title","section","name"]
SUBTITLE_KEYS = ["subtitle","sub_title","page_subtitle","subheading","sub_section_title","sub_section","subsection"]

ALWAYS_KEEP_KEYS = {"title", "subtitle", "description", "period", "quote", "slogan"}

import string
_STOPWORDS = {
    "a","an","and","the","of","in","on","for","to","by","with","as","at","from",
    "is","are","be","was","were","this","that","these","those","it","its"
}

def _toksplit(s: str) -> list[str]:
    s = (s or "").strip()
    if not s:
        return []
    return [t for t in re.split(r"[^0-9A-Za-z]+", s) if t]

def _has_digits_or_symbols(s: str) -> bool:
    if not s:
        return False
    return bool(re.search(r"[0-9%₹$€:+/#]", s))

def is_trivial_heading(text: str) -> bool:
    t = (text or "").strip()
    if not t:
        return True
    if re.search(r"[.!?][\"')\]]?\s*$", t):
        return False
    toks = _toksplit(t)
    if not toks:
        return True
    if len(toks) > 5:
        return False
    if _has_digits_or_symbols(t):
        return False
    avg_len = sum(len(x) for x in toks) / max(1, len(toks))
    all_lower = all(x.islower() for x in toks)
    stop_or_short = sum(1 for x in toks if (x.lower() in _STOPWORDS or len(x) <= 3))
    stop_ratio = stop_or_short / len(toks)
    return (all_lower or stop_ratio >= 0.6 or avg_len <= 6)

def norm_compact(k: str) -> str:
    return re.sub(r"[^0-9a-zA-Z]+", "", (k or "")).strip().lower()

def ensure_text(x: Any) -> str:
    if x is None: return ""
    if isinstance(x, str): return x
    try: return json.dumps(x, ensure_ascii=False)
    except Exception: return str(x)

def word_count(text: str) -> int:
    return 0 if not text else len(re.findall(r"\S+", text))

def _ordinal(n: int) -> str:
    if 10 <= (n % 100) <= 20: suf = "th"
    else: suf = {1:"st",2:"nd",3:"rd"}.get(n % 10, "th")
    return f"{n}{suf}"

def _pdf_meta_year(doc) -> int | None:
    try:
        meta = getattr(doc, "metadata", {}) or {}
        for key in ("creationDate", "modDate", "CreationDate", "ModDate", "title"):
            v = meta.get(key) or ""
            m = re.search(r"(19|20)\d{2}", v)
            if m:
                return int(m.group(0))
    except Exception:
        pass
    return None

def _try_parse_partial_date(s: str, fallback_year: int | None) -> _dt.date | None:
    s = (s or "").strip()
    if not s:
        return None
    s = re.sub(r"\b(\d{1,2})(st|nd|rd|th)\b", r"\1", s, flags=re.I)
    if re.search(r"(19|20)\d{2}", s):
        return _try_parse_ambiguous_date(s)
    if fallback_year is None:
        fallback_year = _dt.date.today().year
    try:
        dt = _dtparse.parse(s, fuzzy=True, default=_dt.datetime(fallback_year, 1, 1)).date()
        if 1990 <= dt.year <= 2100:
            return dt
    except Exception:
        pass
    return None

def _date_from_titleish(title: str, context_year: int | None) -> _dt.date | None:
    if not title:
        return None
    m = re.search(r"\b(\d{1,2}(?:st|nd|rd|th)?\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*)\b", title, re.I)
    if not m:
        m = re.search(r"\b((Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2}(?:st|nd|rd|th)?)\b", title, re.I)
    if m:
        return _try_parse_ambiguous_date(title, context_year=context_year)
    m = DATE_PAT.search(title)
    if m:
        dt = _try_parse_ambiguous_date(m.group(0))
        if dt:
            return dt
        return _try_parse_partial_date(m.group(0), context_year)
    return None

def _only_date(x):
    return x[0] if isinstance(x, (tuple, list)) else x

def _normalize_spoken_date(dt: _dt.date) -> str:
    return f"{_ordinal(dt.day)} {dt.strftime('%B')} {dt.year}"

def _try_parse_ambiguous_date(s: str, context_year: int | None = None) -> _dt.date | None:
    s = (s or "").strip()
    if not s:
        return None
    has_year = bool(re.search(r"\b(19|20)\d{2}\b", s))
    for kw in ({"dayfirst": False, "yearfirst": False},
               {"dayfirst": True,  "yearfirst": False},
               {"dayfirst": False, "yearfirst": True}):
        try:
            dt = _dtparse.parse(
                s,
                fuzzy=True,
                default=_dt.datetime(2000, 1, 1),
                **kw
            ).date()
            if not has_year and dt.year == 2000:
                cy = context_year or _dt.date.today().year
                dt = _dt.date(cy, dt.month, dt.day)
            if 1990 <= dt.year <= 2100:
                return dt
        except Exception:
            pass
    return None

FILENAME_DATE_PATTERNS = [
    re.compile(r"\b(20\d{2})[-_.](\d{1,2})[-_.](\d{1,2})\b"),
    re.compile(r"\b(\d{1,2})[-_.](\d{1,2})[-_.](20\d{2})\b"),
    re.compile(r"\b(20\d{2})(\d{2})(\d{2})\b"),
    re.compile(r"\b(\d{2})(\d{2})(20\d{2})\b"),
    re.compile(r"\b((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*)[\s._-]*(\d{1,2})[\s._-]*(20\d{2})\b", re.I),
    re.compile(r"\b(\d{1,2})[\s._-]*((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*)[\s._-]*(20\d{2})\b", re.I),
]

def _date_from_filename(stem: str) -> _dt.date | None:
    base = stem or ""
    for pat in FILENAME_DATE_PATTERNS:
        m = pat.search(base)
        if not m:
            continue
        parts = [p for p in m.groups() if p is not None]
        guess = " ".join(parts)
        dt = _try_parse_ambiguous_date(guess)
        if dt:
            return dt
    return None

def clean_text_flow(text: str, source_stem: Optional[str] = None) -> str:
    if not text: return ""
    t = _RE_REPEATED_LINES.sub(lambda m: m.group("line") + "\n", text)
    kept = []
    for line in t.splitlines():
        s = line.strip()
        if not s: continue
        # ---- changed: KEEP numbers-only lines (helps small labels & table-like text)
        if _RE_ONLY_NUMBER.match(s):
            kept.append(s); continue
        if _RE_ONLY_BULLETLITE.match(s): continue
        if _RE_PAGE.match(s): continue
        # if _RE_ONLY_ARTIFACT.match(s): continue
        if len(s) <= 2 and s.isalpha(): continue
        kept.append(s)
    if not kept: return ""
    joined = " ".join(kept)
    joined = _RE_MULTI_NL.sub(" ", joined)
    joined = _RE_MULTISPACE.sub(" ", joined).strip()
    joined = _RE_SPACE_BEFORE_PUNC.sub(r"\1", joined)
    joined = re.sub(r"([.,;:!?])([^\s])", r"\1 \2", joined)
    return joined.strip()

def rows_to_markdown(headers: List[str], rows: List[List[Any]]) -> str:
    if not headers:
        ncols = max((len(r) for r in (rows or []) if isinstance(r, (list, tuple))), default=0)
        headers = [f"col_{i+1}" for i in range(ncols)]
    hdrs = [("" if h is None else str(h)) for h in headers]
    sep = ["---"] * len(hdrs)
    lines = []
    lines.append("| " + " | ".join(hdrs) + " |")
    lines.append("| " + " | ".join(sep) + " |")
    for r in rows or []:
        if not isinstance(r, (list, tuple)):
            cells = ["" if r is None else str(r)] + [""] * (len(hdrs) - 1)
        else:
            cells = [("" if c is None else str(c)) for c in r]
            if len(cells) < len(hdrs):
                cells += [""] * (len(hdrs) - len(cells))
            elif len(cells) > len(hdrs):
                cells = cells[: len(hdrs) - 1] + [" | ".join(cells[len(hdrs) - 1 :])]
        cells = [c.replace("|", "\\|") for c in cells]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)

def chunk_text_by_words(text: str, max_words: int = MAX_WORDS) -> List[str]:
    if not text: return []
    words = re.findall(r"\S+", text)
    if len(words) <= max_words:
        return [" ".join(words)]
    return [" ".join(words[i : i + max_words]) for i in range(0, len(words), max_words)]

def split_long_by_chars(s: str, limit: int = CHUNK_CHAR_LIMIT) -> List[str]:
    if not s or len(s) <= limit: return [s or ""]
    out = []
    i, n = 0, len(s)
    while i < n:
        j = min(n, i + limit)
        if j < n:
            k = s.rfind(" ", i + int(limit * 0.6), j)
            if k == -1: k = j
            out.append(s[i:k].strip()); i = k
        else:
            out.append(s[i:j].strip()); i = j
    return [t for t in out if t]

LETTER_KEYS_HINT = {
    "header", "addressee", "company_details", "content",
    "closing", "signatories", "cc", "footer"
}
LETTER_BLOCK_KEYS = {"recipient", "addressee", "address_to", "to"}
LETTER_FIELDS = {"department", "company", "address", "attn", "attention", "name", "designation"}

def _emit_kv_table(title: str, data: Dict[str, Any], meta: Dict[str, Any], page: Optional[int]):
    headers = ["Field", "Value"]
    rows = [[k, v] for k, v in (data or {}).items()]
    nm = dict(meta); nm.update({"title": title, "headers": headers, "rows": rows})
    return {"type": "table", "text": title or "", "meta": nm, "page": page}

def _emit_signatories_table(signatories: List[Dict[str, Any]], meta: Dict[str, Any], page: Optional[int]):
    cols: List[str] = []
    for s in signatories:
        if isinstance(s, dict):
            for k in s.keys():
                if k not in cols:
                    cols.append(k)
    headers = [h.replace("_"," ").title() for h in cols] if cols else ["Name","Designation","Digital Signature"]
    rows = []
    for s in signatories:
        if isinstance(s, dict):
            rows.append([s.get(k, "") for k in cols])
        else:
            rows.append([str(s)])
    nm = dict(meta); nm.update({"title": "Signatories", "headers": headers, "rows": rows})
    return {"type": "table", "text": "Signatories", "meta": nm, "page": page}

def _looks_letter_like(d: Dict[str, Any]) -> bool:
    if not isinstance(d, dict): return False
    keys = set(k for k in d.keys() if isinstance(k, str))
    hits = len(keys & LETTER_KEYS_HINT)
    content = d.get("content")
    has_paragraphs = isinstance(content, list) and any(isinstance(x, dict) and "paragraph" in x for x in content)
    return hits >= 2 or (hits >= 1 and has_paragraphs)

def _maybe_emit_letter_like(d: Dict[str, Any], meta: Dict[str, Any], page: Optional[int]):
    if not _looks_letter_like(d): return None
    out: List[Dict[str, Any]] = []
    for key, title in [
        ("header", "Header"),
        ("addressee", "Addressee"),
        ("company_details", "Company Details"),
        ("footer", "Footer"),
    ]:
        if key in d and isinstance(d[key], dict):
            out.append(_emit_kv_table(title, d[key], meta, page))
    if isinstance(d.get("content"), list):
        for it in d["content"]:
            text = it.get("paragraph") if isinstance(it, dict) else it
            cleaned = clean_text_flow(ensure_text(text))
            if cleaned:
                out.append({"type": "narrative", "text": cleaned, "meta": dict(meta), "page": page})
    if isinstance(d.get("closing"), dict):
        closelines = []
        for k in ("text","signature"):
            if k in d["closing"]:
                s = clean_text_flow(ensure_text(d["closing"][k]))
                if s: closelines.append(s)
        if closelines:
            out.append({"type": "narrative", "text": " ".join(closelines), "meta": dict(meta), "page": page})
    if isinstance(d.get("signatories"), list) and d["signatories"]:
        out.append(_emit_signatories_table(d["signatories"], meta, page))
    if isinstance(d.get("cc"), list) and d["cc"]:
        headers = ["CC"]
        rows = [[ensure_text(x)] for x in d["cc"]]
        nm = dict(meta); nm.update({"title": "CC", "headers": headers, "rows": rows})
        out.append({"type": "table", "text": "CC", "meta": nm, "page": page})
    return out if out else None

def iter_extract_collect(
    obj: Any,
    meta: Optional[Dict[str, Any]] = None,
    page: Optional[int] = None,
    candidates: Optional[List[Dict[str, Any]]] = None,
):
    if candidates is None:
        candidates = []
    if meta is None:
        meta = {}
    if obj is None:
        return

    if isinstance(obj, (str, int, float, bool)):
        s = ensure_text(obj)
        if isinstance(obj, str) and DATE_PAT.fullmatch(s.strip()) and (meta or {}).get("published_date"):
            return
        cleaned = clean_text_flow(s)
        if cleaned:
            yield {"type": "narrative", "text": cleaned, "meta": dict(meta), "page": page}
        return

    if isinstance(obj, list):
        for it in obj:
            yield from iter_extract_collect(it, meta=meta, page=page, candidates=candidates)
        return

    if isinstance(obj, dict):
        DROP_KEYS = {
            "debug","raw_model_text","error","err",
            "sent_image_path","sent_image_mime","sent_image_bytes",
            "image_px","image_token_est"
        }
        obj = {k: v for k, v in obj.items() if k not in DROP_KEYS}
        for k, v in obj.items():
            if "page" in str(k).lower() and isinstance(v, (int, str)):
                try: page = int(v)
                except Exception: pass
                break

        LETTERY_KEYS = {"recipient", "addressee", "sender", "company_details", "header", "closing", "signatories"}
        present_letter_keys = LETTERY_KEYS & set(obj.keys())
        for lk in list(present_letter_keys):
            v = obj.get(lk)
            if isinstance(v, dict) and v:
                rows = [[k2, v2] for k2, v2 in v.items()]
                nm = dict(meta)
                nm.update({"title": lk.replace("_", " ").title(),"headers": ["Field", "Value"],"rows": rows,"force_emit": True})
                yield {"type": "table", "text": lk, "meta": nm, "page": page}
            elif isinstance(v, list) and v and all(isinstance(x, dict) for x in v):
                headers = sorted({kk for x in v for kk in x.keys()})
                rows = [[x.get(h, "") for h in headers] for x in v]
                nm = dict(meta)
                nm.update({"title": lk.replace("_", " ").title(),"headers": headers,"rows": rows,"force_emit": True})
                yield {"type": "table", "text": lk, "meta": nm, "page": page}

        for k_top, v_top in list(obj.items()):
            if isinstance(k_top, str) and k_top.strip().lower() in {"content", "body", "message", "letter_body"}:
                if isinstance(v_top, list) and all(isinstance(x, dict) for x in v_top):
                    merged_body = []
                    for item in v_top:
                        text_val = (item.get("text") or item.get("paragraph") or "").strip()
                        if text_val:
                            merged_body.append(text_val)
                    if merged_body:
                        nm = dict(meta)
                        nm.update({"title": "Body", "force_emit": True})
                        yield {"type": "narrative", "text": "\n\n".join(M for M in merged_body), "meta": nm, "page": page}
                yield from iter_extract_collect(v_top, meta=meta, page=page, candidates=candidates)

        # ---- handle model tables shaped as {"type":"table","data":[...]} with sections/items/columns ----
        if str(obj.get("type", "")).lower() == "table" and isinstance(obj.get("data"), list):
            for seg in obj["data"]:
                if not isinstance(seg, dict):
                    continue
                title = seg.get("section") or obj.get("title") or obj.get("name") or "Table"

                # Case A: list of dicts
                if isinstance(seg.get("items"), list) and seg["items"] and all(isinstance(x, dict) for x in seg["items"]):
                    seen = []
                    for row in seg["items"]:
                        for k in row.keys():
                            if k not in seen:
                                seen.append(k)
                    headers = [h.replace("_", " ").title() for h in seen] if seen else ["Field", "Value"]
                    rows = [[row.get(k, "") for k in seen] for row in seg["items"]]
                    nm = dict(meta); nm.update({"title": title, "headers": headers, "rows": rows})
                    yield {"type": "table", "text": title, "meta": nm, "page": page}
                    continue

                # Case A2: list of strings -> single-column table (RECOVERY)
                if isinstance(seg.get("items"), list) and seg["items"] and all(isinstance(x, str) for x in seg["items"]):
                    headers = ["Value"]
                    rows = [[x] for x in seg["items"]]
                    nm = dict(meta); nm.update({"title": title, "headers": headers, "rows": rows})
                    yield {"type": "table", "text": title, "meta": nm, "page": page}
                    continue

                # Case B: columns pivot
                if isinstance(seg.get("columns"), list) and seg["columns"] and all(isinstance(c, dict) for c in seg["columns"]):
                    cols = seg["columns"]
                    headers, arrays = [], []
                    for c in cols:
                        h = c.get("header") or c.get("name") or ""
                        headers.append(str(h))
                        items = c.get("items") or c.get("values") or []
                        if not isinstance(items, list): items = [items]
                        arrays.append([("" if v is None else str(v)) for v in items])
                    rows = [[cell for cell in r] for r in zip_longest(*arrays, fillvalue="")]
                    nm = dict(meta); nm.update({"title": title, "headers": headers, "rows": rows})
                    yield {"type": "table", "text": title, "meta": nm, "page": page}
                    continue
            return

        keys = set(k.lower() for k in obj.keys())
        if (keys & {"rows", "headers", "cells", "columns", "data"}) and any(
            isinstance(obj.get(x), (list, dict)) for x in ("rows", "cells", "data", "headers")
        ):
            headers = obj.get("headers") or obj.get("columns") or []
            rows = obj.get("rows") or obj.get("cells") or obj.get("data") or []
            title = obj.get("title") or obj.get("name") or ""
            nm = dict(meta); nm.update({"title": title, "headers": headers, "rows": rows})
            yield {"type": "table", "text": title, "meta": nm, "page": page}
            return

        if ("type" in obj and str(obj.get("type")).lower() == "chart") or any(
            k in keys for k in ("series", "data_series", "data", "xaxis", "yaxis", "chart")
        ):
            chart_title = obj.get("title") or obj.get("name") or ""
            chart_type = obj.get("chart_type") or obj.get("chartType") or obj.get("chart") or obj.get("type")
            ds = obj.get("data_series") or obj.get("dataSeries") or obj.get("series") or obj.get("data") or None
            if isinstance(ds, list) and ds and all(isinstance(x, dict) for x in ds):
                if all(("name" in x and "value" in x) for x in ds):
                    ds = [{"name": x["name"], "values": [x.get("value")]} for x in ds]
            nm = dict(meta)
            if chart_title: nm["title"] = chart_title
            nm["chart_type"] = chart_type
            nm["data_series"] = ds
            yield {"type": "chart", "text": chart_title or "", "meta": nm, "page": page, "raw_obj": obj}
            return

        text_fields = []
        keep_keys_present = False
        for k, v in obj.items():
            if (
                isinstance(v, str)
                and len(v) < 5000
                and any(tok in k.lower() for tok in ("text","content","narrative","paragraph","title","subtitle","note","description","speaker","period","quote","slogan"))
            ):
                text_fields.append((k, v))
                if k.strip().lower() in ALWAYS_KEEP_KEYS:
                    keep_keys_present = True

        if text_fields:
            combined = "\n\n".join(v for _, v in text_fields)
            cleaned = clean_text_flow(combined)
            new_meta = dict(meta)
            if "title" in obj and isinstance(obj["title"], str): new_meta["title"] = obj["title"]
            if "subtitle" in obj and isinstance(obj["subtitle"], str): new_meta["subtitle"] = obj["subtitle"]
            if page == 1 or keep_keys_present: new_meta["force_emit"] = True
            if cleaned:
                yield {"type": "narrative", "text": cleaned, "meta": new_meta, "page": page}
            for k2, v2 in obj.items():
                if not any(k2 == tf[0] for tf in text_fields):
                    yield from iter_extract_collect(v2, meta=new_meta, page=page, candidates=candidates)
            return

        TITLE_KEYSET = set(norm_compact(k) for k in TITLE_KEYS)
        SUBTITLE_KEYSET = set(norm_compact(k) for k in SUBTITLE_KEYS)
        for k, v in obj.items():
            new_meta = dict(meta)
            nk = norm_compact(k)
            if nk in TITLE_KEYSET and isinstance(v, str):
                new_meta["title"] = v
            if nk in SUBTITLE_KEYSET and isinstance(v, str):
                new_meta["subtitle"] = v
            yield from iter_extract_collect(v, meta=new_meta, page=page, candidates=candidates)
        return

# ---------------- finalize blocks -> chunks ----------------
def _strip_prefix_head(text: str) -> str:
    return _PREFIX_AT_START.sub("", text or "", count=1).strip()

def _too_small_to_emit(tokenized_word_count: int, meta: Optional[Dict[str, Any]]) -> bool:
    if meta and meta.get("force_emit"):
        return False
    return tokenized_word_count < MIN_CHUNK_TOKENS

def first_nonempty(meta: Dict[str, Any], candidates: List[str]) -> Optional[str]:
    for c in candidates:
        v = meta.get(c)
        if v and isinstance(v, str) and v.strip():
            return v.strip()
    return None

def build_chunk_prefix(source_stem: str, meta: Dict[str, Any]) -> str:
    parts = []
    if source_stem: parts.append(source_stem)
    page_title = first_nonempty(meta, ["page_title","title","pageTitle","heading"])
    page_sub   = first_nonempty(meta, ["page_subtitle","subtitle","sub_title","subheading"])
    if page_title and page_sub: parts.append(f"{page_title} {page_sub}")
    elif page_title: parts.append(page_title)
    if meta.get("published_month"):
        parts.append(f"published_month: {meta['published_month']}")
    elif meta.get("published_date"):
        parts.append(f"published_date: {meta['published_date']}")
    topic_title = first_nonempty(meta, ["section","section_title","section_name","topic","topic_title"])
    topic_sub   = first_nonempty(meta, ["subsection","sub_section","subheading","sub_section_title","topic_subtitle"])
    if topic_title and topic_sub: parts.append(f"{topic_title} {topic_sub}")
    elif topic_title: parts.append(topic_title)
    return (" > ".join([p for p in parts if p]) + "\n\n") if parts else ""

def _page_titles_from_obj(page_obj: dict) -> tuple[Optional[str], Optional[str]]:
    if not isinstance(page_obj, dict):
        return None, None
    title = None
    subtitle = None
    for k in ("page_title", "title", "pageTitle", "heading", "section_title", "name"):
        v = page_obj.get(k)
        if isinstance(v, str) and v.strip():
            title = v.strip(); break
    for k in ("page_subtitle", "subtitle", "sub_title", "subheading", "sub_section_title"):
        v = page_obj.get(k)
        if isinstance(v, str) and v.strip():
            subtitle = v.strip(); break
    return title, subtitle

def _normalize_for_dedup(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^\w\s₹$€%.-]", "", s)
    return s.strip()

_BULLET_LINE_RE = re.compile(r"(?m)^\s*(?:[\u2022•◦\-–]|[0-9]{1,3}[.)]|[A-Za-z][.)])\s+")
_POINT_LINE_RE = re.compile(
    r"""(?mx)
    ^(?P<indent>[ \t]{0,20})
    (?P<bullet>[\u2022•\-–]|\d{1,2}[.)]|[A-Za-z][.)]|[o○◦])
    [ \t]+
    (?P<body>.+?)\s*$
    """
)
_SUB_BULLETS = {"o", "○", "◦"}
_TOP_INDENT_MAX = 3

FULL_DATE_NAME_PAT = re.compile(
    r"\b(?:\d{1,2}(?:st|nd|rd|th)?\s+"
    r"(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
    r"Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
    r"\s*,?\s*(19|20)\d{2}"
    r"|"
    r"(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
    r"Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?)"
    r"\s+\d{1,2}(?:st|nd|rd|th)?\s*,?\s*(19|20)\d{2}"
    r"|"
    r"\d{1,2}[/-]\d{1,2}[/-](19|20)\d{2})\b", re.IGNORECASE
)

def _first_full_date_in_obj(obj) -> _dt.date | None:
    stack = [obj]
    while stack:
        cur = stack.pop()
        if isinstance(cur, dict):
            stack.extend(cur.values())
        elif isinstance(cur, list):
            stack.extend(cur)
        elif isinstance(cur, str):
            m = FULL_DATE_NAME_PAT.search(cur)
            if m:
                dt = _try_parse_ambiguous_date(m.group(0))
                if dt: return dt
    return None

def split_points(text: str, *, min_unit_words: int = 28, hard_max_words: int = 180) -> list[str]:
    if not text:
        return []
    lines = [ln.rstrip() for ln in re.sub(r"\r\n?", "\n", text).split("\n") if ln.strip()]
    units: list[str] = []
    cur: list[str] = []
    cur_root_indent: int | None = None
    for ln in lines:
        m = _POINT_LINE_RE.match(ln)
        if m:
            indent = len(m.group("indent").expandtabs(4))
            bullet = m.group("bullet")
            body = (m.group("body") or "").strip()
            is_sub_bullet = bullet in _SUB_BULLETS or indent > _TOP_INDENT_MAX
            is_top_bullet  = (indent <= _TOP_INDENT_MAX) and (bullet not in _SUB_BULLETS)
            if is_top_bullet:
                if cur:
                    units.append(" ".join(cur).strip())
                    cur = []
                cur_root_indent = indent
                cur.append(body)
            else:
                if not cur:
                    cur_root_indent = indent
                    cur.append(body)
                else:
                    cur.append(body)
        else:
            if not cur:
                cur_root_indent = 0
            cur.append(ln.strip())
    if cur:
        units.append(" ".join(cur).strip())
    glued: list[str] = []
    i = 0
    while i < len(units):
        buf = units[i].strip()
        wc = len(re.findall(r"\S+", buf))
        while wc < min_unit_words and (i + 1) < len(units):
            nxt = units[i + 1].strip()
            buf = (buf + " " + nxt).strip()
            i += 1
            wc = len(re.findall(r"\S+", buf))
        glued.append(buf)
        i += 1
    def _norm(s: str) -> str:
        s = re.sub(r"[^\w\s]", "", s.lower())
        return re.sub(r"\s+", " ", s).strip()
    deduped: list[str] = []
    seen: set[str] = set()
    for u in glued:
        key = _norm(u)
        if key and key not in seen:
            seen.add(key)
            deduped.append(u)
    final_units: list[str] = []
    for u in deduped:
        words = re.findall(r"\S+", u)
        if len(words) <= hard_max_words:
            final_units.append(u)
        else:
            for j in range(0, len(words), hard_max_words):
                final_units.append(" ".join(words[j:j + hard_max_words]))
    return [u for u in final_units if u]

def _append_text_chunk(prelim, source_stem, seq, text_with_prefix, page_b, min_words=8):
    wc = len(re.findall(r"\S+", text_with_prefix))
    if prelim and wc < min_words:
        for j in range(len(prelim) - 1, -1, -1):
            if prelim[j].get("type") in ("text", "narrative"):
                sep = "" if prelim[j]["text"].endswith((" ", "\n")) else " "
                prelim[j]["text"] = (prelim[j]["text"] + sep + text_with_prefix).strip()
                return seq
    seq += 1
    prelim.append({
        "id": f"{source_stem}_chunk_{seq:05d}",
        "type": "text",
        "text": text_with_prefix,
        "meta": {"source": source_stem, "page_range": [page_b, page_b] if page_b is not None else []},
    })
    return seq

def finalize_blocks_to_chunks(blocks: List[Dict[str, Any]], source_stem: str) -> List[Dict[str, Any]]:
    prelim: List[Dict[str, Any]] = []
    seq = 0
    i, n = 0, len(blocks)

    while i < n:
        b = blocks[i]
        btype = b.get("type", "narrative")
        page_b = b.get("page")

        if btype in ("narrative", "text"):
            candidate = ensure_text(b.get("text", ""))
            cleaned_candidate = clean_text_flow(candidate, source_stem) or candidate.strip()
            wc = word_count(cleaned_candidate)
            next_block = blocks[i + 1] if (i + 1) < n else None
            prev_block = blocks[i - 1] if i - 1 >= 0 else None

            # 1) Short caption -> attach to NEXT structured block (table/chart) on same page
            if wc <= TITLE_MERGE_MAX_WORDS and page_b is not None and next_block and (page_b == next_block.get("page")):
                if next_block.get("type") in ("table", "chart"):
                    nm = next_block.get("meta") or {}
                    prev_txt = nm.get("title_prefix") or ""
                    nm["title_prefix"] = ((prev_txt + " " + cleaned_candidate).strip() if prev_txt else cleaned_candidate)
                    next_block["meta"] = nm
                    i += 1
                    continue

            # 2) If it looks like a trivial heading, prefer gluing to NEXT narrative on same page
            if is_trivial_heading(cleaned_candidate) and page_b is not None and next_block and (page_b == next_block.get("page")):
                if next_block.get("type") in ("narrative","text"):
                    nxt_txt = ensure_text(next_block.get("text","")).strip()
                    next_block["text"] = (cleaned_candidate + "\n\n" + nxt_txt).strip()
                    i += 1
                    continue

            # 3) Else try gluing to PREVIOUS structured block on same page
            if is_trivial_heading(cleaned_candidate) and page_b is not None and prev_block and (page_b == prev_block.get("page")):
                if prev_block.get("type") in ("table","chart"):
                    nm = prev_block.get("meta") or {}
                    prev_txt = nm.get("title_prefix") or ""
                    nm["title_prefix"] = ((prev_txt + " " + cleaned_candidate).strip() if prev_txt else cleaned_candidate)
                    prev_block["meta"] = nm
                    i += 1
                    continue

            # 4) CHANGED: do NOT drop trivial headings. Fall through and emit as narrative.

            # 5) Normal narrative processing
            if not cleaned_candidate:
                i += 1
                continue
            if _RE_ONLY_BULLETLITE.match(cleaned_candidate):
                i += 1
                continue
            if cleaned_candidate.strip().lower() in {"bullet_point","bullet"}:
                i += 1
                continue
            if DATE_PAT.fullmatch(cleaned_candidate.strip()):
                i += 1
                continue

            block_meta = b.get("meta") or {}
            prefix = build_chunk_prefix(source_stem, block_meta)

            for point in split_points(cleaned_candidate, min_unit_words=28, hard_max_words=180):
                text_with_prefix = (prefix + point).strip() if prefix else point
                for piece in split_long_by_chars(text_with_prefix, CHUNK_CHAR_LIMIT):
                    seq = _append_text_chunk(prelim, source_stem, seq, piece, page_b)

            i += 1
            continue

        if btype == "table":
            seq += 1
            headers = (b.get("meta") or {}).get("headers")
            rows = (b.get("meta") or {}).get("rows")
            page = b.get("page")
            title_prefix = (b.get("meta") or {}).get("title_prefix") or ""
            title = (b.get("meta") or {}).get("title") or ensure_text(b.get("text", ""))
            if title_prefix:
                title = (title_prefix.strip() + " " + title).strip()
            md = (rows_to_markdown(headers, rows) if headers and rows else ensure_text(b.get("text", "")))
            block_meta = b.get("meta") or {}
            prefix = build_chunk_prefix(source_stem, block_meta)
            text_value = (prefix + title + "\n\n" + md).strip() if prefix else (title + "\n\n" + md)
            prelim.append({
                "id": f"{source_stem}_chunk_{seq:05d}",
                "type": "table",
                "text": text_value,
                "meta": {"source": source_stem, "page_range": [page, page] if page is not None else []},
            })
            i += 1
            continue

        if btype == "chart":
            seq += 1
            chart_meta = b.get("meta") or {}
            raw_obj = b.get("raw_obj") or {}
            ds = chart_meta.get("data_series") or raw_obj.get("data_series") or raw_obj.get("series") or raw_obj.get("data") or None
            if isinstance(ds, list) and ds and all(isinstance(x, dict) and "name" in x and "value" in x for x in ds):
                ds = [{"name": s["name"], "values": [s.get("value")]} for s in ds]
            chart_title = chart_meta.get("title") or ensure_text(b.get("text", ""))
            chart_type = chart_meta.get("chart_type") or raw_obj.get("chart_type") or raw_obj.get("chart") or ""
            title_prefix = chart_meta.get("title_prefix") or ""
            full_title = ((title_prefix.strip() + " " + chart_title).strip()) if title_prefix else chart_title

            lines: List[str] = []
            block_meta = b.get("meta") or {}
            prefix = build_chunk_prefix(source_stem, block_meta)
            if prefix:
                lines.append(prefix.strip())
            if full_title:
                lines.append(full_title)
            if chart_type:
                lines.append("")
                lines.append(f"chart_type: {chart_type}")
            if isinstance(ds, list) and ds:
                for s in ds:
                    if isinstance(s, dict):
                        sname = s.get("name") or s.get("title") or s.get("label") or "series"
                        lines.append("")
                        lines.append(f"Series: {sname}")
                        series_vals = s.get("data") or s.get("values") or s.get("points") or []
                        if isinstance(series_vals, list) and series_vals:
                            try:
                                lines.append(json.dumps(series_vals, ensure_ascii=False))
                            except Exception:
                                lines.append(str(series_vals))
                        else:
                            lines.append("[]")
                    else:
                        lines.append(str(s))
            else:
                lines.append("")
                lines.append("No structured data_series found; raw dump follows:")
                try:
                    lines.append(json.dumps(raw_obj, ensure_ascii=False, indent=2))
                except Exception:
                    lines.append(str(raw_obj))

            text_value = "\n".join(lines).strip()
            page = b.get("page")
            prelim.append({
                "id": f"{source_stem}_chunk_{seq:05d}",
                "type": "chart",
                "text": text_value,
                "meta": {"source": source_stem, "page_range": [page, page] if page is not None else []},
            })
            i += 1
            continue

        raw = ensure_text(b.get("text", ""))
        cleaned = clean_text_flow(raw, source_stem) or raw.strip()
        if cleaned:
            block_meta = b.get("meta") or {}
            prefix = build_chunk_prefix(source_stem, block_meta)
            for point in split_points(cleaned, min_unit_words=28, hard_max_words=180):
                text_with_prefix = (prefix + point).strip() if prefix else point
                for piece in split_long_by_chars(text_with_prefix, CHUNK_CHAR_LIMIT):
                    seq = _append_text_chunk(prelim, source_stem, seq, piece, b.get("page"))
        i += 1

    def _collapse_page(page_range):
        if isinstance(page_range, list):
            if len(page_range) >= 2:
                return page_range[1]
            if len(page_range) == 1:
                return page_range[0]
        return None

    final: List[Dict[str, Any]] = []
    for c in prelim:
        src = (c.get("meta") or {}).get("source", "")
        if isinstance(src, str) and src.lower().endswith(".json"): src = src[:-5]
        pr = (c.get("meta") or {}).get("page_range", [])
        pg = _collapse_page(pr)
        meta = {"source": src, "page": (pg if pg is not None else 1)}
        if (c.get("meta") or {}).get("published_date"):
            meta["published_date"] = (c["meta"]["published_date"])
        if (c.get("meta") or {}).get("published_date_source"):
            meta["published_date_source"] = (c["meta"]["published_date_source"])
        if (c.get("meta") or {}).get("published_month"):
            meta["published_month"] = (c["meta"]["published_month"])
        final.append({"id": c["id"], "type": c["type"], "text": c["text"], "meta": meta})
    return final

# =========================================================
#       Per-page cap enforcement
# =========================================================
def enforce_per_page_cap(page_chunks, cap, source_stem, page_no):
    if cap is None or cap <= 0 or len(page_chunks) <= cap:
        return page_chunks
    structured_idx = [i for i, ch in enumerate(page_chunks) if ch.get("type") in ("table", "chart")]
    text_idx       = [i for i, ch in enumerate(page_chunks) if ch.get("type") not in ("table", "chart")]
    keep_order = []
    for i in structured_idx:
        if len(keep_order) >= cap - 1: break
        keep_order.append(i)
    for i in text_idx:
        if len(keep_order) >= cap - 1: break
        if i not in keep_order:
            keep_order.append(i)
    keep_order.sort()
    kept = [page_chunks[i] for i in keep_order]
    overflow = [page_chunks[i] for i in range(len(page_chunks)) if i not in keep_order]
    if not overflow:
        return kept
    last_text_pos = None
    for j in range(len(kept) - 1, -1, -1):
        if kept[j].get("type") not in ("table", "chart"):
            last_text_pos = j; break
    parts = []
    for ch in overflow:
        ctype = (ch.get("type") or "text").lower()
        meta = ch.get("meta") or {}
        title = meta.get("title") or ""
        if not title:
            txt = (ch.get("text") or "").strip().splitlines()
            title = txt[0][:120] if txt else ""
        if ctype == "table": header = f"## Table: {title}" if title else "## Table"
        elif ctype == "chart": header = f"## Chart: {title}" if title else "## Chart"
        else: header = f"## Text: {title}" if title else "## Text"
        body = (ch.get("text") or "").strip()
        if body:
            parts.append(f"{header}\n\n{body}")
    overflow_text = ("\n\n" + "\n\n".join(parts)) if parts else ""
    if last_text_pos is not None:
        kept[last_text_pos]["text"] = (kept[last_text_pos].get("text") or "") + overflow_text
        m = kept[last_text_pos].setdefault("meta", {})
        m["overflow_consolidated"] = True
        m["overflow_count"] = len(overflow)
        return kept[:cap]
    overflow_chunk = {
        "id": f"{source_stem}_overflow_{page_no:05d}",
        "type": "text",
        "text": overflow_text.lstrip(),
        "meta": {"source": source_stem,"page": page_no,"overflow_consolidated": True,"overflow_count": len(overflow)},
    }
    kept.append(overflow_chunk)
    return kept[:cap]

# =========================================================
#       Page-aware stitcher
# =========================================================
def ends_sentence(text: str) -> bool:
    return bool(SENT_END_RE.search(str(text or "").strip()))

def _looks_like_overflow(text: str) -> bool:
    if not text: return False
    wc = word_count(text)
    return (not ends_sentence(text)) and (wc >= OVERFLOW_MIN_WORDS)

class _StreamStitcher:
    def __init__(self, source_stem: str):
        self.source_stem = source_stem
        self.result: List[Dict[str, Any]] = []
        self.first_idx_by_page: Dict[int, int] = {}
        self.pending_by_next: Dict[int, Dict[str, Any]] = {}

    def _ensure_meta(self, ch: Dict[str, Any], page_no: int):
        meta = ch.setdefault("meta", {})
        meta.setdefault("source", self.source_stem)
        meta.setdefault("page", page_no)

    def _merge_text(self, tail_chunk: Dict[str, Any], head_chunk: Dict[str, Any], head_page: int):
        ta = (tail_chunk.get("text") or "").rstrip()
        hb = _strip_prefix_head(head_chunk.get("text") or "")
        head_chunk["text"] = (ta + " " + hb) if (ta and hb and not ta.endswith((".", "!", "?", "…")) and hb[0] not in ",.;:!?)]}") else (ta + hb)
        head_chunk["type"] = "text"
        head_chunk["meta"] = {"source": self.source_stem, "page": head_page}

    def _attach_context(self, head_chunk: Dict[str, Any], pending_info: Dict[str, Any], head_page: int):
        meta = head_chunk.setdefault("meta", {})
        meta.setdefault("source", self.source_stem)
        meta.setdefault("page", head_page)

    def add_page(self, page_no: int, page_chunks: List[Dict[str, Any]], page_title: Optional[str], page_subtitle: Optional[str]):
        for ch in page_chunks:
            self._ensure_meta(ch, page_no)
        if page_no in self.pending_by_next:
            pending = self.pending_by_next.pop(page_no)
            if page_chunks and page_chunks[0].get("type") == "text":
                self._merge_text(pending["chunk"], page_chunks[0], head_page=page_no)
            elif page_chunks:
                self._attach_context(page_chunks[0], pending, head_page=page_no)
            else:
                page_chunks = [{
                    "id": f"{self.source_stem}_patch_{page_no:05d}",
                    "type": "meta",
                    "text": "",
                    "meta": {"source": self.source_stem, "page": page_no},
                }]
            if page_no in self.first_idx_by_page:
                idx = self.first_idx_by_page[page_no]
                if page_chunks:
                    self.result[idx] = page_chunks[0]
                    if len(page_chunks) > 1:
                        self.result[idx + 1 : idx + 1] = page_chunks[1:]
                return
        if page_chunks:
            self.first_idx_by_page[page_no] = len(self.result)
            self.result.extend(page_chunks)
        if page_chunks:
            last = page_chunks[-1]
            if last.get("type") == "text" and _looks_like_overflow(last.get("text","")):
                self.pending_by_next[page_no + 1] = {
                    "page": page_no,
                    "title": page_title,
                    "subtitle": page_subtitle,
                    "chunk_type": "text",
                    "chunk": {"id": last.get("id"),"type": "text","text": last.get("text",""),"meta": {"source": self.source_stem, "page": page_no}},
                }

    def finalize(self) -> List[Dict[str, Any]]:
        self.pending_by_next.clear()
        return self.result

# =========================================================
#          Empty page detector
# =========================================================
def is_page_completely_empty(fitz_page) -> bool:
    try:
        txt_empty = (fitz_page.get_text("text") or "").strip() == ""
    except Exception:
        txt_empty = True
    try:
        has_images = bool(fitz_page.get_images(full=True))
    except Exception:
        has_images = False
    try:
        has_drawings = bool(fitz_page.get_drawings())
    except Exception:
        has_drawings = False
    try:
        has_annots = bool(fitz_page.first_annot)
    except Exception:
        has_annots = False
    return txt_empty and (not has_images) and (not has_drawings) and (not has_annots)

# =========================================================
#                Persistent resume / checkpoint
# =========================================================
TMP_ROOT = Path("/tmp/.pdf_pipeline")
TMP_ROOT.mkdir(parents=True, exist_ok=True)

def _tmp_dir_for_pdf(pdf_stem: str, provider_key: str) -> Path:
    d = TMP_ROOT / pdf_stem / provider_key
    d.mkdir(parents=True, exist_ok=True)
    return d

def _tmp_manifest_path(pdf_stem: str, provider_key: str) -> Path:
    return _tmp_dir_for_pdf(pdf_stem, provider_key) / "manifest.json"

def _tmp_page_file(pdf_stem: str, provider_key: str, page_num: int) -> Path:
    return _tmp_dir_for_pdf(pdf_stem, provider_key) / f"{provider_key}_{page_num}.json"

def _atomic_write_json(path: Path, obj) -> None:
    tmp = path.with_suffix(path.suffix + ".part")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    tmp.replace(path)

def _read_json_if_exists(path: Path):
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    return None

def _update_manifest(pdf_stem: str, provider_key: str, total_pages: int, completed_page: int = None):
    mpath = _tmp_manifest_path(pdf_stem, provider_key)
    man = _read_json_if_exists(mpath) or {"total_pages": total_pages, "done": []}
    man["total_pages"] = total_pages
    if completed_page is not None and completed_page not in man["done"]:
        man["done"].append(completed_page); man["done"].sort()
    _atomic_write_json(mpath, man)
    return man

def _completed_pages(pdf_stem: str, provider_key: str):
    m = _read_json_if_exists(_tmp_manifest_path(pdf_stem, provider_key)) or {}
    done = set(m.get("done") or [])
    dirp = _tmp_dir_for_pdf(pdf_stem, provider_key)
    for p in dirp.glob(f"{provider_key}_*.json"):
        try:
            num = int(p.stem.split("_")[-1]); done.add(num)
        except Exception:
            pass
    return done

def _prerender_all_pages(pdf_path: Path) -> Dict[int, Dict[str, Any]]:
    out: Dict[int, Dict[str, Any]] = {}
    pdf_stem = pdf_path.stem
    if SAVE_SENT_IMAGES:
        (SENT_IMG_DIR / pdf_stem).mkdir(parents=True, exist_ok=True)

    with fitz.open(str(pdf_path)) as doc:
        total = len(doc)
        for i in range(total):
            pno = i + 1
            page = doc.load_page(i)
            if is_page_completely_empty(page):
                out[pno] = {"bytes": b"", "mime": "image/png", "w": 0, "h": 0, "tokens": 0, "sent_path": None}
                continue
            img_bytes, mime, w, h, est_tokens = render_page_image_bytes(
                page,
                target_max_bytes=(min(IMAGE_MAX_BYTES, _bedrock_raw_image_budget())
                                  if "claude" in _llm_provider_label().lower() or "bedrock" in _llm_provider_label().lower()
                                  else IMAGE_MAX_BYTES),
                target_width_px=IMAGE_TARGET_WIDTH_PX,
                target_height_px=IMAGE_TARGET_HEIGHT_PX,
                force_aspect=IMAGE_FORCE_ASPECT,
            )
            sent_path = None
            if PDF_DUMP_IMAGES and img_bytes:
                SENT_IMG_DIR.mkdir(parents=True, exist_ok=True)
                (SENT_IMG_DIR / pdf_stem).mkdir(parents=True, exist_ok=True)
                ext = ".png" if mime == "image/png" else ".jpg"
                sent_path = str((SENT_IMG_DIR / pdf_stem / f"sent_page_{pno:04d}{ext}").resolve())
                with open(sent_path, "wb") as f:
                    f.write(img_bytes)
            out[pno] = {"bytes": img_bytes, "mime": mime, "w": w, "h": h, "tokens": est_tokens, "sent_path": sent_path}
    return out

def _llm_worker_from_bytes(pdf_path: Path, pdf_stem: str, pno: int, pre: Dict[str, Any]) -> Dict[str, Any]:
    img_bytes = pre.get("bytes") or b""
    mime = pre.get("mime") or "image/png"

    # 1) LLM: tables & charts only (this returns tables/charts JSON)
    page_obj = extract_info_from_image(img_bytes, pno, fitz_page=None, mime_type=mime, tables_only=SEND_TABLES_ONLY)

    # Defensive ensure dict
    if not isinstance(page_obj, dict):
        page_obj = {
            "page_number": pno,
            "title": None,
            "subtitle": None,
            "narrative_text": None,
            "tables": [],
            "charts": [],
            "notes": None,
            "error": "no_page_obj",
            "debug": {},
        }

    # Normalize structured lists
    page_obj.setdefault("tables", [])
    page_obj.setdefault("charts", [])

    # 2) LOCAL: always extract textual structure locally (no LLM)
    local_struct = {}
    try:
        local_struct = _local_extract_text_structured(pdf_path, pno) or {}
    except Exception as e:
        logger.warning("Local text extraction failed for %s page %d: %s", pdf_path, pno, e)
        local_struct = {}

    # Merge local structured text into page_obj, but first FILTER out anything that duplicates LLM tables/charts
    try:
        # Always strip any narrative text returned by LLM (we only want local text for narrative)
        page_obj["narrative_text"] = None

        # Prefer title/subtitle from local if present, otherwise keep existing LLM title/subtitle (if any)
        if local_struct.get("title"):
            page_obj["title"] = local_struct.get("title")
        else:
            page_obj["title"] = page_obj.get("title") or None

        if local_struct.get("subtitle"):
            page_obj["subtitle"] = local_struct.get("subtitle")
        else:
            page_obj["subtitle"] = page_obj.get("subtitle") or None

        # Run filtering to remove local text that duplicates LLM table/chart content
        try:
            filtered_local = _filter_local_struct_by_llm(page_obj, local_struct)
        except Exception as e:
            logger.warning("Filtering local text by LLM outputs failed for %s page %d: %s", pdf_path, pno, e)
            filtered_local = local_struct or {}

        # Attach filtered sections under "sections"
        page_obj["sections"] = filtered_local.get("sections") or []

        # Debug: counts and provenance
        # prefer raw_lines count from filtered_local (but also include original if present)
        raw_count = None
        if isinstance(filtered_local.get("raw_lines"), list):
            raw_count = len(filtered_local.get("raw_lines"))
        elif isinstance(local_struct.get("raw_lines"), list):
            raw_count = len(local_struct.get("raw_lines"))
        if raw_count is not None:
            page_obj.setdefault("debug", {})["local_raw_lines_count"] = raw_count

        # include filtering debug counts if available
        if isinstance(filtered_local.get("debug"), dict):
            page_obj.setdefault("debug", {}).update({"local_filter_debug": filtered_local.get("debug")})

        page_obj.setdefault("debug", {})["local_text_extracted"] = bool(page_obj["sections"])
        page_obj.setdefault("debug", {})["tables_from_llm"] = len(page_obj.get("tables") or [])
        page_obj.setdefault("debug", {})["charts_from_llm"] = len(page_obj.get("charts") or [])

    except Exception as e:
        logger.warning("Merging local text into LLM page_obj failed for %s page %d: %s", pdf_path, pno, e)

    # 3) Persist raw JSON for inspection (unchanged)
    raw_dir = Path(RAW_JSON_DIR) / pdf_stem
    raw_dir.mkdir(parents=True, exist_ok=True)
    try:
        with open(raw_dir / f"page_{pno:04d}.json", "w", encoding="utf-8") as f:
            json.dump(page_obj, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.warning("Failed to write raw JSON for %s page %d: %s", pdf_stem, pno, e)

    return {
        "page_no": pno,
        "page_obj": page_obj,
        "sent_image_path": pre.get("sent_path"),
        "sent_image_mime": pre.get("mime"),
    }




# =========================================================
#                Streamed one-PDF processing
# =========================================================
def process_one_pdf(pdf_path: Path, output_dir: Path) -> Path:
    prerender = _prerender_all_pages(pdf_path)
    total = len(prerender)   

    prompt_tok = _approx_tokens_from_text(BASE_PROMPT)
    img_toks = [max(1, int(pre.get("tokens") or 0)) for pre in prerender.values()]
    avg_img_tok = (sum(img_toks) / max(1, len(img_toks))) if img_toks else 0
    avg_in_tok = max(800, int(prompt_tok + avg_img_tok))

    ASSUMED_LATENCY_SEC = float(os.getenv("ASSUMED_LATENCY_SEC", "5"))
    tpm_calls_per_min = max(1, int(SAFE_TPM // max(1, avg_in_tok)))
    rpm_calls_per_min = SAFE_RPM
    c_from_tpm = int((tpm_calls_per_min * ASSUMED_LATENCY_SEC) // 60)
    c_from_rpm = int((rpm_calls_per_min * ASSUMED_LATENCY_SEC) // 60)
    auto_conc = max(2, min(c_from_tpm, c_from_rpm, 32))

    env_conc = int(os.getenv("CONCURRENCY_PAGES", "0") or "0")
    concurrency_pages = env_conc if env_conc > 0 else auto_conc

    logger.info(
        "Auto concurrency → avg_in_tok=~%d, SAFE_TPM=%d, SAFE_RPM=%d, latency=%.1fs → CONCURRENCY_PAGES=%d",
        avg_in_tok, SAFE_TPM, SAFE_RPM, ASSUMED_LATENCY_SEC, concurrency_pages
    )

    t0 = time.time()
    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / f"{pdf_path.stem}.json"

    if out_file.exists() and not OVERWRITE_OUTPUT:
        logger.info("Skipping %s (found existing output: %s)", pdf_path.name, out_file)
        return out_file
    if pdf_path.suffix.lower() != ".pdf":
        logger.info("Skipping (not .pdf): %s", pdf_path.name)
        return out_file

    try:
        with fitz.open(str(pdf_path)) as doc_probe:
            pdf_meta = getattr(doc_probe, "metadata", {}) or {}
            pdf_meta_title = pdf_meta.get("title")
            _ = _pdf_meta_year(doc_probe)
    except Exception as e:
        logger.info("Skipping (invalid/corrupted PDF): %s (%s)", pdf_path.name, e)
        return out_file

    logger.info("Processing %s | Provider=%s", pdf_path.name, _llm_provider_label())
    provider_key = PROVIDER_KEY
    pdf_stem = pdf_path.stem

    tmp_dir = _tmp_dir_for_pdf(pdf_stem, provider_key)
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir, ignore_errors=True)
    tmp_dir = _tmp_dir_for_pdf(pdf_stem, provider_key)

    if SAVE_SENT_IMAGES:
        (SENT_IMG_DIR / pdf_stem).mkdir(parents=True, exist_ok=True)

    agg_pdf = None
    agg_pages = 0
    if SAVE_SENT_PDF:
        try:
            SENT_PDF_DIR.mkdir(parents=True, exist_ok=True)
            agg_pdf = fitz.open()
        except Exception as e:
            logger.warning("Could not init sent-pages PDF aggregator (%s)", e)
            agg_pdf = None

    _update_manifest(pdf_stem, provider_key, total_pages=total)
    logger.info("Resume pages already done (provider=%s): %s", provider_key, "[]")

    results: Dict[int, Dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=max(1, concurrency_pages)) as ex:
        futs = {
            ex.submit(_llm_worker_from_bytes, pdf_path, pdf_stem, pno, prerender[pno]): pno
            for pno in range(1, total + 1)
        }
        for fut in as_completed(futs):
            pno = futs[fut]
            try:
                res = fut.result()
                results[pno] = res
            except Exception as e:
                logger.error("Worker failed for page %d: %s", pno, e)
                results[pno] = {
                    "page_no": pno,
                    "page_obj": {"empty": True, "page_number": pno, "debug": {}},
                    "sent_image_path": None,
                    "sent_image_mime": None,
                }

    published_date_dt: Optional[_dt.date] = _date_from_filename(pdf_stem)
    published_date_source: Optional[str] = "filename" if published_date_dt else None
    published_date_precision: Optional[str] = "day" if published_date_dt else None

    first_page_obj = (results.get(1) or {}).get("page_obj")
    if first_page_obj:
        date_only, src, prec = resolve_doc_date_once(
            pdf_path=pdf_path, page_obj=first_page_obj, pdf_meta_title=pdf_meta_title
        )
        def _strength(s, p):
            base = {"footer": 5, "title/subtitle": 4, "page": 4, "folder": 3, "pdf_meta_title": 2, "filename": 1, None: 0}.get(s, 0)
            return base + (1 if p == "day" else 0)
        if isinstance(date_only, _dt.date):
            if _strength(src, prec) >= _strength(published_date_source, published_date_precision):
                published_date_dt = date_only
                published_date_source = src
                published_date_precision = prec

    published_date_str = (_normalize_spoken_date(published_date_dt)
                          if (published_date_dt and published_date_precision == "day")
                          else None)

    if published_date_dt:
        logger.info(
            "Published date resolution → source=%s, precision=%s, value=%s",
            published_date_source or "None",
            published_date_precision or "None",
            (published_date_str or (published_date_dt.strftime('%B %Y') if (published_date_precision=='month') else "None"))
        )

    stitcher = _StreamStitcher(source_stem=pdf_stem)
    sticky_title: Optional[str] = None
    sticky_subtitle: Optional[str] = None

    for pno in range(1, total + 1):
        r = results.get(pno) or {}
        page_obj = r.get("page_obj") or {"empty": True, "page_number": pno}

        if isinstance(page_obj, dict):
            if isinstance(page_obj.get("title"), str) and page_obj["title"].strip():
                sticky_title = page_obj["title"].strip()
            if isinstance(page_obj.get("subtitle"), str) and page_obj["subtitle"].strip():
                sticky_subtitle = page_obj["subtitle"].strip()

        page_title, page_subtitle = _page_titles_from_obj(page_obj)
        if not page_title and sticky_title: page_title = sticky_title
        if not page_subtitle and sticky_subtitle: page_subtitle = sticky_subtitle

        base_meta = {"source": pdf_stem}
        if page_title: base_meta["page_title"] = page_title
        if page_subtitle: base_meta["page_subtitle"] = page_subtitle

        parsed_page_doc = {"pages": [page_obj]}
        blocks = list(iter_extract_collect(parsed_page_doc, meta=base_meta, page=pno))

        if published_date_precision == "day" and published_date_dt and published_date_str:
            for b in blocks:
                b.setdefault("meta", {})["published_date"] = published_date_str
                if published_date_source: b["meta"]["published_date_source"] = published_date_source
        elif published_date_precision == "month" and published_date_dt:
            month_label = published_date_dt.strftime("%B %Y")
            for b in blocks:
                b.setdefault("meta", {})["published_month"] = month_label
                if published_date_source: b["meta"]["published_date_source"] = published_date_source

        page_chunks = finalize_blocks_to_chunks(blocks, source_stem=pdf_stem)

        if published_date_precision == "day" and published_date_dt and published_date_str:
            for ch in page_chunks:
                ch.setdefault("meta", {})["published_date"] = published_date_str
                if published_date_source: ch["meta"]["published_date_source"] = published_date_source
        elif published_date_precision == "month" and published_date_dt:
            month_label = published_date_dt.strftime("%B %Y")
            for ch in page_chunks:
                ch.setdefault("meta", {})["published_month"] = month_label
                if published_date_source: ch["meta"]["published_date_source"] = published_date_source

        for ch in page_chunks:
            m = ch.setdefault("meta", {})
            m["source"] = pdf_stem
            m["page"] = pno
            m.pop("page_range", None)

        if PAGE_CHUNK_CAP and len(page_chunks) > PAGE_CHUNK_CAP:
            logger.warning("  page %d: capping chunks %d → %d", pno, len(page_chunks), PAGE_CHUNK_CAP)
            page_chunks = enforce_per_page_cap(page_chunks, PAGE_CHUNK_CAP, pdf_stem, pno)

        if not page_chunks:
            page_chunks = [{
                "id": f"{pdf_stem}_error_{pno:05d}",
                "type": "empty",
                "text": "",
                "meta": {"source": pdf_stem, "page": pno},
            }]

        page_tmp = _tmp_page_file(pdf_stem, provider_key, pno)
        page_tmp.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write_json(page_tmp, page_chunks)
        _update_manifest(pdf_stem, provider_key, total_pages=total, completed_page=pno)
        stitcher.add_page(pno, page_chunks, page_title, page_subtitle)
        logger.info("  page %d: saved %d chunk(s) to %s", pno, len(page_chunks), page_tmp)

        try:
            if SAVE_SENT_PDF and agg_pdf is not None:
                sent_path = r.get("sent_image_path")
                if sent_path:
                    with open(sent_path, "rb") as fh:
                        img_bytes = fh.read()
                    filetype = "png" if sent_path.lower().endswith(".png") else "jpeg"
                    with fitz.open(stream=img_bytes, filetype=filetype) as imdoc:
                        pix = imdoc[0].get_pixmap()
                        w, h = pix.width, pix.height
                    newp = agg_pdf.new_page(width=w, height=h)
                    newp.insert_image(newp.rect, stream=img_bytes)
                    agg_pages += 1
        except Exception as e:
            logger.warning("  page %d: could not append to sent PDF (%s)", pno, e)

    chunks: List[Dict[str, Any]] = stitcher.finalize()
    for ch in chunks:
        m = ch.setdefault("meta", {})
        m["source"] = pdf_stem
        if isinstance(m.get("page"), list):
            rng = m["page"]
            try:
                m["page"] = rng[1] if len(rng) >= 2 else rng[0]
            except Exception:
                m["page"] = None

    out_file.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(out_file, chunks)
    logger.info("→ wrote %s (%d chunks) in %.1fs", out_file.name, len(chunks), time.time() - t0)

    try:
        if SAVE_SENT_PDF and agg_pdf is not None:
            if agg_pages > 0:
                out_sent = SENT_PDF_DIR / f"{pdf_path.stem}.sent.pdf"
                agg_pdf.save(str(out_sent))
                logger.info("Saved sent-pages PDF: %s", out_sent)
            else:
                logger.warning("Skipping sent-pages PDF save: no pages were added.")
            agg_pdf.close()
    except Exception as e:
        logger.warning("Could not save/close sent-pages PDF (%s)", e)

    try:
        shutil.rmtree(_tmp_dir_for_pdf(pdf_stem, provider_key))
    except Exception:
        pass

    return out_file

# =========================================================
#                   Batch all PDFs (single flow)
# =========================================================
def run_all(input_dir: str, output_dir: str):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    all_files = sorted([p for p in input_dir.rglob("*") if p.is_file()])
    pdfs = []
    skipped = 0
    for p in all_files:
        if p.suffix.lower() == ".pdf":
            pdfs.append(p)
        else:
            skipped += 1
            try: rel = p.relative_to(input_dir)
            except Exception: rel = p
            logger.info("Skipping (not .pdf): %s", rel)

    if not pdfs:
        logger.info("Found 0 PDF(s) in %s | LLM: %s", input_dir, _llm_provider_label())
        if skipped:
            logger.info("Also skipped %d non-PDF file(s).", skipped)
        return

    logger.info("Found %d PDF(s) in %s | LLM: %s", len(pdfs), input_dir, _llm_provider_label())
    if skipped:
        logger.info("Also skipped %d non-PDF file(s) because extension != .pdf", skipped)

    for pdf in pdfs:
        out_path = output_dir / f"{pdf.stem}.json"
        if out_path.exists() and not OVERWRITE_OUTPUT:
            logger.info("Skipping %s (already processed → %s)", pdf.name, out_path)
            continue
        process_one_pdf(pdf, output_dir)

# =========================================================
#                         main
# =========================================================
if __name__ == "__main__":
    providers_env = (os.getenv("PDF_LLM_PROVIDER") or "gemini").lower().strip()
    if providers_env in ("both", "all"):
        providers = ["gemini", "bedrock"]
    else:
        providers = [p.strip() for p in providers_env.split(",") if p.strip()] or ["gemini"]

    for p in providers:
        select_provider(p)
        provider_out_dir = Path(OUTPUT_DIR) / PROVIDER_KEY.lower()
        provider_out_dir.mkdir(parents=True, exist_ok=True)

        print("\n" + "="*70)
        print(f"▶ Running with provider: {PROVIDER.label()} → {provider_out_dir}")
        print("="*70)

        TOK = TokenAccumulator()

        t0_main = time.time()
        logger.info("LLM provider: %s", _llm_provider_label())

        run_all(PDF_INPUT_DIR, provider_out_dir)

        tok_summary = TOK.summary()
        elapsed_time = time.time() - t0_main
        input_total  = int(tok_summary.get("input",  {}).get("total",  0) or 0)
        output_total = int(tok_summary.get("output", {}).get("total", 0) or 0)
        grand_total  = input_total + output_total

        in_price, out_price = _price_for_current_provider()
        input_cost   = (input_total  / 1000.0) * in_price
        output_cost  = (output_total / 1000.0) * out_price
        total_cost   = input_cost + output_cost

        print("\n" + "="*50)
        print(f"✅ Complete! Results in {provider_out_dir.resolve()}/")
        print(f"\n⏱️  TIME: {elapsed_time:.1f} seconds")
        print("\n📊 TOKEN USAGE:")
        print(f"   Input Tokens:  {input_total:,}")
        print(f"   Output Tokens: {output_total:,}")
        print(f"   Total Tokens:  {grand_total:,}")
        print("\n💰 COST ESTIMATE  —  " + PROVIDER.label())
        print(f"   Input Cost:    ${input_cost:.4f}")
        print(f"   Output Cost:   ${output_cost:.4f}")
        print(f"   ⭐ TOTAL COST: ${total_cost:.4f}")
        print("="*50)
