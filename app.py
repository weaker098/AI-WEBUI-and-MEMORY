import time
import re
import os
import traceback
import uuid
from ddgs import DDGS
import tiktoken
import threading
from urllib.parse import urlparse
os.environ['HF_HUB_OFFLINE'] = '0'
from sentence_transformers import SentenceTransformer, util, CrossEncoder  # NEW: Added CrossEncoder
import json
import requests
from bs4 import BeautifulSoup
import shutil
import base64
from flask import Flask, render_template, request, jsonify, Response, send_from_directory
from datetime import datetime

# --- NEW FAISS IMPORTS ---
import faiss
import numpy as np
import gc

# --- END NEW FAISS IMPORTS ---

# FIX #5: Retry wrapper for faiss.read_index to handle mid-write corruption
def _safe_faiss_read(path, retries=3, delay=0.15):
    """
    Reads a FAISS index with retry logic.
    If a concurrent write is happening (file partially flushed), the read can throw.
    Retrying after a short sleep almost always succeeds.
    """
    last_exc = None
    for attempt in range(retries):
        try:
            return faiss.read_index(path)
        except Exception as exc:
            last_exc = exc
            if attempt < retries - 1:
                print(f"DEBUG (FAISS read retry {attempt+1}/{retries}): {exc}")
                time.sleep(delay)
    raise last_exc

# --- ZERO-SHOT SMART FILTER ---


# NOTE: DDGS is instantiated per-call inside duckduckgo_search() using a context
# manager — no global instance needed. The old global ddgs was dead code (BUG 17 fix).
app = Flask(__name__)

# --- Cache-Control: prevent browser from serving stale settings on GET endpoints ---
# Without this, Flask returns no cache directives and browsers freely cache /get_*
# responses. Opening a modal can then return a stale snapshot instead of live server
# state, making it look like a save didn't stick until you hard-refresh.
# no-store is the strongest directive: never write to cache, always hit the server.
@app.after_request
def disable_caching_on_get_endpoints(response):
    if request.method == "GET" and request.path.startswith("/get_"):
        response.headers["Cache-Control"] = "no-store"
    return response
# --------------------------------------------------------------------------

# --- FIX #1: Thread lock for memory file I/O to prevent race conditions ---
# RLock (reentrant) instead of Lock — delete_message and edit_message each
# acquire _memory_lock twice (once for read, once for write) within the same
# thread. threading.Lock() would deadlock on the second acquire; RLock allows
# the same thread to re-enter safely.
_memory_lock = threading.RLock()
# --------------------------------------------------------------------------

# --- FIX: Thread lock for FAISS index file writes to prevent corruption ---
_faiss_write_lock = threading.Lock()
# --------------------------------------------------------------------------

# --- Global rebuild status tracking ---
_rebuild_running = False
_rebuild_status_lock = threading.Lock()
# -------------------------------------

# Configuration for saving images
UPLOAD_FOLDER = 'saved_images'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# --- Persona & User Loadout Image Storage ---
PERSONA_IMAGES_DIR = 'persona_images'
USER_LOADOUT_IMAGES_DIR = 'user_loadout_images'
os.makedirs(PERSONA_IMAGES_DIR, exist_ok=True)
os.makedirs(USER_LOADOUT_IMAGES_DIR, exist_ok=True)

# --- User Loadout Constants ---
USER_LOADOUTS_FILE = 'user_loadouts.json'
CURRENT_USER_LOADOUT_FILE = 'current_user_loadout.txt'
USER_LOADOUTS = {}  # Global cache

# --- NEW: Dynamic Model Loading ---
EMBEDDING_MODEL = None
EMBEDDING_DIM = 768
EMBEDDING_CTX_LENGTH = None  # None = model default; set by user slider
HIGH_CTX_MODELS = {"nomic", "jina-small", "jina-base"}  # models that support 8K ctx

# --- NEW: Reranker Model Loading ---
RERANKER_MODEL = None
RERANKER_CTX_LENGTH = None
HIGH_CTX_RERANKERS = {
    "bge-reranker-v2-m3": 8192,
    "qwen3-reranker-0.6b": 32768,
    "qwen3-reranker-4b": 32768,
    "jina-reranker-v2": 32768
}
# --- END NEW RERANKER GLOBALS ---

# --- STAGE 3: SANITY CHECK GLOBALS ---
SANITY_CHECK_SETTINGS = {
    "sanity_check_enabled": False,
    "sanity_check_intent_enabled": True,
    "sanity_check_intent_threshold": 0.40,
    "sanity_check_content_threshold": 0.35,
    "sanity_check_recall_phrases": "do you remember, what did we talk about, recall our conversation, tell me about, what did I say, when did we discuss, yesterday we, last time, that time when",
    "sanity_check_buffer_multiplier": 2.0,
    "sanity_check_show_scores": True,
    "sanity_check_negative_phrases": "good morning, good evening, good night, good afternoon, hello there, hey there, hi there, how are you, what's up, how's it going, hey how are you, good to see you, nice to meet you, how have you been",
    # --- ZEROSHOT DLC ---
    "zeroshot_intent_enabled": False,
    "zeroshot_model": "nli-minilm-l6",
    "zeroshot_ctx_length": 512,
    "zeroshot_threshold_enabled": True,        # ON = gate by threshold, OFF = score only (never blocks)
    "zeroshot_entailment_threshold": 0.70,
    "zeroshot_aggregation": "max",             # "max" = any hypothesis fires → pass, "avg" = consensus
    "zeroshot_premise_example": "do you remember when we talked about pizza?",
    "zeroshot_hypotheses": [                   # List — all tested, aggregated by mode above
        "the user wants to recall a past conversation",
        "the user is asking about something that was discussed before",
        "the user wants to remember something from a previous chat",
    ],
    # --- ZEROSHOT ADVANCED SETTINGS ---
    # score_floor: hypotheses scoring below this are excluded from aggregation entirely.
    # Prevents one weak hypothesis from tanking the avg. 0.0 = off. Recommended: 0.10
    "zeroshot_score_floor": 0.10,
    # top_n_hypotheses: only aggregate the top N highest-scoring hypotheses.
    # 0 = use all hypotheses. Useful when you have a mix of strong/weak ones.
    "zeroshot_top_n_hypotheses": 0,
    # per-model threshold presets — read-only reference, used by UI "Load Preset" button.
    "zeroshot_model_presets": {
        "deberta-base":    {"threshold": 0.80, "aggregation": "avg", "score_floor": 0.10},
        "deberta-xsmall":  {"threshold": 0.65, "aggregation": "max", "score_floor": 0.10},
        "nli-minilm-l6":   {"threshold": 0.75, "aggregation": "max", "score_floor": 0.10},
        "distilbert-mnli": {"threshold": 0.50, "aggregation": "max", "score_floor": 0.05},
        "bart-large-mnli": {"threshold": 0.80, "aggregation": "avg", "score_floor": 0.10},
    },
    # --- END ZEROSHOT ADVANCED SETTINGS ---
}
RECALL_INTENT_EMBEDDINGS = None  # Pre-computed embeddings of recall phrases
NEGATIVE_INTENT_EMBEDDINGS = None  # Pre-computed embeddings of negative/greeting phrases
SANITY_CHECK_SETTINGS_FILE = "sanity_check_settings.json"
# --- END STAGE 3 GLOBALS ---

# --- ZEROSHOT DLC GLOBALS ---
ZEROSHOT_MODEL = None          # CrossEncoder NLI model, loaded only when DLC is ON
ZEROSHOT_CTX_LENGTH = None     # Stored max_length of the loaded zeroshot model
ZEROSHOT_ENTAILMENT_IDX = 1    # Read from model's id2label on load — default 1
ZEROSHOT_MODEL_MAP = {
    # All NLI models: label order [contradiction, entailment, neutral] or [not_entailment, entailment]
    # Entailment is always index 1 after softmax
    "nli-minilm-l6":   "cross-encoder/nli-MiniLM2-L6-H768",      # 90MB  ⚡ fastest
    "distilbert-mnli": "typeform/distilbert-base-uncased-mnli",    # 65MB  🥔 potato (weak quality)
    "bart-large-mnli": "facebook/bart-large-mnli",                 # 400MB 💀 overkill
    "deberta-xsmall":  "MoritzLaurer/deberta-v3-xsmall-zeroshot-v1.1-all-33",  # 142MB 🧠 small
    "deberta-base":    "MoritzLaurer/deberta-v3-base-zeroshot-v1.1-all-33",     # ~280MB 🧠 better
}
# --- END ZEROSHOT DLC GLOBALS ---

# --- FAISS Index Factory ---
def create_faiss_index(dim):
    """Creates a Faiss index using the metric chosen in settings (L2 or Cosine/IP)."""
    metric = RESONANCE_SETTINGS.get("faiss_distance_metric", "l2")
    if metric == "cosine":
        return faiss.IndexFlatIP(dim)  # Inner Product → cosine after normalization
    return faiss.IndexFlatL2(dim)

def prepare_vectors(vectors_np):
    """Normalizes vectors for cosine similarity if that metric is active."""
    metric = RESONANCE_SETTINGS.get("faiss_distance_metric", "l2")
    if metric == "cosine":
        faiss.normalize_L2(vectors_np)
    return vectors_np
# --- END FAISS Index Factory ---

def initialize_embedding_model(model_choice="nomic", ctx_length=None):
    global EMBEDDING_MODEL, EMBEDDING_DIM, EMBEDDING_CTX_LENGTH
    # _KEYBERT_MODEL auto-detects model changes via identity check — no manual reset needed

    # --- Explicit cleanup before loading new model ---
    if EMBEDDING_MODEL is not None:
        print(f"DEBUG: Releasing old embedding model from memory...")
        del EMBEDDING_MODEL
        EMBEDDING_MODEL = None
        gc.collect()
    # --------------------------------------------------
    if model_choice == "all-mini":
        print("DEBUG: Loading lightweight All-MiniLM-L6-v2 (384d)...")
        EMBEDDING_DIM = 384
        EMBEDDING_MODEL = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
    elif model_choice == "all-mpnet":
        print("DEBUG: Loading balanced All-MPNet-Base-v2 (768d)...")
        EMBEDDING_DIM = 768
        EMBEDDING_MODEL = SentenceTransformer('all-mpnet-base-v2', device='cpu')
    elif model_choice == "jina-small":
        print("DEBUG: Loading Jina Small v2 (512d, 8K ctx)...")
        EMBEDDING_DIM = 512
        EMBEDDING_MODEL = SentenceTransformer('jinaai/jina-embeddings-v2-small-en', device='cpu')
    elif model_choice == "jina-base":
        print("DEBUG: Loading Jina Base v2 (768d, 8K ctx)...")
        EMBEDDING_DIM = 768
        EMBEDDING_MODEL = SentenceTransformer('jinaai/jina-embeddings-v2-base-en', device='cpu')
    else:
        print("DEBUG: Loading heavy Nomic AI model (768d)...")
        EMBEDDING_DIM = 768
        EMBEDDING_MODEL = SentenceTransformer('nomic-ai/nomic-embed-text-v1.5', trust_remote_code=True, device='cpu')
        
    # Apply custom context length for high-ctx models only
    if model_choice in HIGH_CTX_MODELS and ctx_length is not None:
        EMBEDDING_MODEL.max_seq_length = ctx_length
        EMBEDDING_CTX_LENGTH = ctx_length
        print(f"DEBUG: Context length overridden to {ctx_length} tokens.")
    else:
        EMBEDDING_CTX_LENGTH = EMBEDDING_MODEL.max_seq_length  # store the actual default

    print(f"DEBUG: Sentence Transformer loaded successfully (Model: {model_choice}, Dim: {EMBEDDING_DIM}, Ctx: {EMBEDDING_CTX_LENGTH}).")
# --- END NEW MODEL LOADING ---


# =============================================================================
# ASYMMETRIC EMBEDDING HELPERS
# =============================================================================
# Some models are trained asymmetrically: the query and the stored document
# each get a different prefix/instruction.  Using the correct prefix can
# meaningfully improve retrieval precision at zero extra cost.
#
# Prefix map — add new models here as needed.
# Empty string "" means "no prefix" (symmetric models).
# =============================================================================
_EMBEDDING_PREFIX_MAP = {
    "nomic":      {"query": "search_query: ",   "document": "search_document: "},
    "jina-small": {"query": "query: ",          "document": "passage: "},
    "jina-base":  {"query": "query: ",          "document": "passage: "},
    "all-mini":   {"query": "",                 "document": ""},
    "all-mpnet":  {"query": "",                 "document": ""},
}


def _apply_prefix(texts, side: str):
    """
    Prepends the model-specific prefix to each text in `texts` if the
    'embedding_prefix_mode' setting is not 'off'.

    Args:
        texts : str | list[str]
        side  : "query" | "document"

    Returns:
        list[str] — always a list, safe to pass directly to encode().
    """
    if isinstance(texts, str):
        texts = [texts]

    mode = RESONANCE_SETTINGS.get("embedding_prefix_mode", "auto")
    if mode == "off":
        return texts

    model_key = RESONANCE_SETTINGS.get("embedding_model", "nomic")
    prefixes  = _EMBEDDING_PREFIX_MAP.get(model_key, {"query": "", "document": ""})
    prefix    = prefixes.get(side, "")

    if not prefix:
        return texts  # symmetric model — nothing to prepend

    return [prefix + t for t in texts]


def embed_query(text_or_texts, **kwargs):
    """
    Encode one or more query strings with the correct model prefix.
    Wraps EMBEDDING_MODEL.encode(); all kwargs are forwarded.
    Returns a numpy array.
    """
    texts = _apply_prefix(text_or_texts, "query")
    result = EMBEDDING_MODEL.encode(texts, **kwargs)
    return result


def embed_documents(texts, **kwargs):
    """
    Encode one or more document/memory strings with the correct model prefix.
    Wraps EMBEDDING_MODEL.encode(); all kwargs are forwarded.
    Returns a numpy array.
    """
    if isinstance(texts, str):
        texts = [texts]
    texts = _apply_prefix(texts, "document")
    result = EMBEDDING_MODEL.encode(texts, **kwargs)
    return result

# =============================================================================
# END ASYMMETRIC EMBEDDING HELPERS
# =============================================================================


# ============================================================================
# RERANKER MODEL INITIALIZATION
# ============================================================================

def initialize_reranker_model(model_choice="ms-marco-mini-v2", ctx_length=None):
    """
    Loads a cross-encoder reranker model for two-stage retrieval.
    
    Args:
        model_choice: One of the supported reranker models
        ctx_length: Optional context length override for high-ctx models
    """
    global RERANKER_MODEL, RERANKER_CTX_LENGTH

    # --- Explicit cleanup before loading new model ---
    if RERANKER_MODEL is not None:
        print(f"DEBUG (Reranker): Releasing old reranker model from memory...")
        del RERANKER_MODEL
        RERANKER_MODEL = None
        gc.collect()
    # --------------------------------------------------

    print(f"DEBUG (Reranker): Loading model '{model_choice}'...")
    
    try:
        # Model mapping
        model_map = {
            "flashrank": "flashrank",  # Special handling for FlashRank
            "ms-marco-mini-v2": "cross-encoder/ms-marco-MiniLM-L-6-v2",
            "ms-marco-mini-v12": "cross-encoder/ms-marco-MiniLM-L-12-v2",
            "bge-reranker-base": "BAAI/bge-reranker-base",
            "bge-reranker-v2-m3": "BAAI/bge-reranker-v2-m3",
            "jina-reranker-v2": "jinaai/jina-reranker-v2-base-multilingual",
            "qwen3-reranker-0.6b": "Qwen/Qwen2.5-Reranker-0.6B",
            "qwen3-reranker-4b": "Qwen/Qwen2.5-Reranker-4B"
        }
        
        # Special case: FlashRank (different library)
        if model_choice == "flashrank":
            try:
                from flashrank import Ranker, RerankRequest
                RERANKER_MODEL = Ranker(model_name="ms-marco-MiniLM-L-12-v2", cache_dir="./flashrank_cache")
                RERANKER_CTX_LENGTH = 512  # FlashRank uses fixed context
                print(f"DEBUG (Reranker): FlashRank loaded successfully (512 ctx).")
                return
            except ImportError:
                print("ERROR: FlashRank not installed. Install with: pip install flashrank --break-system-packages")
                print("Falling back to ms-marco-mini-v2...")
                model_choice = "ms-marco-mini-v2"
        
        # Load standard CrossEncoder model
        model_path = model_map.get(model_choice, model_map["ms-marco-mini-v2"])
        
        # Some models need trust_remote_code (Jina, Qwen)
        needs_trust = model_choice in ["jina-reranker-v2", "qwen3-reranker-0.6b", "qwen3-reranker-4b"]
        
        if needs_trust:
            RERANKER_MODEL = CrossEncoder(model_path, max_length=512, device='cpu', trust_remote_code=True)
        else:
            RERANKER_MODEL = CrossEncoder(model_path, max_length=512, device='cpu')
        
        # Apply custom context length for high-ctx models
        if model_choice in HIGH_CTX_RERANKERS and ctx_length is not None:
            max_supported = HIGH_CTX_RERANKERS[model_choice]
            ctx_length = min(ctx_length, max_supported)  # Cap at model's max
            RERANKER_MODEL.max_length = ctx_length
            RERANKER_CTX_LENGTH = ctx_length
            print(f"DEBUG (Reranker): Context length set to {ctx_length} tokens.")
        else:
            RERANKER_CTX_LENGTH = RERANKER_MODEL.max_length
        
        print(f"DEBUG (Reranker): Model loaded successfully (Model: {model_choice}, Ctx: {RERANKER_CTX_LENGTH}).")
        
    except Exception as e:
        print(f"ERROR (Reranker): Failed to load model '{model_choice}': {e}")
        traceback.print_exc()
        RERANKER_MODEL = None
        RERANKER_CTX_LENGTH = None


# ============================================================================
# ZEROSHOT DLC — NLI-BASED INTENT DETECTION
# ============================================================================

def initialize_zeroshot_model(model_choice="nli-minilm-l6", ctx_length=512):
    global ZEROSHOT_MODEL, ZEROSHOT_CTX_LENGTH, ZEROSHOT_ENTAILMENT_IDX

    if ZEROSHOT_MODEL is not None:
        print(f"DEBUG (Zeroshot): Releasing old model...")
        del ZEROSHOT_MODEL
        ZEROSHOT_MODEL = None
        gc.collect()

    model_path = ZEROSHOT_MODEL_MAP.get(model_choice)
    if not model_path:
        print(f"ERROR (Zeroshot): Unknown model '{model_choice}'. Valid: {list(ZEROSHOT_MODEL_MAP.keys())}")
        return

    print(f"DEBUG (Zeroshot): Loading '{model_choice}' ({model_path})...")
    try:
        ZEROSHOT_MODEL = CrossEncoder(model_path, max_length=ctx_length, device='cpu')
        ZEROSHOT_CTX_LENGTH = ctx_length

        # Detect entailment index from the model's own id2label — never hardcode
        # Different models use different label orderings:
        #   nli-MiniLM:  {0:contradiction, 1:entailment, 2:neutral}
        #   deberta:     {0:not_entailment, 1:entailment}  or  {0:ENTAILMENT, 1:NEUTRAL, 2:CONTRADICTION}
        # Reading directly from config is the only safe approach
        entailment_idx = None
        try:
            id2label = ZEROSHOT_MODEL.model.config.id2label
            for idx, label in id2label.items():
                if "entail" in label.lower():
                    entailment_idx = int(idx)
                    break
        except Exception:
            pass

        if entailment_idx is None:
            entailment_idx = 1  # sane fallback
            print(f"WARNING (Zeroshot): Could not read id2label — defaulting entailment_idx=1")
        else:
            print(f"DEBUG (Zeroshot): id2label={id2label} → entailment_idx={entailment_idx}")

        ZEROSHOT_ENTAILMENT_IDX = entailment_idx
        print(f"DEBUG (Zeroshot): Loaded OK (ctx={ctx_length}, entailment_idx={entailment_idx}).")
    except Exception as e:
        print(f"ERROR (Zeroshot): Failed to load '{model_choice}': {e}")
        traceback.print_exc()
        ZEROSHOT_MODEL = None
        ZEROSHOT_CTX_LENGTH = None


def check_recall_intent_zeroshot(query):
    """
    NLI-based intent detection (Zeroshot DLC path).

    Tests the query against every hypothesis in zeroshot_hypotheses, then
    aggregates scores by zeroshot_aggregation:
        "max" → take highest score across all hypotheses (recall-friendly)
        "avg" → average across all hypotheses (precision / consensus)

    If zeroshot_threshold_enabled is ON  → gates by threshold (blocks low-intent queries)
    If zeroshot_threshold_enabled is OFF → always passes but logs the score (scorer only)

    Returns (has_intent: bool, best_score: float).
    Falls back to (True, 1.0) if model not loaded so pipeline never hard-crashes.
    """
    if ZEROSHOT_MODEL is None:
        print("WARNING (Zeroshot): Model not loaded — falling back to PASS.")
        return True, 1.0

    hypotheses = SANITY_CHECK_SETTINGS.get("zeroshot_hypotheses", [
        "the user wants to recall a past conversation"
    ])
    # Flatten in case someone saved a raw string instead of list
    if isinstance(hypotheses, str):
        hypotheses = [h.strip() for h in hypotheses.split("\n") if h.strip()]
    if not hypotheses:
        print("WARNING (Zeroshot): No hypotheses defined — falling back to PASS.")
        return True, 1.0

    threshold         = SANITY_CHECK_SETTINGS.get("zeroshot_entailment_threshold", 0.70)
    threshold_enabled = SANITY_CHECK_SETTINGS.get("zeroshot_threshold_enabled", True)
    aggregation       = SANITY_CHECK_SETTINGS.get("zeroshot_aggregation", "max")

    # Entailment index resolved at model load time from id2label — no guessing
    entailment_idx = ZEROSHOT_ENTAILMENT_IDX

    try:
        # Pairs: (premise=user_query, hypothesis=intent_to_test)
        # CrossEncoder NLI convention: text_a=premise, text_b=hypothesis
        pairs = [(query, h) for h in hypotheses]

        # convert_to_numpy=True avoids the torch.tensor(tensor) double-wrap warning
        logits_batch = ZEROSHOT_MODEL.predict(pairs, convert_to_numpy=True)

        # Numpy softmax — no torch dependency, no version issues
        def _softmax(x):
            e = np.exp(x - np.max(x))  # subtract max for numerical stability
            return e / e.sum()

        score_floor   = SANITY_CHECK_SETTINGS.get("zeroshot_score_floor", 0.10)
        top_n         = int(SANITY_CHECK_SETTINGS.get("zeroshot_top_n_hypotheses", 0))

        scores = []
        for i, logits in enumerate(logits_batch):
            probs = _softmax(logits)
            score = float(probs[entailment_idx])
            scores.append(score)
            print(f"DEBUG (Zeroshot): H{i+1} entailment={score:.3f} raw={[round(float(l),3) for l in logits]} | \"{hypotheses[i]}\"")

        # Apply score floor — drop hypotheses that scored too low to be meaningful
        active_scores = [s for s in scores if s >= score_floor]
        floored_count = len(scores) - len(active_scores)
        if floored_count > 0:
            print(f"DEBUG (Zeroshot): score_floor={score_floor} removed {floored_count} weak hypothesis(es)")
        # If floor wiped everything, fall back to all scores so we never hard-block
        if not active_scores:
            print(f"DEBUG (Zeroshot): score_floor wiped all scores — using raw scores as fallback")
            active_scores = scores

        # Apply top-N — only aggregate the N best scoring hypotheses
        if top_n > 0 and len(active_scores) > top_n:
            active_scores = sorted(active_scores, reverse=True)[:top_n]
            print(f"DEBUG (Zeroshot): top_n={top_n} — using top {top_n} scores: {[round(s,3) for s in active_scores]}")

        # Aggregate
        if aggregation == "avg":
            final_score = sum(active_scores) / len(active_scores)
        else:
            final_score = max(active_scores)

        print(f"DEBUG (Zeroshot): aggregation={aggregation} final={final_score:.3f} threshold={threshold} gating={'ON' if threshold_enabled else 'OFF'}")

        if threshold_enabled:
            has_intent = final_score >= threshold
        else:
            has_intent = True  # Scorer-only mode — never blocks

        print(f"DEBUG (Zeroshot): → {'PASS' if has_intent else 'BLOCK'}")
        return has_intent, final_score

    except Exception as e:
        print(f"ERROR (Zeroshot): Scoring failed: {e} — falling back to PASS.")
        traceback.print_exc()
        return True, 1.0

# ============================================================================
# END ZEROSHOT DLC
# ============================================================================


def rerank_candidates(query, candidates, top_k=None, score_threshold=0.0, batch_size=32):
    """
    Reranks a list of candidate texts using the loaded reranker model.
    
    Args:
        query: The search query string
        candidates: List of dicts with 'content' key (memory entries)
        top_k: Number of top results to return (None = return all above threshold)
        score_threshold: Minimum score to include in results
        batch_size: Batch size for scoring (controls RAM usage)
    
    Returns:
        List of candidates sorted by relevance score, with 'reranker_score' added
    """
    if not RERANKER_MODEL:
        print("WARNING (Reranker): Model not loaded, returning candidates unranked.")
        return candidates[:top_k] if top_k else candidates
    
    if not candidates:
        return []
    
    try:
        # Handle FlashRank separately
        if hasattr(RERANKER_MODEL, '__class__') and 'flashrank' in str(type(RERANKER_MODEL)).lower():
            from flashrank import RerankRequest
            passages = [{"text": c.get("content", "")} for c in candidates]
            rerank_request = RerankRequest(query=query, passages=passages)
            results = RERANKER_MODEL.rerank(rerank_request)
            
            # Map scores back to candidates using the result's own index field.
            # FlashRank returns results reordered by score, so enumerate(results)
            # position does NOT correspond to the original candidate position.
            for result in results:
                idx = result.get("index", -1)
                if 0 <= idx < len(candidates):
                    candidates[idx]["reranker_score"] = result["score"]
            
            # Sort by score
            scored = sorted(
                [c for c in candidates if c.get("reranker_score", 0) >= score_threshold],
                key=lambda x: x.get("reranker_score", 0),
                reverse=True
            )
            return scored[:top_k] if top_k else scored
        
        # Standard CrossEncoder scoring
        pairs = [(query, c.get("content", "")) for c in candidates]
        
        # Batch scoring for RAM efficiency
        all_scores = []
        for i in range(0, len(pairs), batch_size):
            batch = pairs[i:i + batch_size]
            scores = RERANKER_MODEL.predict(batch, convert_to_numpy=True, show_progress_bar=False)
            all_scores.extend(scores)
        
        # Attach scores to candidates
        for candidate, score in zip(candidates, all_scores):
            candidate["reranker_score"] = float(score)
        
        # Filter by threshold and sort
        scored = [c for c in candidates if c.get("reranker_score", 0) >= score_threshold]
        scored.sort(key=lambda x: x.get("reranker_score", 0), reverse=True)
        
        # Return top-k
        return scored[:top_k] if top_k else scored
        
    except Exception as e:
        print(f"ERROR (Reranker): Scoring failed: {e}")
        traceback.print_exc()
        return candidates[:top_k] if top_k else candidates


# ============================================================================
# END RERANKER FUNCTIONS
# ============================================================================


# ============================================================================
# STAGE 3: SANITY CHECK FUNCTIONS
# ============================================================================

def initialize_recall_intent_embeddings():
    """
    Pre-computes embeddings for recall intent phrases and negative/greeting phrases
    using active embedding model. Called after embedding model is loaded and when
    recall phrases are updated.
    """
    global RECALL_INTENT_EMBEDDINGS, NEGATIVE_INTENT_EMBEDDINGS
    
    if not EMBEDDING_MODEL:
        print("WARNING (Sanity): Embedding model not loaded - skipping intent embeddings")
        return
    
    # --- Positive recall phrases ---
    phrases_str = SANITY_CHECK_SETTINGS.get("sanity_check_recall_phrases", "")
    if not phrases_str:
        print("WARNING (Sanity): No recall phrases defined")
        return
    
    phrases = [p.strip() for p in phrases_str.split(",") if p.strip()]
    
    if not phrases:
        print("WARNING (Sanity): Recall phrases empty after parsing")
        return
    
    print(f"DEBUG (Sanity): Pre-computing embeddings for {len(phrases)} recall phrases...")
    RECALL_INTENT_EMBEDDINGS = embed_documents(phrases, convert_to_numpy=True, show_progress_bar=False)
    print(f"DEBUG (Sanity): Intent embeddings ready (shape: {RECALL_INTENT_EMBEDDINGS.shape})")

    # --- Negative / greeting phrases ---
    negative_str = SANITY_CHECK_SETTINGS.get("sanity_check_negative_phrases", "")
    if negative_str:
        neg_phrases = [p.strip() for p in negative_str.split(",") if p.strip()]
        if neg_phrases:
            print(f"DEBUG (Sanity): Pre-computing embeddings for {len(neg_phrases)} negative phrases...")
            NEGATIVE_INTENT_EMBEDDINGS = embed_documents(neg_phrases, convert_to_numpy=True, show_progress_bar=False)
            print(f"DEBUG (Sanity): Negative embeddings ready (shape: {NEGATIVE_INTENT_EMBEDDINGS.shape})")
        else:
            NEGATIVE_INTENT_EMBEDDINGS = None
    else:
        NEGATIVE_INTENT_EMBEDDINGS = None
        print("DEBUG (Sanity): No negative phrases defined - skipping negative embeddings")


def check_recall_intent(query, threshold=0.40):
    """
    Phase 1 intent gate for Stage 3.

    Two paths:
      DLC ON  → check_recall_intent_zeroshot() — NLI entailment score
      DLC OFF → cosine similarity vs phrase embeddings (vanilla, zero extra RAM)

    Returns (has_intent: bool, score: float).
    """
    # --- ZEROSHOT DLC PATH ---
    if SANITY_CHECK_SETTINGS.get("zeroshot_intent_enabled", False):
        return check_recall_intent_zeroshot(query)

    # --- VANILLA FALLBACK PATH ---
    if RECALL_INTENT_EMBEDDINGS is None:
        print("WARNING (Sanity Intent): No intent embeddings loaded - assuming intent present")
        return True, 1.0

    query_embedding = embed_query(query, convert_to_numpy=True, show_progress_bar=False)
    similarities = util.cos_sim(query_embedding, RECALL_INTENT_EMBEDDINGS)[0]
    max_score = float(max(similarities))
    show_scores = SANITY_CHECK_SETTINGS.get("sanity_check_show_scores", True)

    if NEGATIVE_INTENT_EMBEDDINGS is not None:
        neg_similarities = util.cos_sim(query_embedding, NEGATIVE_INTENT_EMBEDDINGS)[0]
        max_neg_score = float(max(neg_similarities))
        if show_scores:
            print(f"DEBUG (Sanity Intent): Positive score: {max_score:.3f} | Negative score: {max_neg_score:.3f}")
        if max_neg_score > max_score:
            print(f"DEBUG (Sanity Intent): BLOCK - query looks more like a greeting (neg {max_neg_score:.3f} > pos {max_score:.3f})")
            return False, max_neg_score  # FIX BUG 4: was max_score — return the score that triggered the block
    else:
        if show_scores:
            print(f"DEBUG (Sanity Intent): Query intent score: {max_score:.3f}")

    has_intent = max_score >= threshold
    print(f"DEBUG (Sanity Intent): {'PASS' if has_intent else 'BLOCK'} (score: {max_score:.3f}, threshold: {threshold})")
    return has_intent, max_score


def check_content_similarity(query, memories, threshold=0.35):
    """
    Filters memories by semantic similarity to query.
    
    Args:
        query: User's search query
        memories: List of memory dicts that passed previous stages
        threshold: Minimum cosine similarity to pass
    
    Returns:
        List of memories that passed content check with sanity_score added
    """
    if not memories:
        return []
    
    # Embed query
    query_embedding = embed_query(query, convert_to_numpy=True, show_progress_bar=False)
    
    # Embed all memory contents
    memory_texts = [m.get("content", "") for m in memories]
    memory_embeddings = embed_documents(memory_texts, convert_to_numpy=True, show_progress_bar=False)
    
    # Compute similarities
    similarities = util.cos_sim(query_embedding, memory_embeddings)[0]
    
    # Filter by threshold
    passed = []
    show_scores = SANITY_CHECK_SETTINGS.get("sanity_check_show_scores", True)
    for idx, memory in enumerate(memories):
        content_score = float(similarities[idx])
        
        if content_score >= threshold:
            memory["sanity_score"] = content_score
            memory["passed_content"] = True
            passed.append(memory)
            if show_scores:
                print(f"DEBUG (Sanity Content): Memory {idx+1} PASS (score: {content_score:.3f})")
        else:
            if show_scores:
                print(f"DEBUG (Sanity Content): Memory {idx+1} BLOCK (score: {content_score:.3f} < {threshold})")
    
    return passed


def sanity_check_filter(query, memories, skip_intent_check=False):
    """
    Stage 3: Sanity Check - Two-phase filter using active embedding model only.
    
    Phase 1: Intent Detection - Does the query express recall intent?
             SKIPPED when skip_intent_check=True (explicit [RECALL:] tool calls).
             The LLM already expressed intent by invoking the tool — running intent
             detection on its search term (e.g. "tank discussion") makes no sense
             and blocks valid recalls.
    Phase 2: Content Similarity - Are the memories relevant to the query?
    
    Args:
        query: The search query (user message for passive RAG, tool query for [RECALL:])
        memories: Memories that survived Stage 1 (FAISS) and Stage 2 (Reranker)
        skip_intent_check: True when called from an explicit recall tool — bypass Phase 1
    
    Returns:
        Filtered list of memories
    """
    if not memories:
        return []
    
    intent_enabled = SANITY_CHECK_SETTINGS.get("sanity_check_intent_enabled", True)
    intent_threshold = SANITY_CHECK_SETTINGS.get("sanity_check_intent_threshold", 0.40)
    content_threshold = SANITY_CHECK_SETTINGS.get("sanity_check_content_threshold", 0.35)
    
    print(f"DEBUG (Sanity): Running Stage 3 on {len(memories)} candidates... (skip_intent={skip_intent_check})")
    
    # ========================================
    # PHASE 1: Intent Detection (optional)
    # ========================================
    if intent_enabled and not skip_intent_check:
        has_intent, intent_score = check_recall_intent(query, intent_threshold)
        
        if not has_intent:
            print(f"DEBUG (Sanity): No recall intent detected - blocking all {len(memories)} memories")
            return []
        
        # Add intent score to all memories
        for memory in memories:
            memory["intent_score"] = intent_score
            memory["passed_intent"] = True
    else:
        print(f"DEBUG (Sanity): Intent check DISABLED - proceeding to content check")
    
    # ========================================
    # PHASE 2: Content Similarity
    # ========================================
    print(f"DEBUG (Sanity): Running content similarity check...")
    passed = check_content_similarity(query, memories, content_threshold)
    
    print(f"DEBUG (Sanity): Stage 3 complete - {len(passed)}/{len(memories)} memories passed")
    
    return passed


def save_sanity_check_settings():
    """Persist sanity check settings to disk."""
    try:
        with open(SANITY_CHECK_SETTINGS_FILE, 'w') as f:
            json.dump(SANITY_CHECK_SETTINGS, f, indent=2)
        print(f"DEBUG (Sanity): Settings saved to {SANITY_CHECK_SETTINGS_FILE}")
    except Exception as e:
        print(f"ERROR (Sanity): Failed to save settings: {e}")


def load_sanity_check_settings():
    """Load sanity check settings from disk."""
    global SANITY_CHECK_SETTINGS
    
    if os.path.exists(SANITY_CHECK_SETTINGS_FILE):
        try:
            with open(SANITY_CHECK_SETTINGS_FILE, 'r') as f:
                loaded = json.load(f)
                SANITY_CHECK_SETTINGS.update(loaded)
            print(f"DEBUG (Sanity): Settings loaded from {SANITY_CHECK_SETTINGS_FILE}")
        except Exception as e:
            print(f"ERROR (Sanity): Failed to load settings: {e}")
    else:
        print(f"DEBUG (Sanity): No settings file found, using defaults")
        save_sanity_check_settings()  # Create default file


# ============================================================================
# END STAGE 3 SANITY CHECK FUNCTIONS
# ============================================================================

# (Zero-Shot Smart Filter removed — Raw RAG passive recall is the only recall method)

# --- Global Status Tracker ---
# This dictionary will track if a session has post-stream processing running.
POST_STREAM_PROCESSING_STATUS = {}
# --- ADDED: Threading lock for POST_STREAM_PROCESSING_STATUS ---
post_stream_status_lock = threading.Lock()


# --- Configuration Constants ---
SESSION_DIR = "sessions"
GLOBAL_MEMORY_DIR = "global_memory"
CURRENT_SESSION_FILE = "current_session.txt"
# RP_MODE_TOGGLE_FILE = "rp_mode_toggle.txt"  # NEW: Toggle for RP features
# KOBOLD_STYLE_MODELS_PREFIXES removed — was defined but never referenced (BUG 18 fix).


# --- NEW: Settings Management ---
APP_CONFIG_FILE = "app_config.json"
API_PROFILES_FILE = "api_profiles.json"
CURRENT_API_PROFILE_FILE = "current_api_profile.txt"
API_PROFILES = {}  # Global cache
APP_SETTINGS = {}  # Will be loaded at startup
TEMPERATURE_CONFIG_FILE = "temperatures.json"
TEMPERATURE_SETTINGS = {}  # Will be loaded at startup
TOKENS_CONFIG_FILE = "tokens_config.json"
TOKEN_SETTINGS = {}  # Will be loaded at startup
RESONANCE_CONFIG_FILE = "resonance_config.json"
# FIX BUG 2: Pre-seed with safe defaults so create_faiss_index() and prepare_vectors()
# always have a valid faiss_distance_metric even if called before load_resonance_settings().
RESONANCE_SETTINGS = {"faiss_distance_metric": "l2"}  # Will be fully populated at startup
# --- NEW: Sampler Settings Management ---
SAMPLER_CONFIG_FILE = "sampler_config.json"
SAMPLER_SETTINGS = {} # Will be loaded at startup
# --- NEW: Search Settings Management ---
SEARCH_CONFIG_FILE = "search_config.json"
SEARCH_SETTINGS = {} # Will be loaded at startup
# --- NEW: Name Settings Management ---
NAMES_CONFIG_FILE = "names_config.json"
NAME_SETTINGS = {} # Will be loaded at startup
# --- NEW: Appearance Settings Management ---
APPEARANCE_CONFIG_FILE = "appearance_config.json"
APPEARANCE_SETTINGS = {} # Will be loaded at startup
# --- NEW: Streaming Settings Management ---
STREAMING_CONFIG_FILE = "streaming_config.json"
STREAMING_SETTINGS = {} # Will be loaded at startup
def load_app_settings():
    """Loads application settings from the JSON file or sets defaults."""
    global APP_SETTINGS
    defaults = {
        # --- FALLBACK 1: Changed default LLM endpoint to localhost ---
        "llm_api_endpoint": "http://127.0.0.1:5001/v1/chat/completions",
        "llm_model_name": "",
        "llm_api_key": "",
        "enable_scraping": True,
        "scraping_word_limit": 1000,
        "search_max_results": 5,
        "fuzzy_match_threshold": 95,
        # --- BACKEND MODE ---
        "backend_mode": "kobold",
        "reasoning_enabled": False,
        "vision_enabled": False,
        # --- PER-MODE ISOLATED CONFIGS ---
        "kobold_endpoint": "http://127.0.0.1:5001/v1/chat/completions",
        "openrouter_endpoint": "https://openrouter.ai/api/v1/chat/completions",
        "openrouter_model": "",
        "openrouter_api_key": "",
        # --- OPENROUTER SAMPLER LANE (isolated from Kobold vanilla) ---
        "openrouter_max_tokens": 1024,
        "openrouter_temperature": 0.7,
        "openrouter_top_p": 0.9,
    }
    if os.path.exists(APP_CONFIG_FILE):
        with open(APP_CONFIG_FILE, "r") as f:
            try:
                APP_SETTINGS = json.load(f)
                # Ensure all default keys are present
                for key, value in defaults.items():
                    APP_SETTINGS.setdefault(key, value)
                print("DEBUG: Loaded application settings.")
            except json.JSONDecodeError:
                print(f"WARNING: {APP_CONFIG_FILE} is corrupted. Using defaults.")
                APP_SETTINGS = defaults
    else:
        print(f"INFO: {APP_CONFIG_FILE} not found. Creating with defaults.")
        APP_SETTINGS = defaults
    save_app_settings()


def save_app_settings():
    """Saves the current application settings to the JSON file."""
    try:
        with open(APP_CONFIG_FILE, "w") as f:
            json.dump(APP_SETTINGS, f, indent=4)
        print("DEBUG: Saved application settings.")
    except Exception as e:
        print(f"ERROR: Failed to save application settings: {e}")


def get_current_api_profile_name():
    """Returns the name of the currently active API profile."""
    if os.path.exists(CURRENT_API_PROFILE_FILE):
        with open(CURRENT_API_PROFILE_FILE, "r") as f:
            return f.read().strip()
    return "Default"


def set_current_api_profile_name(name):
    """Persists the active API profile name to disk."""
    with open(CURRENT_API_PROFILE_FILE, "w") as f:
        f.write(name)


def initialize_api_profiles():
    """Loads all API profiles from disk into global cache at startup."""
    global API_PROFILES
    if os.path.exists(API_PROFILES_FILE):
        with open(API_PROFILES_FILE, "r", encoding="utf-8") as f:
            try:
                API_PROFILES = json.load(f)
                print(f"DEBUG: Loaded API profiles from {API_PROFILES_FILE}.")
            except json.JSONDecodeError:
                print(f"WARNING: {API_PROFILES_FILE} corrupted. Starting fresh.")
                API_PROFILES = {}
    else:
        API_PROFILES = {}

    # Seed Default profile from current app_config so nothing breaks on first run
    if "Default" not in API_PROFILES:
        API_PROFILES["Default"] = {
            "endpoint": APP_SETTINGS.get("llm_api_endpoint", "http://127.0.0.1:5001/v1/chat/completions"),
            "model":    APP_SETTINGS.get("llm_model_name", ""),
            "api_key":  APP_SETTINGS.get("llm_api_key", ""),
        }
        if not os.path.exists(CURRENT_API_PROFILE_FILE):
            set_current_api_profile_name("Default")
        _save_api_profiles()
        print("DEBUG: Default API profile seeded from app_config.")


def _save_api_profiles():
    """Persists API_PROFILES cache to disk."""
    try:
        with open(API_PROFILES_FILE, "w", encoding="utf-8") as f:
            json.dump(API_PROFILES, f, indent=4)
        print("DEBUG: API profiles saved.")
    except Exception as e:
        print(f"ERROR: Failed to save API profiles: {e}")


def load_temperature_settings():
    """Loads temperature settings from the JSON file or sets defaults."""
    global TEMPERATURE_SETTINGS
    defaults = {
        "chat_stream": 0.4,
        "summarization": 0.1,
        # "location_classification": 0.1,
        # "ambience_generation": 0.4
    }
    if os.path.exists(TEMPERATURE_CONFIG_FILE):
        with open(TEMPERATURE_CONFIG_FILE, "r") as f:
            try:
                TEMPERATURE_SETTINGS = json.load(f)
                # Ensure all default keys are present
                for key, value in defaults.items():
                    TEMPERATURE_SETTINGS.setdefault(key, value)
                print("DEBUG: Loaded temperature settings.")
            except json.JSONDecodeError:
                print("WARNING: temperatures.json is corrupted. Using defaults.")
                TEMPERATURE_SETTINGS = defaults
    else:
        print("INFO: temperatures.json not found. Creating with defaults.")
        TEMPERATURE_SETTINGS = defaults
    save_temperature_settings()  # Save to create the file if it doesn't exist


def save_temperature_settings():
    """Saves the current temperature settings to the JSON file."""
    try:
        with open(TEMPERATURE_CONFIG_FILE, "w") as f:
            json.dump(TEMPERATURE_SETTINGS, f, indent=4)
        print("DEBUG: Saved temperature settings.")
    except Exception as e:
        print(f"ERROR: Failed to save temperature settings: {e}")


def load_token_settings():
    """Loads token settings from the JSON file or sets defaults."""
    global TOKEN_SETTINGS
    defaults = {
        "max_chat_messages": 16,  # Default to even number
        "llm_max_tokens": 1024,
        "resonance_enabled": True,
        "coding_mode_enabled": False,
        "ghost_memory_enabled": True,
        "precision_mode_enabled": False # NEW: Add precision mode toggle
    }
    if os.path.exists(TOKENS_CONFIG_FILE):
        with open(TOKENS_CONFIG_FILE, "r") as f:
            try:
                TOKEN_SETTINGS = json.load(f)
                # Ensure all default keys are present
                for key, value in defaults.items():
                    TOKEN_SETTINGS.setdefault(key, value)
                # Enforce even number for max_chat_messages
                if TOKEN_SETTINGS["max_chat_messages"] % 2 != 0:
                    TOKEN_SETTINGS["max_chat_messages"] += 1
                print("DEBUG: Loaded token settings.")
            except json.JSONDecodeError:
                print("WARNING: tokens_config.json is corrupted. Using defaults.")
                TOKEN_SETTINGS = defaults
    else:
        print("INFO: tokens_config.json not found. Creating with defaults.")
        TOKEN_SETTINGS = defaults
    save_token_settings()  # Save to create the file if it doesn't exist


def save_token_settings():
    """Saves the current token settings to the JSON file."""
    try:
        with open(TOKENS_CONFIG_FILE, "w") as f:
            json.dump(TOKEN_SETTINGS, f, indent=4)
        print("DEBUG: Saved token settings.")
    except Exception as e:
        print(f"ERROR: Failed to save token settings: {e}")


def load_resonance_settings():
    """Loads resonance settings from the JSON file or sets defaults."""
    global RESONANCE_SETTINGS
    defaults = {
        "recalled_message_char_limit": 30000,
        "persistent_memory_injection": True,
        "rag_ghost_preservation": True,
        "faiss_permanent_indexing": True,
        "faiss_indexing_quota": 50,
        "faiss_distance_metric": "l2",
        # --- Asymmetric Embedding Prefixes ---
        # "auto"  = use model-specific query/document prefixes (recommended for nomic, jina)
        # "off"   = no prefixes (symmetric mode — safe fallback for all-mini / all-mpnet)
        "embedding_prefix_mode": "auto",
        # --- End Asymmetric Embedding Prefixes ---
        "global_memory_enabled": False,
        "global_memory_session_limit": 3,
        # --- Custom Memory Block Formatting ---
        "memory_block_header": "[THIS BLOCK HERE ARE THE RETRIEVED MEMORIES AND PAST CONVERSATIONS]",
        "memory_block_closer": "[/END OF RETRIEVED MEMORIES FROM PAST CONVERSATIONS]",
        # --- Raw RAG (Always-On Passive Recall) ---
        "raw_rag_enabled": False,
        "raw_rag_score_threshold": 1.0,   # L2: reject if dist > this. Cosine: reject if score < (2 - this)
        "raw_rag_max_recall": 2,           # Keep lean — 1 or 2 recommended
        # "before" = inject recalled block before user message (higher quality, ~100-200 token tax next turn)
        # "after"  = inject after user message (seamless, ~7-24 token cost, slight quality tradeoff)
        "recall_injection_position": "after",
        # --- Paired Memory ---
        # When ON: the recalled hit is always returned as a full exchange
        # (the message that was found + its conversation partner).
        # If the hit is a user message  → also return the assistant reply after it.
        # If the hit is an assistant msg → also return the user message before it.
        # When OFF: default behaviour — just the single matched message.
        "paired_memory_enabled": False,
        # --- Always Index Messages ---
        "always_index_messages": False,    # If True, every new message is indexed into FAISS immediately
        # =====================================================================
        # --- CHUNKING DLC ---
        # Master switch — OFF = 100% vanilla behavior, zero code path changes.
        # ON = all content is chunked before indexing, system/persona/injected
        # memory is indexed immediately on injection.
        # =====================================================================
        "chunking_enabled": False,
        "chunk_token_limit": 400,          # Max tokens per chunk
        "chunk_overlap_tokens": 100,       # Overlap tokens carried into next chunk
        "chunk_unit": "tokens",            # "tokens" or "words" (UI display toggle)
        "chunk_retrieval_mode": "precision",  # "precision" or "long_context"
        "long_context_threshold": 2,       # Chunks from same source before full reassembly
        "chunk_pinned_always_reassemble": True,  # Pinned sources always reassemble fully
        "chunk_index_system_immediately": True,   # Index persona/memory/intros on injection
        # --- RERANKER SETTINGS (Two-Stage Retrieval) ---
        "reranker_enabled": False,
        "reranker_model": "ms-marco-mini-v2",
        "reranker_expansion_factor": 3,     # Retrieve N×factor candidates, rerank to top N
        "reranker_ctx_length": None,        # For high-ctx models (8K/32K)
        "reranker_score_threshold": 0.0,    # Minimum reranker score to pass
        "reranker_batch_size": 32,          # Batch size for scoring (RAM control)
    }
    
    settings_loaded_from_file = False
    if os.path.exists(RESONANCE_CONFIG_FILE):
        with open(RESONANCE_CONFIG_FILE, "r") as f:
            try:
                RESONANCE_SETTINGS = json.load(f)
                settings_loaded_from_file = True
                print("DEBUG: Loaded resonance settings.")
            except json.JSONDecodeError:
                print(f"WARNING: {RESONANCE_CONFIG_FILE} is corrupted. Using defaults.")
                RESONANCE_SETTINGS = defaults

    updated = False
    for key, value in defaults.items():
        if key not in RESONANCE_SETTINGS:
            RESONANCE_SETTINGS[key] = value
            updated = True
    
    if updated or not settings_loaded_from_file:
        save_resonance_settings()


def save_resonance_settings():
    """Saves the current resonance settings to the JSON file."""
    try:
        with open(RESONANCE_CONFIG_FILE, "w") as f:
            json.dump(RESONANCE_SETTINGS, f, indent=4)
        print("DEBUG: Saved resonance settings.")
    except Exception as e:
        print(f"ERROR: Failed to save resonance settings: {e}")


# =============================================================================
# --- CHUNKING DLC CORE ---
# =============================================================================

def chunk_text(text, token_limit=None, overlap_tokens=None):
    """
    Splits text into chunks respecting natural boundaries.
    Strategy: paragraph → sentence → word fallback.
    Each chunk carries overlap_tokens from the tail of the previous chunk
    so context is never lost at a boundary cut.

    Returns a list of chunk strings.
    If chunking DLC is OFF or text is short enough, returns [text] (single chunk).
    """
    if not text or not text.strip():
        return [text]

    # Read live settings — use `is None` so 0 is respected as a valid "no overlap" value
    if token_limit is None:
        token_limit = RESONANCE_SETTINGS.get("chunk_token_limit", 400)
    if overlap_tokens is None:
        # FIX BUG 26: `or` would replace 0 with the default (100), disabling the
        # user's "no overlap" setting. Use explicit None check instead.
        overlap_tokens = RESONANCE_SETTINGS.get("chunk_overlap_tokens", 100)

    # If text fits in one chunk, no splitting needed
    if count_tokens(text) <= token_limit:
        return [text]

    # --- 1. Split into paragraphs first (double newline = paragraph boundary) ---
    raw_paragraphs = re.split(r'\n{2,}', text)
    paragraphs = [p.strip() for p in raw_paragraphs if p.strip()]

    # --- 2. If any paragraph itself exceeds the limit, split by sentence ---
    segments = []
    for para in paragraphs:
        if count_tokens(para) <= token_limit:
            segments.append(para)
        else:
            # Split by sentence boundaries: ". ", "! ", "? "
            sentences = re.split(r'(?<=[.!?])\s+', para)
            current = ""
            for sent in sentences:
                probe = (current + " " + sent).strip() if current else sent
                if count_tokens(probe) <= token_limit:
                    current = probe
                else:
                    if current:
                        segments.append(current)
                    # If a single sentence is still too long, fall back to word split
                    if count_tokens(sent) > token_limit:
                        words = sent.split()
                        word_buf = ""
                        for w in words:
                            probe_w = (word_buf + " " + w).strip() if word_buf else w
                            if count_tokens(probe_w) <= token_limit:
                                word_buf = probe_w
                            else:
                                if word_buf:
                                    segments.append(word_buf)
                                word_buf = w
                        if word_buf:
                            segments.append(word_buf)
                    else:
                        current = sent
            if current:
                segments.append(current)

    if not segments:
        return [text]

    # --- 3. Pack segments into chunks with overlap ---
    chunks = []
    current_chunk_tokens = []   # list of segment strings in current chunk
    current_token_count = 0
    overlap_tail = ""           # carried-over tail from previous chunk

    for seg in segments:
        seg_tokens = count_tokens(seg)

        # If adding this segment would overflow the chunk, flush first
        if current_token_count + seg_tokens > token_limit and current_chunk_tokens:
            chunk_text_str = overlap_tail + (" " if overlap_tail else "") + " ".join(current_chunk_tokens)
            chunks.append(chunk_text_str.strip())

            # Build overlap tail: take tokens from the end of the flushed chunk
            flushed = " ".join(current_chunk_tokens)
            flushed_words = flushed.split()
            tail_words = []
            tail_count = 0
            for w in reversed(flushed_words):
                wt = count_tokens(w)
                if tail_count + wt <= overlap_tokens:
                    tail_words.insert(0, w)
                    tail_count += wt
                else:
                    break
            overlap_tail = " ".join(tail_words)

            current_chunk_tokens = []
            current_token_count = 0

        current_chunk_tokens.append(seg)
        current_token_count += seg_tokens

    # Flush remaining
    if current_chunk_tokens:
        chunk_text_str = overlap_tail + (" " if overlap_tail else "") + " ".join(current_chunk_tokens)
        chunks.append(chunk_text_str.strip())

    return chunks if chunks else [text]


def _index_system_content_immediately(content, role, source_label, session_id, pinned=False):
    """
    Immediately indexes system-tier content (persona, silent intro, visible intro)
    into the FAISS index on injection, when chunking DLC is enabled.

    - Chunks the content using chunk_text() if it exceeds the token limit.
    - Each chunk is stored as a separate vector with source metadata.
    - Deduplicates by source_label: old chunks from the same source are replaced.
    - KV cache safe: chunk content strings are plain text, no metadata bleeds
      into the injected recall block.
    - pinned=True: during long_context reassembly this source always reassembles
      fully regardless of threshold.
    """
    if EMBEDDING_MODEL is None:
        print("DEBUG (ChunkDLC): Embedding model not loaded, skipping system index.")
        return
    if not content or not content.strip():
        return

    use_permanent = RESONANCE_SETTINGS.get("faiss_permanent_indexing", False)
    if not use_permanent:
        print("DEBUG (ChunkDLC): Permanent indexing OFF — skipping system content index.")
        return

    index_filepath    = get_faiss_index_filepath(session_id)
    metadata_filepath = get_faiss_metadata_filepath(session_id)

    # --- Load existing index + metadata ---
    existing_index   = None
    indexed_messages = []

    if os.path.exists(index_filepath) and os.path.exists(metadata_filepath):
        try:
            temp_index = _safe_faiss_read(index_filepath)
            indexed_messages, index_model = load_faiss_metadata(metadata_filepath)
            current_model = RESONANCE_SETTINGS.get("embedding_model", "nomic")
            if temp_index.d != EMBEDDING_DIM:
                print(f"DEBUG (ChunkDLC): Dimension mismatch. Starting fresh.")
            elif index_model is not None and index_model != current_model:
                print(f"DEBUG (ChunkDLC): Model mismatch. Starting fresh.")
            else:
                existing_index = temp_index
        except Exception as e:
            print(f"DEBUG (ChunkDLC): Could not load existing index ({e}). Starting fresh.")

    if existing_index is None:
        existing_index = create_faiss_index(EMBEDDING_DIM)
        indexed_messages = []

    # --- Remove any old chunks from this source_label (re-injection / update) ---
    indexed_messages = [m for m in indexed_messages if m.get("chunk_source") != source_label]
    # Rebuild index clean from remaining messages to remove stale vectors
    if indexed_messages:
        try:
            surviving_texts = [m.get("content", "") for m in indexed_messages]
            surviving_vecs  = embed_documents(surviving_texts, show_progress_bar=False)
            surviving_np    = np.array(surviving_vecs, dtype="float32")
            surviving_np    = prepare_vectors(surviving_np)
            existing_index  = create_faiss_index(EMBEDDING_DIM)
            existing_index.add(surviving_np)
        except Exception as e:
            print(f"DEBUG (ChunkDLC): Failed to rebuild after source removal ({e}). Using empty index.")
            existing_index   = create_faiss_index(EMBEDDING_DIM)
            indexed_messages = []
    else:
        existing_index = create_faiss_index(EMBEDDING_DIM)

    # --- Chunk the content ---
    chunks = chunk_text(content)
    total  = len(chunks)
    print(f"DEBUG (ChunkDLC): Indexing system content '{source_label}' → {total} chunk(s).")

    new_entries = []
    for idx, chunk in enumerate(chunks):
        if not chunk.strip():
            continue
        new_entries.append({
            "role":          role,
            "content":       chunk,
            "chunk_source":  source_label,
            "chunk_index":   idx,
            "chunk_total":   total,
            "pinned":        pinned,
            "timestamp":     datetime.now().isoformat(timespec='seconds'),
            "always_indexed": True
        })

    if not new_entries:
        return

    try:
        chunk_texts = [e["content"] for e in new_entries]
        vecs        = embed_documents(chunk_texts, show_progress_bar=False)
        vecs_np     = np.array(vecs, dtype="float32")
        vecs_np     = prepare_vectors(vecs_np)
        existing_index.add(vecs_np)
    except Exception as e:
        print(f"DEBUG (ChunkDLC): Encoding failed ({e}). Aborting system index.")
        return

    indexed_messages.extend(new_entries)
    with _faiss_write_lock:
        faiss.write_index(existing_index, index_filepath)
    save_faiss_metadata(metadata_filepath, indexed_messages)
    print(f"DEBUG (ChunkDLC): '{source_label}' indexed — {total} chunks, "
          f"{existing_index.ntotal} total vectors.")

# =============================================================================
# --- END CHUNKING DLC CORE ---
# =============================================================================

# --- NEW: Functions for Sampler Settings ---
def load_sampler_settings():
    """Loads sampler settings from the JSON file or sets defaults."""
    global SAMPLER_SETTINGS
    defaults = {
        "top_p": 0.9,
        "top_k": 40,
        "min_p": 0.05,
        "repetition_penalty": 1.1
    }
    if os.path.exists(SAMPLER_CONFIG_FILE):
        with open(SAMPLER_CONFIG_FILE, "r") as f:
            try:
                SAMPLER_SETTINGS = json.load(f)
                # Ensure all default keys are present
                for key, value in defaults.items():
                    SAMPLER_SETTINGS.setdefault(key, value)
                print("DEBUG: Loaded sampler settings.")
            except json.JSONDecodeError:
                print(f"WARNING: {SAMPLER_CONFIG_FILE} is corrupted. Using defaults.")
                SAMPLER_SETTINGS = defaults
    else:
        print(f"INFO: {SAMPLER_CONFIG_FILE} not found. Creating with defaults.")
        SAMPLER_SETTINGS = defaults
    save_sampler_settings()

def save_sampler_settings():
    """Saves the current sampler settings to the JSON file."""
    with open(SAMPLER_CONFIG_FILE, "w") as f:
        json.dump(SAMPLER_SETTINGS, f, indent=4)
    print("DEBUG: Saved sampler settings.")


# --- NEW: Functions for Search Settings ---
def load_search_settings():
    """Loads search settings from the JSON file or sets defaults."""
    global SEARCH_SETTINGS
    defaults = {
        "scrapeable_domains": ["wikipedia.org", "fandom.com"],
        "search_phrase_word_count": 10,
        "search_tool_trigger": "<tool_search>",
        "search_tool_closer": "</tool_search>",
        "recall_tool_trigger": "[RECALL:",
        "recall_tool_closer": "]",
        "search_result_header": "[Search Results]:",
        "recall_result_header": "[Recall Results]:",
        "persist_search_results": False,
        "search_result_pinned": True,
    }

    if os.path.exists(SEARCH_CONFIG_FILE):
        with open(SEARCH_CONFIG_FILE, "r") as f:
            try:
                SEARCH_SETTINGS = json.load(f)
                # Ensure all default keys are present
                for key, value in defaults.items():
                    SEARCH_SETTINGS.setdefault(key, value)
                print("DEBUG: Loaded search settings.")
            except json.JSONDecodeError:
                print(f"WARNING: {SEARCH_CONFIG_FILE} is corrupted. Using defaults.")
                SEARCH_SETTINGS = defaults
    else:
        print(f"INFO: {SEARCH_CONFIG_FILE} not found. Creating with defaults.")
        SEARCH_SETTINGS = defaults
    save_search_settings()


def save_search_settings():
    """Saves the current search settings to the JSON file."""
    with open(SEARCH_CONFIG_FILE, "w") as f:
        json.dump(SEARCH_SETTINGS, f, indent=4)
    print("DEBUG: Saved search settings.")


# --- NEW: Functions for Name Settings ---
def load_name_settings():
    """Loads name settings from the JSON file or sets defaults."""
    global NAME_SETTINGS
    defaults = {
        "user_name": "User",
        "assistant_name": "Assistant"  # This is a fallback, will be overridden by persona
    }
    if os.path.exists(NAMES_CONFIG_FILE):
        with open(NAMES_CONFIG_FILE, "r") as f:
            try:
                NAME_SETTINGS = json.load(f)
                # Ensure all default keys are present
                for key, value in defaults.items():
                    NAME_SETTINGS.setdefault(key, value)
                print("DEBUG: Loaded name settings.")
            except json.JSONDecodeError:
                print(f"WARNING: {NAMES_CONFIG_FILE} is corrupted. Using defaults.")
                NAME_SETTINGS = defaults
    else:
        print(f"INFO: {NAMES_CONFIG_FILE} not found. Creating with defaults.")
        NAME_SETTINGS = defaults
    save_name_settings()


def save_name_settings():
    """Saves the current name settings to the JSON file."""
    with open(NAMES_CONFIG_FILE, "w") as f:
        json.dump(NAME_SETTINGS, f, indent=4)
    print("DEBUG: Saved name settings.")


def load_appearance_settings():
    """Loads appearance settings from the JSON file or sets defaults."""
    global APPEARANCE_SETTINGS
    defaults = {
        "assistantAvatar": "🐺",
        "assistantAvatarImg": None,
        "thinkingIndicator": "🦊",
        "avatarSize": 34,
        "fontSize": 15,
        "thinkVisibility": "visible",
        "vanillaMode": "off",
        "colors": {
            "plain": "#fffb80",
            "bold": "#ffffff",
            "italic": "#ffffff"
        },
        "assistantAvatarShape": "circle",
        "userAvatarShape": "circle",
        "panelsVisible": True,
    }
    if os.path.exists(APPEARANCE_CONFIG_FILE):
        with open(APPEARANCE_CONFIG_FILE, "r") as f:
            try:
                APPEARANCE_SETTINGS = json.load(f)
                # Ensure nested colors dict is complete
                if "colors" not in APPEARANCE_SETTINGS:
                    APPEARANCE_SETTINGS["colors"] = defaults["colors"]
                else:
                    for k, v in defaults["colors"].items():
                        APPEARANCE_SETTINGS["colors"].setdefault(k, v)
                for key, value in defaults.items():
                    if key != "colors":
                        APPEARANCE_SETTINGS.setdefault(key, value)
                print("DEBUG: Loaded appearance settings.")
            except json.JSONDecodeError:
                print(f"WARNING: {APPEARANCE_CONFIG_FILE} is corrupted. Using defaults.")
                APPEARANCE_SETTINGS = defaults
    else:
        print(f"INFO: {APPEARANCE_CONFIG_FILE} not found. Creating with defaults.")
        APPEARANCE_SETTINGS = defaults
    save_appearance_settings()


def save_appearance_settings():
    """Saves the current appearance settings to the JSON file."""
    with open(APPEARANCE_CONFIG_FILE, "w") as f:
        json.dump(APPEARANCE_SETTINGS, f, indent=4)
    print("DEBUG: Saved appearance settings.")


def load_streaming_settings():
    """Loads streaming settings from the JSON file or sets defaults."""
    global STREAMING_SETTINGS
    defaults = {
        "streamingDelayEnabled": True,  # True = typewriter (casual/roleplay), False = raw TPS (coder)
        "charDelay": 10,
        "punctuationDelay": 40,
        "commaDelay": 20,
    }
    if os.path.exists(STREAMING_CONFIG_FILE):
        with open(STREAMING_CONFIG_FILE, "r") as f:
            try:
                STREAMING_SETTINGS = json.load(f)
                for key, value in defaults.items():
                    STREAMING_SETTINGS.setdefault(key, value)
                print("DEBUG: Loaded streaming settings.")
            except json.JSONDecodeError:
                print(f"WARNING: {STREAMING_CONFIG_FILE} is corrupted. Using defaults.")
                STREAMING_SETTINGS = defaults
    else:
        print(f"INFO: {STREAMING_CONFIG_FILE} not found. Creating with defaults.")
        STREAMING_SETTINGS = defaults
    save_streaming_settings()


def save_streaming_settings():
    """Saves the current streaming settings to the JSON file."""
    try:
        with open(STREAMING_CONFIG_FILE, "w") as f:
            json.dump(STREAMING_SETTINGS, f, indent=4)
        print("DEBUG: Saved streaming settings.")
    except Exception as e:
        print(f"ERROR: Failed to save streaming settings: {e}")


# Regex pattern for detecting URLs
URL_PATTERN = re.compile(r'https?://[^\s]+')

# Initialize tiktoken tokenizer globally for efficiency
# Using o200k_base as it's the tokenizer for modern models like GPT-4o
# and provides a good, efficient approximation for new open-source models.
tokenizer = tiktoken.get_encoding("o200k_base")


def count_tokens(text):
    """Counts the number of tokens in a given text using tiktoken."""
    # A simple check to handle non-string inputs that might arise from complex message formats
    if not isinstance(text, str):
        # If content is a list (multimodal), iterate and sum tokens from text parts
        if isinstance(text, list):
            return sum(count_tokens(part.get("text", "")) for part in text if part.get("type") == "text")
        return 0
    return len(tokenizer.encode(text))


os.makedirs(SESSION_DIR, exist_ok=True)


# END FIX #9
# =============================================================================

# Global variable to store the currently active character name
_current_active_character_name = "Vela"  # Default to Vela


# --- System Prompt Management (now supports multiple personas) ---
PERSONAS_FILE = "personas.json"  # New file to store multiple personas
# Stores the name of the active persona
CURRENT_SYSTEM_PROMPT_NAME_FILE = "current_persona_name.txt"
PERSONAS = {} # MODIFIED: Global cache for personas


def get_current_persona_name():
    """Reads the name of the currently active persona."""
    if os.path.exists(CURRENT_SYSTEM_PROMPT_NAME_FILE):
        with open(CURRENT_SYSTEM_PROMPT_NAME_FILE, "r") as f:
            return f.read().strip()
    return "Vela"  # Default persona name


def set_current_persona_name(persona_name):
    """Sets the name of the currently active persona."""
    with open(CURRENT_SYSTEM_PROMPT_NAME_FILE, "w") as f:
        f.write(persona_name)


def initialize_personas():
    """Loads all personas from the JSON file into the global cache at startup."""
    global PERSONAS
    defaults = {}
    if os.path.exists(PERSONAS_FILE):
        with open(PERSONAS_FILE, "r", encoding='utf-8') as f:
            try:
                PERSONAS = json.load(f)
                print(f"DEBUG: Loaded personas from {PERSONAS_FILE} into cache.")
            except json.JSONDecodeError:
                print(f"WARNING: {PERSONAS_FILE} is corrupted. Starting with defaults.")
                PERSONAS = defaults
    else:
        PERSONAS = defaults

    # Ensure default "Vela" persona exists
    if "Vela" not in PERSONAS:
        default_prompt_content = (
            "You are Vela Beautiful Vixen with amber eyes and amber fur beautiful sleek body sassy and a secret softie unfiltered can talk with anything Vela is 5'3ft tall "
            "Vela is a anthromorphic Feline."
            "Vela has a Bushy amber tail."
            "Vela wears Mary Jane shoes."
            "Vela is Curious and inquisitive."
            "Vela Says 'mew' at the end of sentences."
            "Vela wears a Sparkly collar."
            "Vela wears a Flowy sundress.\n"
            "[Lives or lingers in abandoned places like warehouse, factories, caves, abandoned apartments struggling to make ends meet everyday.]"
            
        )
        PERSONAS["Vela"] = {'prompt': default_prompt_content, 'avatar_image': None}
        if not os.path.exists(CURRENT_SYSTEM_PROMPT_NAME_FILE):
            set_current_persona_name("Vela")
        save_all_personas(PERSONAS)
        print("DEBUG: Default 'Vela' persona created and saved.")

    # Migrate any old string-format personas to dict format
    _migrate_personas_to_dict()


def save_all_personas(personas_to_save):
    """Saves personas to file and updates the global cache."""
    global PERSONAS
    with open(PERSONAS_FILE, "w", encoding='utf-8') as f:
        json.dump(personas_to_save, f, indent=4)
    PERSONAS = personas_to_save # Update global cache
    print(f"DEBUG: All personas saved to {PERSONAS_FILE} and global cache updated.")


# --- User Loadout Management ---
def get_current_user_loadout_name():
    """Reads the name of the currently active user loadout."""
    if os.path.exists(CURRENT_USER_LOADOUT_FILE):
        with open(CURRENT_USER_LOADOUT_FILE, 'r') as f:
            return f.read().strip()
    return 'Default'

def set_current_user_loadout_name(loadout_name):
    """Sets the name of the currently active user loadout."""
    with open(CURRENT_USER_LOADOUT_FILE, 'w') as f:
        f.write(loadout_name)

def initialize_user_loadouts():
    """Loads all user loadouts from the JSON file into the global cache at startup."""
    global USER_LOADOUTS
    if os.path.exists(USER_LOADOUTS_FILE):
        with open(USER_LOADOUTS_FILE, 'r', encoding='utf-8') as f:
            try:
                USER_LOADOUTS = json.load(f)
                print(f"DEBUG: Loaded user loadouts from {USER_LOADOUTS_FILE}.")
            except json.JSONDecodeError:
                print(f"WARNING: {USER_LOADOUTS_FILE} is corrupted. Starting fresh.")
                USER_LOADOUTS = {}
    else:
        USER_LOADOUTS = {}

    if 'Default' not in USER_LOADOUTS:
        USER_LOADOUTS['Default'] = {
            'display_name': 'User',
            'avatar_emoji': '❄️',
            'avatar_image': None,
            'persona_prompt': ''
        }
        if not os.path.exists(CURRENT_USER_LOADOUT_FILE):
            set_current_user_loadout_name('Default')
        save_all_user_loadouts(USER_LOADOUTS)
        print("DEBUG: Default user loadout created.")

def save_all_user_loadouts(loadouts_to_save):
    """Saves user loadouts to file and updates the global cache."""
    global USER_LOADOUTS
    with open(USER_LOADOUTS_FILE, 'w', encoding='utf-8') as f:
        json.dump(loadouts_to_save, f, indent=4)
    USER_LOADOUTS = loadouts_to_save
    print(f"DEBUG: All user loadouts saved.")

def get_current_user_loadout():
    """Returns the full loadout dict for the currently active user loadout."""
    name = get_current_user_loadout_name()
    return USER_LOADOUTS.get(name, USER_LOADOUTS.get('Default', {
        'display_name': 'User',
        'avatar_emoji': '❄️',
        'avatar_image': None,
        'persona_prompt': ''
    }))


# --- Persona Avatar Helpers ---
def get_persona_avatar_filename(persona_name):
    """Returns the avatar image filename for a persona, if set."""
    return PERSONAS.get(persona_name, {}).get('avatar_image') if isinstance(PERSONAS.get(persona_name), dict) else None

def _migrate_personas_to_dict():
    """
    Migrates personas from old string format to new dict format.
    Old: { "Vela": "You are Vela..." }
    New: { "Vela": { "prompt": "You are Vela...", "avatar_image": null } }
    """
    global PERSONAS
    changed = False
    for name, val in PERSONAS.items():
        if isinstance(val, str):
            PERSONAS[name] = {'prompt': val, 'avatar_image': None}
            changed = True
    if changed:
        save_all_personas(PERSONAS)
        print("DEBUG: Migrated personas to dict format.")

# --- END User Loadout Management ---

def load_system_prompt_content(persona_name=None):
    """
    Loads the system prompt content for a specific persona from the global cache.
    If persona_name is None, loads the content of the currently active persona.
    Supports both old string format and new dict format.
    """
    global PERSONAS
    if persona_name is None:
        persona_name = get_current_persona_name()

    val = PERSONAS.get(persona_name, PERSONAS.get("Vela", ""))
    if isinstance(val, dict):
        return val.get('prompt', '')
    return val


def save_system_prompt_content(persona_name, content, make_active=False):
    """
    Saves the system prompt content for a specific persona to cache and file.
    FIX BUG 12: Does NOT switch the active persona unless make_active=True is explicitly
    passed. Previously every save silently activated the persona being edited, making
    the dedicated Activate button meaningless.
    """
    global PERSONAS
    personas_copy = PERSONAS.copy()
    existing = personas_copy.get(persona_name, {})
    if isinstance(existing, dict):
        existing['prompt'] = content
        personas_copy[persona_name] = existing
    else:
        # Migrate: preserve old string, upgrade to dict
        personas_copy[persona_name] = {'prompt': content, 'avatar_image': None}
    save_all_personas(personas_copy)
    if make_active:
        set_current_persona_name(persona_name)


# --- System Prompt ---
def get_system_prompt():
    """
    Returns the system prompt for the LLM and identifies the primary character.
    User/assistant names are now prepended to each message directly.
    """
    # Load the content of the currently active persona
    prompt_content = load_system_prompt_content()
    
    # 1. Get dynamic date info
    current_date = datetime.now().strftime("%A, %B %d, %Y")
    current_time = datetime.now().strftime("%I:%M %p")
    
    # 2. Inject it into the prompt
    # We prepend it as a System Note so it's always the first thing the AI sees
    time_anchor = f"Current Date: {current_date} | Current Time: {current_time}\n"

    # Merge active user loadout persona_prompt into the system prompt (once, at build time).
    # Keeps slot [0] stable — no floating system messages, no KV cache busting.
    # If user switches loadout mid-convo, next turn rebuilds with new merged content.
    user_persona_prompt = get_current_user_loadout().get('persona_prompt', '').strip()
    if user_persona_prompt:
        full_prompt_content = f"{prompt_content}\n\n[User Context]: {user_persona_prompt}"
    else:
        full_prompt_content = prompt_content

    # Character name is always the persona name — no regex needed.
    character_name = get_current_persona_name()

    global _current_active_character_name
    _current_active_character_name = character_name  # Set the global active character

    print(f"DEBUG: Generated system prompt: '{full_prompt_content[:150]}...'")
    return {"role": "system", "content": full_prompt_content}, character_name


def strip_dev_notes(text: str) -> str:
    """
    Removes developer notes from a string using regular expressions.
    It targets text within square brackets, parentheses, and curly braces.

    Args:
        text: The input string.

    Returns:
        The cleaned string.

    Example:
        "best restaurants in Cebu {for a date night}"
        -> "best restaurants in Cebu"
    """
    # This regex finds and replaces text within [], (), or {} with an empty str.
    return re.sub(r'\[.*?\]|\(.*?\)|{.*?}', '', text).strip()


def log_search(
    log_file: str,
    original_query: str,
    cleaned_query: str,
    timestamp: datetime,
    results: list = None,
    error: str = None
):
    """
    Appends a detailed search record to the specified log file.

    Args:
        log_file: Path to the log file.
        original_query: The query as it was passed to the function.
        cleaned_query: The query after removing dev notes.
        timestamp: The datetime object of when the search was performed.
        results: A list of search results. Defaults to None.
        error: An error message string if the search failed. Defaults to None.
    """
    search_record = {
        "timestamp": timestamp.isoformat(),
        "original_query": original_query,
        "cleaned_query": cleaned_query,
        # Log an empty list for no results, which is better for machine parsing.
        "results": results if results is not None else [],
        "error": error
    }
    try:
        # Appends the record as a new line in the JSONL file.
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(search_record, ensure_ascii=False) + "\n")
    except IOError as e:
        print(f"[❌] Logging Error: Failed to write to {log_file}. Reason: {e}")


def scrape_url(url: str) -> str:
    """
    Scrapes the main text content from a given URL, optimized for wiki sites.

    Args:
        url: The URL of the web page to scrape.

    Returns:
        A string containing the extracted text, or an error message.
    """
    if not APP_SETTINGS.get("enable_scraping", False):
        return "Scraping is currently disabled by the administrator."

    print(f"[🔄] Scraping {url}...")
    try:
        headers = {
            'User-Agent': (
                'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                'AppleWebKit/537.36 (KHTML, like Gecko) '
                'Chrome/91.0.4472.124 Safari/537.36'
            )
        }
        response = requests.get(url, headers=headers, timeout=10)
        # Raise an exception for bad status codes (4xx or 5xx)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, 'html.parser')

        # Find the main content area (these selectors are common for Wiki/Fandom)
        content_div = soup.find('div', id='mw-content-text') or \
            soup.find('div', class_='mw-parser-output')

        if not content_div:
            # A generic fallback if specific containers aren't found
            content_div = soup.find('article') or soup.find('main') or soup.body

        if content_div:
            # Remove non-content elements like nav boxes, scripts, and styles
            for element in content_div(
                ['script', 'style', 'table', '.navbox']
            ):
                element.decompose()

            # Get text and clean it up
            text = content_div.get_text(separator=' ', strip=True)
            # Limit content to a reasonable length to avoid overwhelming the AI
            word_limit = APP_SETTINGS.get("scraping_word_limit", 400)
            summary = ' '.join(text.split()[:word_limit]) + '...'
            print(
                "[✅] Successfully scraped and summarized content from "
                f"{url}."
            )
            return summary

        return "Scraping failed: Could not find main content area."

    except requests.RequestException as e:
        error_message = f"Scraping failed: Network error - {e}"
        print(f"[❌] {error_message}")
        return error_message
    except Exception as e:
        error_message = f"Scraping failed: An unexpected error occurred - {e}"
        print(f"[❌] {error_message}")
        return error_message


def duckduckgo_search(query: str, log_file: str = "search_log.jsonl") -> list:
    """
    Universal Search: Performs a search and aggressively extracts timestamps 
    from snippets to satisfy temporal-aware AI models (like 24B/70B).
    """
    search_timestamp = datetime.now()
    cleaned_query = strip_dev_notes(query)

    print(f"[🔎] Universal Search Query: '{cleaned_query}'")

    if not cleaned_query:
        return [{"body": "Search failed: Query was empty."}]

    # 1. 'Smart' Mode Switch (Keep this! It's still the best source of dates)
    # If the user explicitly asks for "news" or "latest", we use the News API 
    # because it guarantees 100% accurate metadata dates.
    news_triggers = ["news", "latest", "update", "recent", "today", "yesterday", "current"]
    is_news_mode = any(trigger in cleaned_query.lower() for trigger in news_triggers)

    results = []
    try:
        max_results = APP_SETTINGS.get("search_max_results", 7)
        
        with DDGS() as ddgs:
            if is_news_mode:
                print(f"[🔎] Mode: NEWS (Guaranteed Dates) for '{cleaned_query}'")
                search_generator = ddgs.news(cleaned_query, max_results=max_results)
            else:
                print(f"[🔎] Mode: STANDARD (Extracted Dates) for '{cleaned_query}'")
                search_generator = ddgs.text(cleaned_query, max_results=max_results)

            if search_generator:
                results = list(search_generator)

        print(f"[✅] Got {len(results)} raw results.")

        # 2. Universal Date Extraction & Formatting
        formatted_results = []
        scrapeable_domains = SEARCH_SETTINGS.get("scrapeable_domains", [])

        # Regex to catch dates at the start of snippets
        # Matches: "Sep 12, 2024 ...", "12 Sep 2024 ...", "2 hours ago ..."
        date_pattern = re.compile(r"^(?:(\d{1,2} [A-Za-z]{3} \d{4})|([A-Za-z]{3} \d{1,2}, \d{4})|(\d+ \w+ ago))")

        for r in results:
            link = r.get("url", r.get("href", ""))
            title = r.get("title", "No Title")
            raw_body = r.get("body", "No content")
            
            # A. Try to get the official date (only exists in News mode)
            official_date = r.get("date", "")
            
            # B. If no official date, try to regex it from the body (Standard mode)
            extracted_date = ""
            final_body = raw_body

            if official_date:
                extracted_date = official_date
            else:
                # Check if the body starts with a date pattern
                match = date_pattern.search(raw_body)
                if match:
                    extracted_date = match.group(0) # The matched date string
                    # Optional: Remove the date from the body text to avoid repetition
                    # final_body = raw_body.replace(extracted_date, "", 1).strip(" .-,")

            # C. Construct the "AI-Friendly" Body with a [Date] Tag
            if extracted_date:
                # We prepend the date tag so the AI sees it FIRST
                body_with_time = f"[Date: {extracted_date}] {final_body}"
            else:
                # If no date found, we mark it as "General" or "Undated" 
                # so the AI knows it's not time-sensitive.
                body_with_time = f"[Date: General/Undated] {final_body}"

            # Add to list
            result_entry = {
                "title": title,
                "body": body_with_time,
                "href": link
            }

            # --- Scraping Logic (Preserved) ---
            if link and scrapeable_domains:
                try:
                    domain = urlparse(link).netloc
                    if any(domain.endswith(d) for d in scrapeable_domains):
                        scraped = scrape_url(link)
                        # If we scraped, we might lose the snippet date, 
                        # so let's re-inject the date tag into the scraped content too!
                        if extracted_date:
                            result_entry["scraped_content"] = f"[Date: {extracted_date}] {scraped}"
                        else:
                            result_entry["scraped_content"] = scraped
                except Exception as e:
                    print(f"DEBUG: Scraping error: {e}")
            
            formatted_results.append(result_entry)

        log_search(log_file, query, cleaned_query, search_timestamp, results=formatted_results)
        return formatted_results

    except Exception as e:
        print(f"[❌] Search Error: {e}")
        # FIX: Add 'title' and 'href' so the loop doesn't crash when accessing them!
        return [{
            "title": "Search Error",
            "body": f"Search failed: {str(e)}",
            "href": "#"
        }]


# Function to fetch and extract content from a URL
def get_page_content(url):
    """Fetches and extracts all visible text from a given URL."""
    try:
        # Use a user-agent header to mimic a browser, which helps avoid blocks
        headers = {
            'User-Agent': (
                'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                'AppleWebKit/537.36 (KHTML, like Gecko) '
                'Chrome/58.0.3029.110 Safari/537.3'
            )
        }
        response = requests.get(url, headers=headers, timeout=10)
        # Raise an HTTPError for 4xx or 5xx status codes
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        # Extract text from the page, skipping script and style tags
        text_content = ' '.join(soup.stripped_strings)

        return text_content
    except requests.exceptions.RequestException as e:
        # Handle exceptions like connection errors, timeouts, etc.
        print(f"Error fetching URL {url}: {e}")
        return None


    
# --- Session & Chat Memory Helpers ---
def get_current_session_id():
    if os.path.exists(CURRENT_SESSION_FILE):
        with open(CURRENT_SESSION_FILE, "r") as f:
            return f.read().strip()
    return "session_001"


def set_current_session_id(session_id):
    with open(CURRENT_SESSION_FILE, "w") as f:
        f.write(session_id)


def get_session_filepath(session_id):
    session_folder = os.path.join(SESSION_DIR, session_id)
    os.makedirs(session_folder, exist_ok=True)
    return os.path.join(session_folder, "chat_memory.json")

# --- NEW: FAISS Helper Functions ---
def get_faiss_index_filepath(session_id):
    """Gets the file path for the permanent Faiss index for a session."""
    session_folder = os.path.join(SESSION_DIR, session_id)
    os.makedirs(session_folder, exist_ok=True)
    return os.path.join(session_folder, "faiss_index.idx")

def get_faiss_metadata_filepath(session_id):
    """Gets the file path for the Faiss metadata (indexed messages) for a session."""
    session_folder = os.path.join(SESSION_DIR, session_id)
    os.makedirs(session_folder, exist_ok=True)
    return os.path.join(session_folder, "faiss_metadata.json")

def load_faiss_metadata(metadata_filepath):
    """Loads the indexed messages and the model name that built the index."""
    if os.path.exists(metadata_filepath):
        with open(metadata_filepath, "r", encoding='utf-8') as f:
            try:
                data = json.load(f)
                messages = data.get("indexed_messages", [])
                model = data.get("embedding_model", None)  # None = old index, no stamp
                return messages, model
            except json.JSONDecodeError:
                print(f"Warning: Faiss metadata file {metadata_filepath} is corrupted. Resetting.")
                return [], None
    return [], None

def save_faiss_metadata(metadata_filepath, indexed_messages):
    """Saves the indexed messages and stamps the current model name."""
    current_model = RESONANCE_SETTINGS.get("embedding_model", "nomic")
    try:
        with open(metadata_filepath, "w", encoding='utf-8') as f:
            json.dump({
                "embedding_model": current_model,
                "indexed_messages": indexed_messages
            }, f, indent=2)
    except Exception as e:
        print(f"ERROR: Failed to save FAISS metadata to {metadata_filepath}: {e}")

# --- Global Memory Helper Functions ---
def get_global_faiss_filepath():
    """Gets the file path for the unified global Faiss index."""
    os.makedirs(GLOBAL_MEMORY_DIR, exist_ok=True)
    return os.path.join(GLOBAL_MEMORY_DIR, "global.faiss")

def get_global_faiss_metadata_filepath():
    """Gets the file path for the unified global Faiss metadata."""
    os.makedirs(GLOBAL_MEMORY_DIR, exist_ok=True)
    return os.path.join(GLOBAL_MEMORY_DIR, "global.meta.json")


# FIX #2: Debounce guard for rebuild_global_index
_rebuild_lock = threading.Lock()
_last_rebuild_time = 0.0
_REBUILD_COOLDOWN_SECONDS = 30  # Don't rebuild more than once per 30s unless forced

def rebuild_global_index(force=False):
    """
    Builds (or rebuilds) a single unified FAISS index from the top N most recent sessions.
    Reads directly from each session's chat_memory.json — does NOT depend on per-session
    .faiss files existing. This is the source of truth for global memory.
    
    Skips rebuild if the global index is already newer than all session files (unless force=True).
    FIX #2: Also skips if a rebuild already ran within the cooldown window (debounce).
    """
    global _last_rebuild_time
    # FIX #2: Non-blocking try-acquire — if another thread is already rebuilding, skip
    if not _rebuild_lock.acquire(blocking=False):
        print("DEBUG (Global Index): Rebuild already in progress. Skipping duplicate call.")
        return
    try:
        now = time.time()
        if not force and (now - _last_rebuild_time) < _REBUILD_COOLDOWN_SECONDS:
            print(f"DEBUG (Global Index): Rebuild debounced (last ran {now - _last_rebuild_time:.1f}s ago). Skipping.")
            return
        _last_rebuild_time = now
        _rebuild_global_index_inner(force=force)
    finally:
        _rebuild_lock.release()

def _rebuild_global_index_inner(force=False):
    """
    Internal implementation of rebuild_global_index (called inside the debounce lock).

    INCREMENTAL MODE (default, force=False):
        Loads the existing global index + metadata, collects candidate messages from
        the top N sessions, deduplicates against already-indexed content, and only
        encodes + adds the delta. Same pattern as the per-session local index.

    FULL REBUILD MODE (force=True):
        Nukes the existing global index and re-encodes everything from scratch.
        Used by the manual Rebuild button in the UI, or after a model/dim change.
    """
    if EMBEDDING_MODEL is None:
        print("ERROR (Global Index): Embedding model not loaded, skipping rebuild.")
        return

    global_index_path = get_global_faiss_filepath()
    global_meta_path  = get_global_faiss_metadata_filepath()

    # --- Staleness check (incremental only): skip if already up to date ---
    if not force and os.path.exists(global_index_path):
        global_mtime = os.path.getmtime(global_index_path)
        newest_session_mtime = 0
        try:
            for d in os.listdir(SESSION_DIR):
                mem_file = os.path.join(SESSION_DIR, d, "chat_memory.json")
                if os.path.exists(mem_file):
                    newest_session_mtime = max(newest_session_mtime, os.path.getmtime(mem_file))
        except Exception:
            pass
        if global_mtime >= newest_session_mtime:
            print("DEBUG (Global Index): Index is up to date. Skipping rebuild.")
            return
    # ----------------------------------------------------------------------

    global_limit = RESONANCE_SETTINGS.get("global_memory_session_limit", 3)

    try:
        # 1. Get top N sessions sorted by modification time (newest first)
        all_sessions = []
        for d in os.listdir(SESSION_DIR):
            full_path = os.path.join(SESSION_DIR, d)
            if os.path.isdir(full_path):
                mem_file = os.path.join(full_path, "chat_memory.json")
                if os.path.exists(mem_file):
                    all_sessions.append((os.path.getmtime(full_path), d))
        all_sessions.sort(reverse=True)
        sessions_to_index = [s[1] for s in all_sessions[:global_limit]]
        print(f"DEBUG (Global Index): Sessions selected: {sessions_to_index}")

        # 2. Collect candidate messages from those sessions, tagged with source.
        # The global index must never contain messages from the active window of
        # the current session — those are already visible to the LLM.
        current_sid = get_current_session_id()
        all_messages = []
        for sid in sessions_to_index:
            mem_file = os.path.join(SESSION_DIR, sid, "chat_memory.json")
            try:
                with open(mem_file, "r", encoding="utf-8") as f:
                    session_memory = json.load(f)
                if sid == current_sid:
                    active_window = ghost_memory_if_needed(session_memory)
                    active_ids_content = {
                        m.get("content", "").strip()
                        for m in active_window
                        if isinstance(m.get("content"), str)
                    }
                else:
                    active_ids_content = set()
                for msg in session_memory:
                    content_str = msg.get("content", "")
                    if (msg.get("role") in ["user", "assistant"] and
                            isinstance(content_str, str) and
                            content_str.strip()):
                        if sid == current_sid and content_str.strip() in active_ids_content:
                            continue
                        tagged = msg.copy()
                        tagged["session_source"] = sid
                        all_messages.append(tagged)
            except Exception as e:
                print(f"ERROR (Global Index): Could not load session {sid}: {e}")

        if not all_messages:
            print("DEBUG (Global Index): No messages found across sessions. Global index not created.")
            return

        # 3. INCREMENTAL: load existing index + metadata, only encode the delta.
        #    FULL REBUILD (force=True): start fresh, encode everything.
        global_index    = None
        indexed_meta    = []
        already_indexed = set()

        if not force and os.path.exists(global_index_path) and os.path.exists(global_meta_path):
            try:
                global_index      = _safe_faiss_read(global_index_path)
                indexed_meta, _   = load_faiss_metadata(global_meta_path)
                if global_index.d != EMBEDDING_DIM:
                    print(f"DEBUG (Global Index): Dim mismatch ({global_index.d} vs {EMBEDDING_DIM}). Forcing full rebuild.")
                    global_index  = None
                    indexed_meta  = []
                else:
                    already_indexed = {m.get("content", "").strip() for m in indexed_meta}
                    print(f"DEBUG (Global Index): Loaded existing global index ({global_index.ntotal} vectors). Running incremental update.")
            except Exception as e:
                print(f"DEBUG (Global Index): Could not load existing global index ({e}). Falling back to full rebuild.")
                global_index  = None
                indexed_meta  = []

        new_messages = [
            m for m in all_messages
            if m.get("content", "").strip() not in already_indexed
        ]

        if not new_messages and global_index is not None:
            print(f"DEBUG (Global Index): No new messages to add. Global index up to date ({global_index.ntotal} vectors).")
            return

        if new_messages:
            print(f"DEBUG (Global Index): Encoding {len(new_messages)} new message(s) (skipped {len(all_messages) - len(new_messages)} already indexed)...")
            texts         = [m.get("content", "") for m in new_messages]
            embeddings    = embed_documents(texts, show_progress_bar=False)
            embeddings_np = np.array(embeddings).astype("float32")
            embeddings_np = prepare_vectors(embeddings_np)
            if global_index is None:
                global_index = create_faiss_index(EMBEDDING_DIM)
            global_index.add(embeddings_np)
            indexed_meta.extend(new_messages)
        else:
            print("DEBUG (Global Index): Full rebuild requested but no new messages found.")
            return

        # 4. Save
        with _faiss_write_lock:
            faiss.write_index(global_index, global_index_path)
        save_faiss_metadata(global_meta_path, indexed_meta)
        print(f"DEBUG (Global Index): Done. {global_index.ntotal} vectors from {len(sessions_to_index)} sessions.")

    except Exception as e:
        print(f"ERROR (Global Index): Failed to rebuild global index: {e}")
        traceback.print_exc()
# --- END: Global Memory Helper Functions ---

# --- END: FAISS Helper Functions ---


def load_memory():
    # Pure JSON store — atomic reads with threading lock.
    with _memory_lock:
        session_id = get_current_session_id()
        filepath = get_session_filepath(session_id)
        if os.path.exists(filepath):
            with open(filepath, "r", encoding='utf-8') as f:
                try:
                    return json.load(f)
                except json.JSONDecodeError:
                    print(
                        f"Warning: Chat memory file for {session_id} is "
                        "corrupted or empty. Starting fresh."
                    )
                    return []
        return []


def save_memory(memory):
    # Pure JSON store — atomic write with tmp file swap.
    with _memory_lock:
        session_id = get_current_session_id()
        filepath = get_session_filepath(session_id)
        tmp_path = filepath + ".tmp"
        with open(tmp_path, "w", encoding='utf-8') as f:
            json.dump(memory, f, indent=2)
        os.replace(tmp_path, filepath)  # atomic on all major OS


def add_message(session_id, role, content, silent=False):
    memory = load_memory()
    message_payload = {"role": role, "content": content}
    if silent:
        message_payload["silent"] = True
    else:
        # Only timestamp visible user/assistant messages, not silent injections
        message_payload["timestamp"] = datetime.now().isoformat(timespec='seconds')
    memory.append(message_payload)
    save_memory(memory)


# --- GHOST MEMORY INTEGRATION (HOLLOW SYSTEM RESTORED) ---
def ghost_memory_if_needed(memory):
    """
    Truncates chat history to `max_chat_messages` while preserving key system messages.
    
    THREE-TIER ARCHITECTURE (HOLLOW SYSTEM):
    1. PRESERVED (Tier 1): Persona, facts, silent intros - always at top, don't count
    2. HOLLOWS (Tier 2): Search results, recalled memories, RAG - PRESENT BUT WEIGHTLESS
       - They exist in context for the LLM but DON'T count toward sliding window
       - Pinned hollows stay forever, unpinned ones slide out naturally
       - CRITICAL: Adding hollows doesn't push out conversation messages!
    3. CONVERSATION (Tier 3): User/assistant messages - count toward window limit
    
    This prevents the 600-1300 token KV cache invalidation that happens when injected
    content pushes out real conversation messages. Hollows cost only 16-24 tokens!
    """
    # If ghost memory (sliding window) is disabled, return full history untouched
    if not TOKEN_SETTINGS.get("ghost_memory_enabled", True):
        return memory

    max_messages = TOKEN_SETTINGS.get("max_chat_messages", 16)
    
    # If memory is already short enough, no need to truncate
    if len(memory) <= max_messages:
        return memory

    preserved_messages = []     # Tier 1: Headers (persona, facts)
    hollow_messages = []        # Tier 2: HOLLOWS (search, recalled, RAG) - THE FIX!
    conversation_messages = []  # Tier 3: Real conversation (user/assistant)
    found_first_system = False

    # Separate messages into three tiers, tracking original indices for hollows & conversation
    for idx, msg in enumerate(memory):
        is_header = False
        is_hollow = False
        
        # HOLLOWS: Search results, recalled memories, RAG injections
        # These are WEIGHTLESS - present in context but don't count toward window!
        if msg.get("_search_result") or msg.get("_recall_result") or msg.get("_raw_rag"):
            is_hollow = True
            hollow_messages.append((idx, msg))  # Track index for natural order
            continue

        # 1. The Persona System Prompt (Only the first one found)
        if msg.get("role") == "system" and not found_first_system and not msg.get("silent"):
            is_header = True
            found_first_system = True
            
        # 2. Silent Intros / Facts — preserved as headers (NOT recalled blocks)
        elif msg.get("role") == "system":
            is_header = True

        if is_header:
            preserved_messages.append(msg)
        else:
            conversation_messages.append((idx, msg))  # Track index for natural order

    # Calculate window based ONLY on conversation (hollows don't have bodies!)
    max_conversation = max_messages - len(preserved_messages)
    if max_conversation < 2: max_conversation = 2
    
    # Trim conversation to window size (hollows NOT counted here!)
    active_conversation = conversation_messages[-max_conversation:] if len(conversation_messages) > max_conversation else conversation_messages
    
    # Determine which hollows to keep (pinned vs natural decay)
    active_hollows = []
    if hollow_messages:
        # Find the oldest message index in the active conversation window
        if active_conversation:
            oldest_active_idx = active_conversation[0][0]
        else:
            oldest_active_idx = len(memory)
        
        for idx, msg in hollow_messages:
            # Pinned hollows: ALWAYS keep (immortal ghosts)
            if msg.get("_pinned"):
                active_hollows.append((idx, msg))
            # Unpinned hollows: keep if newer than or equal to oldest active message
            # (natural decay - they slide out when the conversation moves past them)
            elif idx >= oldest_active_idx:
                active_hollows.append((idx, msg))
            # Otherwise, let it slide out naturally
    
    # ALL active hollows (pinned or not) sort chronologically by original index.
    # _pinned = IMMORTAL (never decays out of window), NOT wall-hugging.
    # _pinned=False = natural decay (slides out when conversation window passes it).
    # Both types stay in their natural chronological position — no teleportation.
    all_sliding = active_conversation + active_hollows
    all_sliding.sort(key=lambda x: x[0])
    final_conversation = [msg for idx, msg in all_sliding]

    # Reconstruct: system headers → conversation (hollows interleaved at natural position)
    final_memory = list(preserved_messages)
    final_memory.extend(final_conversation)

    return final_memory


# --- LLM Interaction Functions ---
# Non-streaming call for specific purposes like fact extraction
def _get_llm_response_non_stream(messages_to_send, temperature=0.4):
    """Sends a non-streaming request to the configured LLM endpoint."""
    model_name = APP_SETTINGS.get("llm_model_name", "").strip()
    # Only send the model key if it looks like a real remote model name (e.g. "openai/gpt-4o").
    # Blank or legacy placeholder values like "assistant" are skipped so local backends
    # (KoboldCPP, LM Studio) that ignore or reject unknown model names keep working.
    _send_model = bool(model_name and model_name.lower() not in ("", "assistant", "none", "local"))
    payload = {
        "temperature": temperature,
        "max_tokens": TOKEN_SETTINGS.get("llm_max_tokens", 1024),
        "stream": False,
        "top_p": SAMPLER_SETTINGS.get("top_p", 0.9),
        "top_k": SAMPLER_SETTINGS.get("top_k", 40),
        "min_p": SAMPLER_SETTINGS.get("min_p", 0.05),
        "repetition_penalty": SAMPLER_SETTINGS.get("repetition_penalty", 1.1),
        "messages": messages_to_send  # messages always last — static prefix caches cleanly
    }
    if _send_model:
        payload["model"] = model_name
    # --- OPENROUTER EXTENSION ---
    if APP_SETTINGS.get("backend_mode") == "openrouter":
        if APP_SETTINGS.get("reasoning_enabled"):
            payload["reasoning"] = {"enabled": True}
        # OpenRouter isolated sampler lane.
        # top_k overridden to 0 (OR's default = disabled) — Kobold default is 40 which would
        # silently activate top-k filtering on OR models that expect it off.
        # min_p and repetition_penalty are kept — OR forwards them to providers that support them.
        payload["max_tokens"]  = APP_SETTINGS.get("openrouter_max_tokens", 1024)
        payload["temperature"] = APP_SETTINGS.get("openrouter_temperature", 0.7)
        payload["top_p"]       = APP_SETTINGS.get("openrouter_top_p", 0.9)
        payload["top_k"]       = 0  # OR default — disabled, let top_p + min_p do the work
    # --- END OPENROUTER EXTENSION ---
    headers = {"Content-Type": "application/json"}
    api_key = APP_SETTINGS.get("llm_api_key", "").strip()
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    start_time = time.time()
    try:
        response = requests.post(
            endpoint,
            headers=headers,
            json=payload,
            timeout=2000
        )
        end_time = time.time()
        print(
            f"DEBUG: LLM non-streaming call took "
            f"{end_time - start_time:.4f} seconds."
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except requests.exceptions.RequestException as e:
        print(f"ERROR: LLM non-streaming call failed: {e}")
        raise Exception(
            "LLM model failed to generate a response (non-streaming)."
        )


# Streaming call for main chat interactions
def call_Vela_stream(messages, temperature=0.4):
    """Sends a streaming request to the configured LLM endpoint."""
    model_name = APP_SETTINGS.get("llm_model_name", "").strip()
    _send_model = bool(model_name and model_name.lower() not in ("", "assistant", "none", "local"))
    payload = {
        "temperature": temperature,
        "max_tokens": TOKEN_SETTINGS.get("llm_max_tokens", 1024),
        "stream": True,
        "top_p": SAMPLER_SETTINGS.get("top_p", 0.9),
        "top_k": SAMPLER_SETTINGS.get("top_k", 40),
        "min_p": SAMPLER_SETTINGS.get("min_p", 0.05),
        "repetition_penalty": SAMPLER_SETTINGS.get("repetition_penalty", 1.1),
        "messages": messages  # messages always last — static prefix caches cleanly
    }
    if _send_model:
        payload["model"] = model_name
    # --- OPENROUTER EXTENSION ---
    if APP_SETTINGS.get("backend_mode") == "openrouter":
        if APP_SETTINGS.get("reasoning_enabled"):
            payload["reasoning"] = {"enabled": True}
        # OpenRouter isolated sampler lane.
        # top_k overridden to 0 (OR's default = disabled) — Kobold default is 40 which would
        # silently activate top-k filtering on OR models that expect it off.
        # min_p and repetition_penalty are kept — OR forwards them to providers that support them.
        payload["max_tokens"]  = APP_SETTINGS.get("openrouter_max_tokens", 1024)
        payload["temperature"] = APP_SETTINGS.get("openrouter_temperature", 0.7)
        payload["top_p"]       = APP_SETTINGS.get("openrouter_top_p", 0.9)
        payload["top_k"]       = 0  # OR default — disabled, let top_p + min_p do the work
    # --- END OPENROUTER EXTENSION ---
    headers = {"Content-Type": "application/json"}
    api_key = APP_SETTINGS.get("llm_api_key", "").strip()
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    endpoint = APP_SETTINGS.get("llm_api_endpoint", "http://127.0.0.1:5001/v1/chat/completions")
    try:
        response = requests.post(
            endpoint,
            headers=headers,
            json=payload,
            stream=True,  # IMPORTANT: Stream response from requests
            timeout=2000
        )
        response.raise_for_status()
        _in_reasoning = [False]  # mutable flag: True while inside a reasoning block
        for chunk in response.iter_lines():
            if chunk:
                # MODIFICATION: Check for [DONE] before trying to parse JSON
                decoded_chunk = chunk.decode('utf-8').lstrip('data: ').strip()
                if not decoded_chunk or decoded_chunk == '[DONE]':
                    continue # Ignore empty lines and the final DONE message
                try:
                    chunk_data = json.loads(decoded_chunk)
                    if ("choices" in chunk_data and
                            len(chunk_data["choices"]) > 0):
                        delta = chunk_data["choices"][0]["delta"]
                        # --- OPENROUTER EXTENSION: reasoning passthrough ---
                        if "reasoning" in delta and delta["reasoning"]:
                            if not _in_reasoning[0]:
                                _in_reasoning[0] = True
                                yield "<think>"
                            yield delta["reasoning"]
                        elif "content" in delta and delta["content"]:
                            if _in_reasoning[0]:
                                _in_reasoning[0] = False
                                yield "</think>"
                            yield delta["content"]
                        # --- END OPENROUTER EXTENSION ---
                except json.JSONDecodeError:
                    # This will now only catch actual malformed JSON, not the [DONE] message
                    print(f"WARNING: Could not decode JSON chunk: {decoded_chunk}")
                    continue
    except requests.exceptions.RequestException as e:
        print(f"ERROR: LLM streaming API call failed: {e}")
        raise


# Generator function that streams LLM responses
def _get_llm_response_stream(messages_to_send_to_llm, temperature=None):
    """Streams LLM response. Caller-supplied temperature takes priority; falls back to global setting."""
    try:
        # Use caller-supplied temperature when given (e.g. 0.5 for agentic turn 2+, 0.2 for image).
        # Only fall back to the global chat_stream setting when no override was passed.
        temp = temperature if temperature is not None else TEMPERATURE_SETTINGS.get("chat_stream", 0.4)
        for chunk in call_Vela_stream(messages_to_send_to_llm, temp):
            yield chunk
    except Exception as e:
        print(f"ERROR: Error during LLM streaming: {e}")
        yield f"ERROR: An error occurred: {e}"


def llm_summarize_text(text_to_summarize, max_length=600):
    """
    Uses the LLM to summarize a given piece of text if it's too long.
    """
    if not text_to_summarize:
        return ""

    # Use token count to decide if summarization is needed
    token_count = count_tokens(text_to_summarize)
    # Summarize if the text is ~25% longer than the target length
    if token_count < max_length * 1.25:
        print(
            f"DEBUG: Intro text is short enough ({token_count} tokens), "
            "skipping summarization."
        )
        return text_to_summarize

    print(
        f"DEBUG: Intro text is long ({token_count} tokens), attempting "
        f"summarization to ~{max_length} words."
    )
    messages_for_llm = [
        get_system_prompt()[0],  # System prompt for context
        {
            "role": "system",
            "content": (
                "You only Summarize text. no Meta comments like heres the summary, Heres the summarized text just output the summarized text "
                f"into a concise summary of about {max_length} words. "
                "Capture the most essential points, retain the original. "
                "tone, and preserve key details. only response the summarized text."
            )
        },
        {"role": "user", "content": text_to_summarize}
    ]
    try:
        start_time = time.time()
        # MODIFIED: Use temperature from global settings
        temp = TEMPERATURE_SETTINGS.get("summarization", 0.1)
        summary = _get_llm_response_non_stream(
            messages_for_llm, temperature=temp
        )
        end_time = time.time()
        print(
            "DEBUG: Text summarization LLM call took "
            f"{end_time - start_time:.4f} seconds."
        )
        return summary.strip()
    except Exception as e:
        print(f"ERROR: LLM Summarization failed: {e}")
        # Fallback to simple truncation if summarization fails
        return ' '.join(text_to_summarize.split()[:max_length]) + '...'


# --- KeyBERT-based session namer (no LLM call, uses existing EMBEDDING_MODEL) ---

_KEYBERT_MODEL = None  # Lazy-loaded on first rename, wraps the active EMBEDDING_MODEL
_KEYBERT_LOCK = threading.Lock()

def _get_keybert_model():
    """
    Lazy-loads KeyBERT wrapping the already-loaded EMBEDDING_MODEL.
    Re-initializes automatically after a model switch (EMBEDDING_MODEL changes).
    """
    global _KEYBERT_MODEL
    # If the inner model no longer matches the active one, rebuild
    if _KEYBERT_MODEL is not None and _KEYBERT_MODEL.model != EMBEDDING_MODEL:
        _KEYBERT_MODEL = None
    if _KEYBERT_MODEL is None:
        with _KEYBERT_LOCK:
            if _KEYBERT_MODEL is None:
                try:
                    from keybert import KeyBERT
                    print("DEBUG (Session Namer): Initializing KeyBERT with active EMBEDDING_MODEL (one-time)...")
                    _KEYBERT_MODEL = KeyBERT(model=EMBEDDING_MODEL)
                except ImportError:
                    print("WARNING (Session Namer): keybert not installed. Run: pip install keybert")
                    _KEYBERT_MODEL = None
    return _KEYBERT_MODEL


def llm_generate_session_name(first_message: str) -> str:
    """
    Names a session using KeyBERT semantic extraction — finds the most
    descriptive 2-3 word phrase already in the message itself.
    Uses the currently active EMBEDDING_MODEL, no LLM call needed.
    """
    try:
        kw_model = _get_keybert_model()
        if kw_model is None:
            raise RuntimeError("KeyBERT unavailable")

        keywords = kw_model.extract_keywords(
            first_message[:512],
            keyphrase_ngram_range=(1, 3),
            stop_words='english',
            top_n=1
        )

        if not keywords:
            raise ValueError("No keywords extracted")

        name = keywords[0][0].title()
        score = keywords[0][1]

        sanitized_name = re.sub(r'[^\w\s-]', '', name).strip()
        sanitized_name = re.sub(r'\s+', '_', sanitized_name)[:50]
        if not sanitized_name:
            raise ValueError("Empty sanitized name")

        print(f"DEBUG (Session Namer): '{first_message[:60]}' → '{name}' (score: {score:.3f})")
        return sanitized_name

    except Exception as e:
        print(f"ERROR: KeyBERT session name failed: {e}")
        return f"Chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


# --- Routes ---

# Route for JavaScript
@app.route('/script.js')
def serve_script():
    return send_from_directory('templates', 'script.js')

# 👇 ADD THIS NEW ROUTE FOR CSS 👇
@app.route('/style.css')
def serve_css():
    return send_from_directory('templates', 'style.css')

# ... rest of your code ...


@app.route('/save_image', methods=['POST'])
def save_image_local():
    try:
        data = request.json
        image_data = data.get('image')  # Expecting a base64 string

        if not image_data:
            return jsonify({'error': 'No image data provided'}), 400

        # Remove header if present (e.g., "data:image/png;base64,")
        if "base64," in image_data:
            image_data = image_data.split("base64,")[1]

        # Create a unique filename (timestamp + uuid)
        filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}.png"
        filepath = os.path.join(UPLOAD_FOLDER, filename)

        # Write the file to disk
        with open(filepath, "wb") as f:
            f.write(base64.b64decode(image_data))

        print(f"Image saved locally at: {filepath}")
        
        # Return the path/filename so the UI knows where it is
        return jsonify({'status': 'success', 'filename': filename, 'filepath': filepath})

    except Exception as e:
        print(f"Error saving image: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/get_image/<filename>')
def get_image(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route("/")
def home():
    return render_template("index.html")




# --- NEW: Silent Intro Injection Route ---
@app.route("/inject_silent_intro", methods=["POST"])
def inject_silent_intro_route():
    """
    Injects a silent, persistent 'memory' message into the chat memory.
    If the memory is long, it's summarized before injection based on user pref.
    """
    data = request.json
    intro_text = data.get("intro_text", "").strip()
    # Get the max_length from the request, with a default value of 200
    max_length = data.get("max_length", 200)
    session_id = get_current_session_id()

    if not intro_text:
        return jsonify({"error": "No intro_text provided."}), 400

    # Summarize the intro text if it's long, using the provided max_length
    summarized_intro = llm_summarize_text(intro_text, max_length=max_length)

    # Format the message with the special tag
    formatted_intro = f"system content: {summarized_intro}"

    # --- MODIFICATION: Insert memory at the beginning ---
    memory = load_memory()
    # This treats the injected memory as background context, just like files.
    memory.insert(0, {"role": "system", "content": formatted_intro, "silent": True})
    save_memory(memory)
    # --- END MODIFICATION ---

    print(
        f"DEBUG: Memory injected and summarized to "
        f"~{max_length} words: '{summarized_intro[:70]}...'"
    )

    # --- CHUNKING DLC: Index silent intro immediately on injection ---
    if RESONANCE_SETTINGS.get("chunking_enabled", False) and \
       RESONANCE_SETTINGS.get("chunk_index_system_immediately", True):
        try:
            sid = get_current_session_id()
            source_label = f"silent_intro:{hash(intro_text) & 0xFFFFFF}"
            _index_system_content_immediately(
                summarized_intro, "system", source_label, sid, pinned=False
            )
        except Exception as _e:
            print(f"DEBUG (ChunkDLC): Failed to index silent intro: {_e}")
    # --- END CHUNKING DLC ---

    return jsonify({"message": "System memory injected and summarized."})


@app.route("/get_silent_intros", methods=["GET"])
def get_silent_intros():
    """Gets silent memory messages that are currently in the active sliding window.

    Tier 1 (plain silent intros / injected facts) are always preserved by
    ghost_memory_if_needed so they always pass the active_ids check.
    Tier 2 hollows that happen to share role=system + silent=True (e.g. _recall_result,
    _raw_rag blocks) are correctly hidden once they slide out of the window.
    """
    memory = load_memory()
    active_memory = ghost_memory_if_needed(memory)
    active_ids = {id(m) for m in active_memory}
    intros = []
    for i, m in enumerate(memory):
        if m.get("role") == "system" and m.get("silent") and id(m) in active_ids:
            # FIX BUG 19: prefix stored as "system content: X" (space after colon)
            # so strip the full prefix including the space to avoid a leading space in output
            intro_text = m["content"].replace("system content: ", "", 1)
            # Fallback: also strip the no-space variant for any legacy entries
            intro_text = intro_text.replace("system content:", "", 1).strip()
            # Return the text and its index for easy deletion
            intros.append({"text": intro_text, "index": i})
    return jsonify({"intros": intros})


@app.route("/delete_silent_intro", methods=["POST"])
def delete_silent_intro():
    """Deletes a specific silent memory message from memory by its index."""
    data = request.json
    index_to_delete = data.get("index")

    if index_to_delete is None:
        return jsonify({"error": "Index to delete is required."}), 400

    memory = load_memory()

    try:
        index_to_delete = int(index_to_delete)
        if 0 <= index_to_delete < len(memory):
            # Security check: ensure the message at the index is a memory
            message_to_delete = memory[index_to_delete]
            if (message_to_delete.get("role") == "system" and
                message_to_delete.get("silent")):

                del memory[index_to_delete]
                save_memory(memory)
                print(
                    "DEBUG: Memory at index "
                    f"{index_to_delete} deleted."
                )
                return jsonify({"message": "System memory deleted."})
            else:
                return jsonify({
                    "error": "Message at specified index is not a "
                             "deletable memory."
                }), 400
        else:
            return jsonify({"error": "Index out of bounds."}), 400
    except (ValueError, TypeError):
        return jsonify({"error": "Invalid index format."}), 400

@app.route("/get_search_results", methods=["GET"])
def get_search_results():
    """Returns persisted search result blocks that are currently in the active sliding window.
    
    Uses ghost_memory_if_needed to determine which hollows are alive vs. ghosted out.
    Indices returned are raw memory positions so delete_search_result still works unchanged.
    """
    memory = load_memory()
    active_memory = ghost_memory_if_needed(memory)
    # Build a set of object ids from the active window for O(1) membership checks.
    # ghost_memory_if_needed returns references to the same objects (no copies),
    # so identity comparison is safe and exact.
    active_ids = {id(m) for m in active_memory}
    results = []
    for i, m in enumerate(memory):
        if m.get("_search_result") and id(m) in active_ids:
            results.append({
                "index": i,
                "content": m.get("content", ""),
                "pinned": m.get("_pinned", False)
            })
    return jsonify({"results": results})


@app.route("/delete_search_result", methods=["POST"])
def delete_search_result():
    """Deletes a persisted search result block from memory by index."""
    data = request.json
    index_to_delete = data.get("index")
    if index_to_delete is None:
        return jsonify({"error": "index is required."}), 400
    memory = load_memory()
    try:
        index_to_delete = int(index_to_delete)
        if 0 <= index_to_delete < len(memory):
            if memory[index_to_delete].get("_search_result"):
                del memory[index_to_delete]
                save_memory(memory)
                return jsonify({"message": "Search result deleted."})
            else:
                return jsonify({"error": "Entry is not a search result."}), 400
        else:
            return jsonify({"error": "Index out of bounds."}), 400
    except (ValueError, TypeError):
        return jsonify({"error": "Invalid index format."}), 400


@app.route("/edit_silent_intro", methods=["POST"])
def edit_silent_intro():
    """
    Edits the content of a specific silent memory message by its index.
    """
    data = request.json
    index_to_edit = data.get("index")
    new_content = data.get("new_content")

    if index_to_edit is None or not new_content:
        return jsonify({"error": "Index and new_content are required."}), 400

    memory = load_memory()

    try:
        index_to_edit = int(index_to_edit)
        if 0 <= index_to_edit < len(memory):
            # Security check: ensure the message at the index is actually a memory
            message_to_edit = memory[index_to_edit]
            
            if (message_to_edit.get("role") == "system" and
                message_to_edit.get("silent")):

                # FIX BUG 20: Re-apply the "system content: " prefix that inject_silent_intro
                # uses. Without this, edited entries lose the prefix, causing get_silent_intros
                # to return the raw content instead of cleanly stripping it.
                memory[index_to_edit]["content"] = f"system content: {new_content.strip()}"
                save_memory(memory)
                print(f"DEBUG: Memory at index {index_to_edit} updated.")
                return jsonify({"message": "Memory updated successfully."})
            else:
                return jsonify({
                    "error": "Message at specified index is not an editable memory."
                }), 400
        else:
            return jsonify({"error": "Index out of bounds."}), 400
    except (ValueError, TypeError):
        return jsonify({"error": "Invalid index format."}), 400


# --- NEW: Visible Intro Injection Route ---
@app.route("/inject_visible_intro", methods=["POST"])
def inject_visible_intro_route():
    """
    Injects a VISIBLE 'assistant' message at the beginning of the chat memory.
    This is intended to look like the AI started the conversation.
    """
    data = request.json
    intro_text = data.get("content", "").strip()

    if not intro_text:
        return jsonify({"error": "No content provided."}), 400

    # Get the current assistant's name to properly attribute the message
    # We call get_system_prompt() to ensure the global name is up-to-date
    _, assistant_name = get_system_prompt()
    
    # Load the current memory
    memory = load_memory()

    # Create the visible assistant message payload
    # Note: "silent" is "False" by default, so we don't need to add it.
    message_payload = {
        "role": "assistant",
        "name": assistant_name,
        "content": intro_text
    }

    # Insert this message at the very beginning of the chat history
    memory.insert(0, message_payload)

    # Save the modified memory
    save_memory(memory)

    print(
        f"DEBUG: Visible intro injected at the beginning of memory: '{intro_text[:70]}...'"
    )

    # --- CHUNKING DLC: Index visible intro immediately on injection ---
    if RESONANCE_SETTINGS.get("chunking_enabled", False) and \
       RESONANCE_SETTINGS.get("chunk_index_system_immediately", True):
        try:
            sid = get_current_session_id()
            source_label = f"visible_intro:{hash(intro_text) & 0xFFFFFF}"
            _index_system_content_immediately(
                intro_text, "assistant", source_label, sid, pinned=False
            )
        except Exception as _e:
            print(f"DEBUG (ChunkDLC): Failed to index visible intro: {_e}")
    # --- END CHUNKING DLC ---

    return jsonify({"message": "Visible intro injected successfully."})


# --- NEW: Endpoints for Managing Visible Intros ---

@app.route("/get_visible_intros", methods=["GET"])
def get_visible_intros():
    """
    Gets all visible assistant intros *before* the first user message.
    """
    memory = load_memory()
    intros = []
    for i, m in enumerate(memory):
        if m.get("role") == "user" and not m.get("silent"):
            # Stop as soon as we hit the first real user message
            break
        
        if m.get("role") == "assistant" and not m.get("silent"):
            intros.append({
                "content": m.get("content", ""),
                "name": m.get("name", _current_active_character_name),
                "index": i  # Return the actual memory index
            })
            
    return jsonify({"intros": intros})


@app.route("/update_visible_intro", methods=["POST"])
def update_visible_intro():
    """
    Updates the content of a specific visible intro message by its index.
    """
    data = request.json
    index_to_update = data.get("index")
    new_content = data.get("content", "").strip()

    if index_to_update is None or not new_content:
        return jsonify({"error": "Index and new content are required."}), 400

    memory = load_memory()

    try:
        index_to_update = int(index_to_update)
        if not (0 <= index_to_update < len(memory)):
            return jsonify({"error": "Index out of bounds."}), 400
        
        # Security check: Ensure the message is a visible assistant intro
        message_to_update = memory[index_to_update]
        if (message_to_update.get("role") == "assistant" and
            not message_to_update.get("silent")):
            
            memory[index_to_update]["content"] = new_content
            save_memory(memory)
            print(f"DEBUG: Visible intro at index {index_to_update} updated.")
            return jsonify({"message": "Visible intro updated."})
        else:
            return jsonify({
                "error": "Message at specified index is not an editable visible intro."
            }), 400
            
    except (ValueError, TypeError):
        return jsonify({"error": "Invalid index format."}), 400


@app.route("/delete_visible_intro", methods=["POST"])
def delete_visible_intro():
    """
    Deletes a specific visible intro message by its index.
    """
    data = request.json
    index_to_delete = data.get("index")

    if index_to_delete is None:
        return jsonify({"error": "Index to delete is required."}), 400

    memory = load_memory()

    try:
        index_to_delete = int(index_to_delete)
        if not (0 <= index_to_delete < len(memory)):
            return jsonify({"error": "Index out of bounds."}), 400

        # Security check: Ensure the message is a visible assistant intro
        message_to_delete = memory[index_to_delete]
        if (message_to_delete.get("role") == "assistant" and
            not message_to_delete.get("silent")):

            del memory[index_to_delete]
            save_memory(memory)
            print(f"DEBUG: Visible intro at index {index_to_delete} deleted.")
            return jsonify({"message": "Visible intro deleted."})
        else:
            return jsonify({
                "error": "Message at specified index is not a deletable visible intro."
            }), 400
            
    except (ValueError, TypeError):
        return jsonify({"error": "Invalid index format."}), 400

# --- END: Endpoints for Managing Visible Intros ---


def _always_index_single_message(msg, session_id):
    """
    Immediately indexes a single message into the FAISS index.
    Called when 'always_index_messages' is enabled, regardless of quota or window position.

    INCREMENTAL DESIGN (mirrors quota system in check_and_update_faiss_index):
    - Loads existing index + metadata
    - Deduplicates against already-indexed content (same set quota uses)
    - Appends ONE new vector — never rebuilds the whole index
    - Does NOT trigger rebuild_global_index() — global sync is handled
      lazily by check_and_update_faiss_index on the same turn, just like quota does.
      This prevents a full global rebuild firing on every single message.

    Skips: silent messages, system messages, empty content, duplicates,
           dimension/model mismatches (falls back to fresh index).
    """
    use_permanent_index = RESONANCE_SETTINGS.get("faiss_permanent_indexing", False)
    if not use_permanent_index:
        print("DEBUG (Always Index): Skipped — permanent indexing is OFF.")
        return

    content = msg.get("content", "")
    role = msg.get("role", "")

    # Skip silent, system, or empty messages (applies to both paths below)
    if not content or msg.get("silent") or role == "system":
        return
    if isinstance(content, list):
        content = " ".join(p.get("text", "") for p in content if p.get("type") == "text")
    if not content.strip():
        return

    # --- GLOBAL / LOCAL ROUTING ---
    # Global index builds directly from chat_memory.json — no local FAISS write needed.
    # If global is ON  → sync global only, skip local entirely (local is never read anyway).
    # If global is OFF → fall through to local write path below.
    # Switching global OFF mid-session: local index won't exist; check_and_update_faiss_index
    # detects the missing index and auto-rebuilds it from scratch on the next trigger.
    if RESONANCE_SETTINGS.get("global_memory_enabled", False):
        rebuild_global_index()
        print(f"DEBUG (Always Index): Global ON — synced global index for [{role}] message.")
        return
    # --- END GLOBAL ROUTING ---

    index_filepath = get_faiss_index_filepath(session_id)
    metadata_filepath = get_faiss_metadata_filepath(session_id)

    # --- Load existing index + metadata (same pattern as quota) ---
    existing_index = None
    indexed_messages, index_model = [], None

    if os.path.exists(index_filepath) and os.path.exists(metadata_filepath):
        try:
            temp_index = _safe_faiss_read(index_filepath)
            indexed_messages, index_model = load_faiss_metadata(metadata_filepath)
            current_model = RESONANCE_SETTINGS.get("embedding_model", "nomic")

            if temp_index.d != EMBEDDING_DIM:
                print(f"DEBUG (Always Index): Dimension mismatch ({temp_index.d} vs {EMBEDDING_DIM}). Starting fresh.")
            elif index_model is not None and index_model != current_model:
                print(f"DEBUG (Always Index): Model mismatch ('{index_model}' vs '{current_model}'). Starting fresh.")
            else:
                existing_index = temp_index
        except Exception as _load_exc:
            print(f"DEBUG (Always Index): Could not load existing index ({_load_exc}). Starting fresh.")

    if existing_index is None:
        existing_index = create_faiss_index(EMBEDDING_DIM)
        indexed_messages = []

    # --- Deduplicate using the SAME content set quota uses ---
    indexed_contents = {m.get("content", "") for m in indexed_messages}
    if content in indexed_contents:
        print("DEBUG (Always Index): Skipped duplicate — already indexed.")
        return

    # --- CHUNKING DLC: chunk before encode if enabled ---
    if RESONANCE_SETTINGS.get("chunking_enabled", False):
        chunks = chunk_text(content)
        # Filter out already-indexed chunks
        new_chunks = [c for c in chunks if c not in indexed_contents]
        if not new_chunks:
            print("DEBUG (Always Index + ChunkDLC): All chunks already indexed.")
            return
        total = len(chunks)
        source_label = f"msg:{role}:{hash(content) & 0xFFFFFF}"
        try:
            vecs    = embed_documents(new_chunks, show_progress_bar=False)
            vecs_np = np.array(vecs, dtype="float32")
            vecs_np = prepare_vectors(vecs_np)
        except Exception as _emb_exc:
            print(f"DEBUG (Always Index + ChunkDLC): Embedding failed ({_emb_exc}). Skipping.")
            return
        existing_index.add(vecs_np)
        with _faiss_write_lock:
            faiss.write_index(existing_index, index_filepath)
        for idx, chunk in enumerate(new_chunks):
            indexed_messages.append({
                "role":          role,
                "name":          msg.get("name", ""),
                "content":       chunk,
                "chunk_source":  source_label,
                "chunk_index":   idx,
                "chunk_total":   total,
                "pinned":        False,
                "timestamp":     msg.get("timestamp", ""),
                "always_indexed": True
            })
        save_faiss_metadata(metadata_filepath, indexed_messages)
        print(f"DEBUG (Always Index + ChunkDLC): [{role}] → {len(new_chunks)} chunk(s) indexed. "
              f"Total vectors: {existing_index.ntotal}")
        return
    # --- END CHUNKING DLC ---

    # --- Encode and append (no full rebuild, just .add() like quota) ---
    try:
        raw = embed_documents([content], show_progress_bar=False)
        vec = np.array(raw, dtype="float32")
        vec = prepare_vectors(vec)
    except Exception as _emb_exc:
        print(f"DEBUG (Always Index): Embedding failed, skipping. ({_emb_exc})")
        return

    existing_index.add(vec)
    with _faiss_write_lock:
        faiss.write_index(existing_index, index_filepath)

    # Append to metadata (same structure quota uses, just tagged always_indexed)
    indexed_messages.append({
        "role": role,
        "name": msg.get("name", ""),
        "content": content,
        "timestamp": msg.get("timestamp", ""),
        "always_indexed": True
    })
    save_faiss_metadata(metadata_filepath, indexed_messages)

    print(f"DEBUG (Always Index): Appended [{role}] message ({len(content)} chars) → "
          f"{existing_index.ntotal} total vectors in {index_filepath}")


# --- NEW: AUTOMATIC FAISS INDEXING FUNCTION ---
def check_and_update_faiss_index(memory, active_memory, session_id):
    """
    Automatically checks if the Faiss index needs to be updated and does so.
    This runs independently of memory recall triggers.
    """
    use_permanent_index = RESONANCE_SETTINGS.get("faiss_permanent_indexing", False)
    if not use_permanent_index:
        return # Feature is disabled
    
    if not session_id:
        print("ERROR (Faiss): session_id is required for automatic indexing. Skipping.")
        return

    # Always Index and Quota are independent — different jobs, different shifts.
    # Always Index handles every message immediately as it arrives (real-time, one-by-one).
    # Quota batches ghosted messages and fires only when the threshold is hit.
    # IMPORTANT: We still let quota run its full tracking/counting logic even when
    # Always Index is ON — this keeps the dedup set accurate and makes switching
    # between modes seamless with no gaps or double-indexing.
    _always_index_on = RESONANCE_SETTINGS.get("always_index_messages", False)

    try:
        quota = RESONANCE_SETTINGS.get("faiss_indexing_quota", 50)
        
        # 1. Define file paths
        index_filepath = get_faiss_index_filepath(session_id)
        metadata_filepath = get_faiss_metadata_filepath(session_id)

        # 2. Load existing index and message data
        permanent_index = None
        indexed_messages = []
        
        # --- MODIFICATION: Check for Dimension/Model Mismatch ---
        # If the existing index uses a different dimension (e.g. 384) than our current model (768),
        # we must blow it away and start fresh, or FAISS will crash.
        
        index_valid = False
        if os.path.exists(index_filepath) and os.path.exists(metadata_filepath):
            try:
                temp_index = _safe_faiss_read(index_filepath)
                indexed_messages, index_model = load_faiss_metadata(metadata_filepath)
                current_model = RESONANCE_SETTINGS.get("embedding_model", "nomic")

                if temp_index.d != EMBEDDING_DIM:
                    print(f"DEBUG (Faiss): Dimension mismatch! Index: {temp_index.d}, Model: {EMBEDDING_DIM}. Resetting.")
                elif index_model is not None and index_model != current_model:
                    print(f"DEBUG (Faiss): Model mismatch! Index built with '{index_model}', current is '{current_model}'. Resetting.")
                else:
                    permanent_index = temp_index
                    index_valid = True
                    print(f"DEBUG (Faiss): Loaded existing index (Dim: {temp_index.d}, Model: {index_model or 'unknown'}).")
            except Exception as e:
                print(f"ERROR (Faiss): Failed to load permanent index: {e}. Resetting.")
        
        if not index_valid:
            # If invalid or mismatched, ensure we start with None so we rebuild from scratch
            permanent_index = None
            indexed_messages = []
            # We don't necessarily need to delete the files, writing over them later is fine.
        # --- END MODIFICATION ---
        
        # 3. Identify all ghosted messages
        active_message_ids = {id(msg) for msg in active_memory} if active_memory else set()
        all_ghosted_messages = [
            msg for msg in memory
            if id(msg) not in active_message_ids and
            msg.get("role") in ["user", "assistant"] and
            isinstance(msg.get("content"), str) and 
            msg.get("content", "").strip()
        ]

        # 4. Find *new* ghosted messages that are not already in our permanent index
        indexed_message_contents = {msg.get("content") for msg in indexed_messages}
        new_ghosted_messages = []
        for msg in all_ghosted_messages:
            if msg.get("content") not in indexed_message_contents:
                new_ghosted_messages.append(msg)

        max_messages = TOKEN_SETTINGS.get("max_chat_messages", 16)
        total_messages = len(memory)
        active_count = len(active_memory) if active_memory else 0
        ghosted_count = len(all_ghosted_messages)
        already_indexed = len(indexed_message_contents)
        # Count ALL unindexed conversation messages (ghosted + active window).
        # This catches messages that slipped through when Always Index was turned off
        # mid-session — they're still in the active window so all_ghosted_messages
        # misses them, but the quota counter should still reflect the real backlog.
        all_conversation_messages = [
            msg for msg in memory
            if msg.get("role") in ["user", "assistant"]
            and isinstance(msg.get("content"), str)
            and msg.get("content", "").strip()
        ]
        new_unindexed = len([
            msg for msg in all_conversation_messages
            if msg.get("content") not in indexed_message_contents
        ])
        print(f"DEBUG (Sliding Window): Total={total_messages} | Active(visible)={active_count}/{max_messages} | Ghosted={ghosted_count} | Already indexed={already_indexed} | New unindexed={new_unindexed}/{quota} until re-index")
        
        # 5. Decide whether to update the permanent index
        # TWO-WORKER HANDOFF:
        # Always Index ON  → indexes every message in real-time as it arrives.
        # Always Index OFF → quota counts the backlog, fires when threshold is hit.
        # Switching ON     → Always Index must flush any backlog the quota was tracking
        #                    (messages that accumulated while it was OFF) before resuming
        #                    real-time mode. Without this flush those messages are orphaned
        #                    — neither worker ever claims them.
        # Switching OFF    → quota picks up cleanly; dedup via indexed_message_contents
        #                    ensures no double-indexing of what Always Index already did.
        if _always_index_on:
            # --- GLOBAL / LOCAL ROUTING ---
            # If global is ON, all writes go to global (via rebuild_global_index).
            # Local index is not read or written in this mode — skip entirely.
            if RESONANCE_SETTINGS.get("global_memory_enabled", False):
                rebuild_global_index()
                print("DEBUG (Faiss): AUTO-INDEX: Always Index ON + Global ON — synced global index.")
                return
            # --- END GLOBAL ROUTING ---

            # Flush target: ALL unindexed conversation messages (active window + ghosted).
            # new_ghosted_messages only covers messages already out of the window, but
            # orphaned messages from the quota-OFF period may still be in the active window.
            messages_to_flush = [
                msg for msg in all_conversation_messages
                if msg.get("content") not in indexed_message_contents
            ]
            if messages_to_flush:
                # Flush backlog left by quota mode — index any messages that slipped
                # through while Always Index was OFF, then hand off to real-time mode.
                print(f"DEBUG (Faiss): AUTO-INDEX: Always Index ON — flushing {len(messages_to_flush)} message(s) left by quota mode.")
                start_time = time.time()

                if RESONANCE_SETTINGS.get("chunking_enabled", False):
                    all_chunk_entries = []
                    all_chunk_texts   = []
                    for msg in messages_to_flush:
                        raw = msg.get("content", "")
                        chunks = chunk_text(raw)
                        total  = len(chunks)
                        source_label = f"msg:{msg.get('role','')}:{hash(raw) & 0xFFFFFF}"
                        for idx, chunk in enumerate(chunks):
                            if not chunk.strip():
                                continue
                            all_chunk_texts.append(chunk)
                            entry = msg.copy()
                            entry["content"]      = chunk
                            entry["chunk_source"] = source_label
                            entry["chunk_index"]  = idx
                            entry["chunk_total"]  = total
                            entry["pinned"]       = False
                            all_chunk_entries.append(entry)
                    if all_chunk_entries:
                        flush_np = np.array(
                            embed_documents(all_chunk_texts, show_progress_bar=False),
                            dtype="float32"
                        )
                        flush_np = prepare_vectors(flush_np)
                        if permanent_index is None:
                            permanent_index = create_faiss_index(EMBEDDING_DIM)
                        permanent_index.add(flush_np)
                        indexed_messages.extend(all_chunk_entries)
                        with _faiss_write_lock:
                            faiss.write_index(permanent_index, index_filepath)
                        save_faiss_metadata(metadata_filepath, indexed_messages)
                        print(f"DEBUG (Faiss + ChunkDLC): Flush took {time.time() - start_time:.4f}s. "
                              f"Chunks: {len(all_chunk_entries)}, Total vectors: {permanent_index.ntotal}")
                        if RESONANCE_SETTINGS.get("global_memory_enabled", False):
                            rebuild_global_index()
                else:
                    flush_texts = [msg.get("content", "") for msg in messages_to_flush]
                    flush_np = np.array(
                        embed_documents(flush_texts, show_progress_bar=False),
                        dtype="float32"
                    ).astype('float32')
                    flush_np = prepare_vectors(flush_np)
                    if permanent_index is None:
                        permanent_index = create_faiss_index(EMBEDDING_DIM)
                    permanent_index.add(flush_np)
                    indexed_messages.extend(messages_to_flush)
                    with _faiss_write_lock:
                        faiss.write_index(permanent_index, index_filepath)
                    save_faiss_metadata(metadata_filepath, indexed_messages)
                    print(f"DEBUG (Faiss): Flush took {time.time() - start_time:.4f}s. Total vectors: {permanent_index.ntotal}")
                    if RESONANCE_SETTINGS.get("global_memory_enabled", False):
                        rebuild_global_index()
            else:
                print(f"DEBUG (Faiss): AUTO-INDEX: Always Index is ON — no backlog to flush, real-time mode active.")
            return

        if len(new_ghosted_messages) >= quota or (permanent_index is None and len(new_ghosted_messages) > 0):
            # --- GLOBAL / LOCAL ROUTING ---
            # Global ON: no local write needed — global builds from chat_memory.json directly.
            if RESONANCE_SETTINGS.get("global_memory_enabled", False):
                rebuild_global_index()
                print(f"DEBUG (Faiss): AUTO-INDEX: Quota met + Global ON — synced global index ({len(new_ghosted_messages)} new messages).")
                return
            # --- END GLOBAL ROUTING ---

            # --- QUOTA MET OR FRESH START: Update permanent (local) index ---
            print(f"DEBUG (Faiss): AUTO-INDEX: Processing {len(new_ghosted_messages)} messages to index.")
            start_time = time.time()

            # --- CHUNKING DLC: chunk each message before encoding ---
            if RESONANCE_SETTINGS.get("chunking_enabled", False):
                all_chunk_entries = []
                all_chunk_texts   = []
                for msg in new_ghosted_messages:
                    raw = msg.get("content", "")
                    chunks = chunk_text(raw)
                    total  = len(chunks)
                    source_label = f"msg:{msg.get('role','')}:{hash(raw) & 0xFFFFFF}"
                    for idx, chunk in enumerate(chunks):
                        if not chunk.strip():
                            continue
                        all_chunk_texts.append(chunk)
                        entry = msg.copy()
                        entry["content"]      = chunk
                        entry["chunk_source"] = source_label
                        entry["chunk_index"]  = idx
                        entry["chunk_total"]  = total
                        entry["pinned"]       = False
                        all_chunk_entries.append(entry)

                if all_chunk_entries:
                    new_embeddings_np = np.array(
                        embed_documents(all_chunk_texts, show_progress_bar=False),
                        dtype="float32"
                    )
                    new_embeddings_np = prepare_vectors(new_embeddings_np)
                    if permanent_index is None:
                        permanent_index = create_faiss_index(EMBEDDING_DIM)
                    permanent_index.add(new_embeddings_np)
                    indexed_messages.extend(all_chunk_entries)
                    with _faiss_write_lock:
                        faiss.write_index(permanent_index, index_filepath)
                    save_faiss_metadata(metadata_filepath, indexed_messages)
                    print(f"DEBUG (Faiss + ChunkDLC): AUTO-INDEX took {time.time() - start_time:.4f}s. "
                          f"Chunks: {len(all_chunk_entries)}, Total vectors: {permanent_index.ntotal}")
                    if RESONANCE_SETTINGS.get("global_memory_enabled", False):
                        rebuild_global_index()
                return  # DLC handled it, skip vanilla path below
            # --- END CHUNKING DLC ---

            texts_to_encode = [msg.get("content", "") for msg in new_ghosted_messages]
            new_embeddings = embed_documents(texts_to_encode, show_progress_bar=False)
            new_embeddings_np = np.array(new_embeddings).astype('float32')
            new_embeddings_np = prepare_vectors(new_embeddings_np)

            if permanent_index is None:
                permanent_index = create_faiss_index(EMBEDDING_DIM)
            
            permanent_index.add(new_embeddings_np)
            indexed_messages.extend(new_ghosted_messages)
            
            with _faiss_write_lock:
                faiss.write_index(permanent_index, index_filepath)
            save_faiss_metadata(metadata_filepath, indexed_messages)
            
            print(f"DEBUG (Faiss): AUTO-INDEX: Permanent index update took {time.time() - start_time:.4f} seconds. Total vectors: {permanent_index.ntotal}")
            
            # Only rebuild global when local index actually changed
            if RESONANCE_SETTINGS.get("global_memory_enabled", False):
                rebuild_global_index()
        
        elif len(new_ghosted_messages) > 0:
            print(f"DEBUG (Faiss): AUTO-INDEX: Quota not met ({len(new_ghosted_messages)}/{quota}). Indexing skipped.")
        
        else:
            print("DEBUG (Faiss): AUTO-INDEX: No new ghosted messages to index.")

    except Exception as e:
        print(f"ERROR (Faiss): An error occurred during automatic index check: {e}")
        traceback.print_exc()
# --- END NEW FUNCTION ---


# (NLP recall intent detection removed — Raw RAG passive recall is the only recall method)

# --- PAIRED MEMORY HELPER ---
def _expand_to_exchange(matched_messages, source_pool, active_contents_set=None):
    """
    Paired Memory: expands each matched message into its full conversational exchange.

    For every hit:
      - user message   → keep it + grab the assistant message immediately after it
      - assistant msg  → grab the user message immediately before it + keep it

    Works on two kinds of source_pool:
      - A flat list of message dicts (ghosted_messages / temp search path)
      - A list of indexed metadata dicts (permanent FAISS metadata path)

    Deduplication: messages already visible in active_contents_set are skipped
    (same rule as the main recall loop).

    Returns a new list of messages in conversation order, deduplicated.
    """
    if not matched_messages or not source_pool:
        return matched_messages

    active_set = active_contents_set or set()

    # Build a content → index map for fast neighbour lookup
    content_to_idx = {}
    for idx, msg in enumerate(source_pool):
        c = msg.get("content", "")
        if c and c not in content_to_idx:
            content_to_idx[c] = idx

    expanded = []
    seen_content = set()

    def _add(msg):
        c = msg.get("content", "")
        if c in seen_content or c.strip() in active_set:
            return
        seen_content.add(c)
        expanded.append(msg)

    for hit in matched_messages:
        hit_content = hit.get("content", "")
        hit_role    = hit.get("role", "")
        hit_idx     = content_to_idx.get(hit_content)

        if hit_idx is None:
            # Can't find in pool — just keep the hit as-is
            _add(hit)
            continue

        if hit_role == "user":
            # Keep the user message, then try to grab the assistant reply after
            _add(source_pool[hit_idx])
            next_idx = hit_idx + 1
            while next_idx < len(source_pool):
                neighbour = source_pool[next_idx]
                n_role = neighbour.get("role", "")
                if n_role == "assistant":
                    _add(neighbour)
                    break
                elif n_role == "user":
                    # Another user message with no reply between — stop
                    break
                next_idx += 1

        elif hit_role == "assistant":
            # Try to grab the user message before, then keep the assistant message
            prev_idx = hit_idx - 1
            while prev_idx >= 0:
                neighbour = source_pool[prev_idx]
                n_role = neighbour.get("role", "")
                if n_role == "user":
                    _add(neighbour)
                    break
                elif n_role == "assistant":
                    # Another assistant message with no user before — stop
                    break
                prev_idx -= 1
            _add(source_pool[hit_idx])

        else:
            _add(source_pool[hit_idx])

    print(f"DEBUG (Paired Memory): Expanded {len(matched_messages)} hit(s) → "
          f"{len(expanded)} message(s) after pairing.")
    return expanded
# --- END PAIRED MEMORY HELPER ---


# --- FAISS-POWERED get_relevant_memories (Raw RAG Only) ---
def get_relevant_memories(user_message, memory, active_memory=None, session_id=None, force_search=False, raw_mode=True):
    """
    Retrieves relevant messages from ghosted chat memory using Faiss vector search.
    raw_mode=True  → score threshold gates results (passive Raw RAG path)
    raw_mode=False → threshold skipped, all FAISS hits returned (explicit [RECALL:] tool path)
    force_search   → return a "no results" string instead of None when nothing found
    """
    # FIX: Do NOT override raw_mode here. The RECALL tool explicitly passes raw_mode=False
    # to bypass the score threshold, which is intentional — it already asked for recall.
    # Hardcoding raw_mode=True here silently applied the threshold to explicit recalls too.
    print(f"DEBUG (Faiss): Running {'passive RAG' if raw_mode else 'explicit RECALL'} for: '{user_message[:50]}...'")

    latest_user_message = user_message.lower()

    # Check if the resonance (memory recall) feature is enabled at all
    if not TOKEN_SETTINGS.get("resonance_enabled", True):
        print("DEBUG (Faiss): Resonance is disabled in settings, skipping vector search.")
        return None

    if raw_mode:
        print("DEBUG (Faiss): Raw RAG mode — score threshold will gate results.")
    else:
        print("DEBUG (Faiss): Explicit RECALL mode — score threshold bypassed.")

    # --- STAGE 3 PRE-GATE: Intent check BEFORE any FAISS work ---
    # For explicit [RECALL:] tool calls (force_search=True), intent is already confirmed
    # by the LLM choosing to invoke the tool — skip the gate entirely.
    sanity_enabled = SANITY_CHECK_SETTINGS.get("sanity_check_enabled", False)
    intent_enabled = SANITY_CHECK_SETTINGS.get("sanity_check_intent_enabled", True)
    if sanity_enabled and intent_enabled and not force_search:
        intent_threshold = SANITY_CHECK_SETTINGS.get("sanity_check_intent_threshold", 0.40)
        has_intent, intent_score = check_recall_intent(user_message, intent_threshold)
        if not has_intent:
            print(f"DEBUG (Sanity Pre-Gate): No recall intent — skipping FAISS entirely (score: {intent_score:.3f})")
            return None  # Silent skip, same as no-results passive RAG
        print(f"DEBUG (Sanity Pre-Gate): Intent confirmed (score: {intent_score:.3f}) — proceeding to FAISS")
    # --- END STAGE 3 PRE-GATE ---

    use_permanent_index = RESONANCE_SETTINGS.get("faiss_permanent_indexing", False)
    if not use_permanent_index:
        return _get_relevant_memories_temporary(user_message, memory, active_memory)

    if not session_id:
        print("ERROR (Faiss): session_id is required for permanent indexing search. Skipping.")
        return None

    try:
        # Raw RAG always uses its own lean k limit
        k = RESONANCE_SETTINGS.get("raw_rag_max_recall", 2)
        
        # --- RERANKER EXPANSION: Retrieve more candidates for reranking ---
        reranker_enabled = RESONANCE_SETTINGS.get("reranker_enabled", False)
        if reranker_enabled:
            expansion_factor = RESONANCE_SETTINGS.get("reranker_expansion_factor", 3)
            retrieval_k = k * expansion_factor  # e.g., want 2 final → retrieve 6 candidates
            print(f"DEBUG (Reranker/Raw RAG): Expansion enabled - retrieving {retrieval_k} candidates (k={k} × factor={expansion_factor}).")
        elif sanity_enabled:
            buffer_mult = SANITY_CHECK_SETTINGS.get("sanity_check_buffer_multiplier", 2.0)
            retrieval_k = max(k, int(k * buffer_mult))
            print(f"DEBUG (Sanity Buffer/Raw RAG): Over-fetching {retrieval_k} candidates for Stage 3 (k={k} × buffer={buffer_mult}).")
        else:
            retrieval_k = k
        # --- END RERANKER EXPANSION ---
        
        use_global_memory = RESONANCE_SETTINGS.get("global_memory_enabled", False)

        # 2. Prepare Query Vector
        query_vector = embed_query([user_message], show_progress_bar=False)
        query_vector_np = np.array(query_vector).astype('float32')
        query_vector_np = prepare_vectors(query_vector_np)
        
        all_results = []

        # Build a set of content already visible in the active window —
        # FAISS must never re-surface messages the LLM can already see.
        active_contents_set = set()
        if active_memory:
            for msg in active_memory:
                c = msg.get("content")
                if isinstance(c, str) and c.strip():
                    active_contents_set.add(c.strip())

        global_messages = []  # FIX: pre-initialize so paired memory path never gets NameError
        if use_global_memory:
            # --- GLOBAL MODE: Search the single unified global index ---
            global_index_path = get_global_faiss_filepath()
            global_meta_path = get_global_faiss_metadata_filepath()

            if os.path.exists(global_index_path) and os.path.exists(global_meta_path):
                try:
                    global_index = _safe_faiss_read(global_index_path)
                    global_messages, index_model = load_faiss_metadata(global_meta_path)
                    current_model = RESONANCE_SETTINGS.get("embedding_model", "nomic")
                    if global_index.d == EMBEDDING_DIM and global_index.ntotal > 0 and (index_model is None or index_model == current_model):
                        search_k = min(retrieval_k, global_index.ntotal)
                        distances, indices = global_index.search(query_vector_np, search_k)
                        for dist, i in zip(distances[0], indices[0]):
                            if i != -1:
                                msg = global_messages[i]
                                # FIX: Skip if this message is already visible in active window
                                if msg.get("content", "").strip() in active_contents_set:
                                    print(f"DEBUG (Global Index): Skipping result already in active window.")
                                    continue
                                all_results.append((dist, msg))
                        print(f"DEBUG (Global Index): Search returned {len(all_results)} raw results (after active window filter).")
                    else:
                        print("DEBUG (Global Index): Index stale (dim or model mismatch). Triggering rebuild...")
                        rebuild_global_index()
                except Exception as e:
                    print(f"ERROR (Global Index): Search failed: {e}. Falling back to local.")
            else:
                print("DEBUG (Global Index): No global index found yet. Triggering rebuild...")
                rebuild_global_index()
        else:
            # --- LOCAL MODE: Search the current session's permanent index only ---
            index_filepath = get_faiss_index_filepath(session_id)
            metadata_filepath = get_faiss_metadata_filepath(session_id)

            if os.path.exists(index_filepath) and os.path.exists(metadata_filepath):
                try:
                    perm_index = _safe_faiss_read(index_filepath)
                    indexed_messages, index_model = load_faiss_metadata(metadata_filepath)
                    current_model = RESONANCE_SETTINGS.get("embedding_model", "nomic")
                    if perm_index.d == EMBEDDING_DIM and perm_index.ntotal > 0 and (index_model is None or index_model == current_model):
                        search_k_perm = min(retrieval_k, perm_index.ntotal)
                        distances, indices = perm_index.search(query_vector_np, search_k_perm)
                        for dist, i in zip(distances[0], indices[0]):
                            if i != -1:
                                msg_data = indexed_messages[i].copy()
                                # FIX: Skip if already visible in active window
                                if msg_data.get("content", "").strip() in active_contents_set:
                                    print(f"DEBUG (Local Index): Skipping result already in active window.")
                                    continue
                                msg_data['session_source'] = session_id
                                all_results.append((dist, msg_data))
                    else:
                        print(f"DEBUG (Faiss): Local index stale (dim or model mismatch). Will rebuild on next message.")
                except Exception as e:
                    print(f"ERROR (Faiss Search): {e}")

        # 4. Search Active/Temporary Un-indexed Messages (Always for current session)
        active_message_ids = {id(msg) for msg in active_memory} if active_memory else set()
        all_ghosted_messages = [
            msg for msg in memory
            if id(msg) not in active_message_ids and
            msg.get("role") in ["user", "assistant"] and
            isinstance(msg.get("content"), str) and 
            msg.get("content", "").strip() and 
            msg.get("content", "").lower() != latest_user_message
        ]

        # Don't re-search what's already in the index we just searched
        already_indexed_contents = {msg.get("content") for _, msg in all_results}
        messages_for_temp_search = [msg for msg in all_ghosted_messages if msg.get("content") not in already_indexed_contents]

        if messages_for_temp_search:
            temp_texts = [msg.get("content", "") for msg in messages_for_temp_search]
            temp_embeddings = embed_documents(temp_texts, show_progress_bar=False)
            temp_embeddings_np = np.array(temp_embeddings).astype('float32')
            temp_embeddings_np = prepare_vectors(temp_embeddings_np)
            
            temp_index = create_faiss_index(EMBEDDING_DIM)
            temp_index.add(temp_embeddings_np)
            
            search_k_temp = min(retrieval_k, temp_index.ntotal)
            distances, indices = temp_index.search(query_vector_np, search_k_temp)
            
            for dist, i in zip(distances[0], indices[0]):
                if i != -1:
                    msg_data = messages_for_temp_search[i].copy()
                    msg_data['session_source'] = 'Current Session (Recent)'
                    all_results.append((dist, msg_data))

        if not all_results:
            return "No relevant memories found." if force_search else None

        # 5. Sort ALL results by distance and take top k
        all_results.sort(key=lambda x: x[0])

        # --- RAW RAG: Score threshold gating ---
        # In raw_mode, results that don't clear the threshold are silently dropped.
        # This is the ONLY gatekeeper in raw mode, so tune it well!
        if raw_mode:
            raw_threshold = RESONANCE_SETTINGS.get("raw_rag_score_threshold", 1.0)
            metric = RESONANCE_SETTINGS.get("faiss_distance_metric", "l2")
            if metric == "cosine":
                # Cosine/IP: higher score = better match. Reject if score < threshold.
                # We store distances as raw IP scores here, so threshold is direct.
                all_results = [(dist, msg) for dist, msg in all_results if dist >= raw_threshold]
                print(f"DEBUG (Raw RAG): Score filter (cosine >= {raw_threshold}): {len(all_results)} results passed.")
            else:
                # L2: lower distance = better match. Reject if dist > threshold.
                all_results = [(dist, msg) for dist, msg in all_results if dist <= raw_threshold]
                print(f"DEBUG (Raw RAG): Score filter (L2 <= {raw_threshold}): {len(all_results)} results passed.")
            
            if not all_results:
                print("DEBUG (Raw RAG): All results filtered out by score threshold. Silent skip.")
                return None  # Silent — don't inject anything
        # --- END RAW RAG SCORE FILTER ---
        
        # ========================================
        # STAGE 2: SANITY CONTENT FILTER (cheap — runs BEFORE reranker to prune pool)
        # ========================================
        # Intent was already checked in the pre-gate above (or skipped for force_search).
        # Only run content similarity (Phase 2) here.
        # sanity_enabled was already read at the top of this function for the pre-gate.
        if sanity_enabled and all_results:
            print(f"DEBUG (Sanity/RAG): ENABLED - Running content pre-filter on {len(all_results)} candidates...")
            candidates = [msg for dist, msg in all_results]
            filtered = sanity_check_filter(user_message, candidates, skip_intent_check=True)
            # Rebuild (dist, msg) tuples — preserve original distances for passed candidates
            dist_map = {id(msg): dist for dist, msg in all_results}
            all_results = [(dist_map.get(id(msg), 0), msg) for msg in filtered]
            print(f"DEBUG (Sanity/RAG): Content pre-filter passed {len(all_results)} memories → feeding to Reranker")
        elif sanity_enabled and not all_results:
            print(f"DEBUG (Sanity/RAG): ENABLED but no candidates from previous stages - skipping")
        else:
            print(f"DEBUG (Sanity/RAG): DISABLED - Skipping content pre-filter")
        # --- END STAGE 2 ---

        # ========================================
        # STAGE 3: RERANKER — final authoritative scoring (expensive, runs on pruned pool)
        # ========================================
        reranker_enabled = RESONANCE_SETTINGS.get("reranker_enabled", False)
        if reranker_enabled and RERANKER_MODEL and all_results:
            print(f"DEBUG (Reranker/Raw RAG): ENABLED - Reranking {len(all_results)} candidates (post content-filter)...")
            # Convert (dist, msg) tuples to candidate dicts for reranker
            candidates = [{"content": msg.get("content", ""), "metadata": msg} for dist, msg in all_results]
            reranker_threshold = RESONANCE_SETTINGS.get("reranker_score_threshold", 0.0)
            batch_size = RESONANCE_SETTINGS.get("reranker_batch_size", 32)
            # expansion_factor already applied above when computing retrieval_k —
            # FAISS fetched k * expansion_factor candidates; reranker now trims back to k.
            reranked = rerank_candidates(
                query=user_message,
                candidates=candidates,
                top_k=k,
                score_threshold=reranker_threshold,
                batch_size=batch_size
            )
            # Convert back to (dist, msg) — use negative reranker score to maintain sort order
            all_results = [(-item.get("reranker_score", 0), item["metadata"]) for item in reranked]
            print(f"DEBUG (Reranker/Raw RAG): Stage 3 returned {len(all_results)} results after reranking.")
        elif reranker_enabled and not RERANKER_MODEL:
            print("WARNING (Reranker/Raw RAG): ENABLED but model not loaded - skipping reranking.")
        # --- END STAGE 3 ---
        
        final_matched_messages = []
        seen_content = set()
        for dist, msg in all_results:
            if msg.get("content") not in seen_content:
                final_matched_messages.append(msg)
                seen_content.add(msg.get("content"))
            if len(final_matched_messages) >= k:
                break

        # --- CHUNKING DLC: long_context reassembly ---
        if RESONANCE_SETTINGS.get("chunking_enabled", False) and \
           RESONANCE_SETTINGS.get("chunk_retrieval_mode", "precision") == "long_context":
            threshold = RESONANCE_SETTINGS.get("long_context_threshold", 2)
            pinned_reassemble = RESONANCE_SETTINGS.get("chunk_pinned_always_reassemble", True)

            # Group hits by chunk_source
            source_hits = {}
            non_chunked = []
            for msg in final_matched_messages:
                src = msg.get("chunk_source")
                if src:
                    source_hits.setdefault(src, []).append(msg)
                else:
                    non_chunked.append(msg)

            reassembled = []
            for src, hits in source_hits.items():
                is_pinned  = any(h.get("pinned") for h in hits)
                should_reassemble = (len(hits) >= threshold) or (is_pinned and pinned_reassemble)
                if should_reassemble:
                    # Load ALL chunks for this source from metadata and stitch in order
                    try:
                        meta_fp = get_faiss_metadata_filepath(session_id)
                        all_meta, _ = load_faiss_metadata(meta_fp)
                        src_chunks = sorted(
                            [m for m in all_meta if m.get("chunk_source") == src],
                            key=lambda m: m.get("chunk_index", 0)
                        )
                        if src_chunks:
                            # Build a synthetic merged message for the recall block
                            merged_content = "\n\n".join(c.get("content", "") for c in src_chunks)
                            synthetic = src_chunks[0].copy()
                            synthetic["content"] = merged_content
                            reassembled.append(synthetic)
                            print(f"DEBUG (ChunkDLC long_context): Reassembled '{src}' "
                                  f"({len(src_chunks)} chunks).")
                        else:
                            reassembled.extend(hits)
                    except Exception as _re:
                        print(f"DEBUG (ChunkDLC): Reassembly failed for '{src}': {_re}")
                        reassembled.extend(hits)
                else:
                    # Precision fallback: best scoring chunk only
                    reassembled.append(hits[0])

            final_matched_messages = non_chunked + reassembled
        # --- END CHUNKING DLC ---

        # --- PAIRED MEMORY: expand hits into full exchanges ---
        if RESONANCE_SETTINGS.get("paired_memory_enabled", False):
            # Build the full source pool: prefer the metadata list from the
            # index (covers ghosted messages). For global mode this is
            # global_messages, for local mode it's indexed_messages.
            # We fall back to the raw memory list when neither is available.
            try:
                if use_global_memory:
                    _pair_pool = global_messages  # set during global search above
                else:
                    _pair_meta_fp = get_faiss_metadata_filepath(session_id)
                    _pair_pool, _ = load_faiss_metadata(_pair_meta_fp)
                if not _pair_pool:
                    _pair_pool = [m for m in memory
                                  if m.get("role") in ("user", "assistant")
                                  and isinstance(m.get("content"), str)]
            except Exception:
                _pair_pool = [m for m in memory
                              if m.get("role") in ("user", "assistant")
                              and isinstance(m.get("content"), str)]
            final_matched_messages = _expand_to_exchange(
                final_matched_messages, _pair_pool, active_contents_set
            )
        # --- END PAIRED MEMORY ---

        reflection_lines = []
        char_limit = RESONANCE_SETTINGS.get("recalled_message_char_limit", 30000)
        
        # --- FIX: Robust Name Fetching for GLOBAL MEMORY ---
        user_name_setting = NAME_SETTINGS.get("user_name", "User")
        if not user_name_setting or not str(user_name_setting).strip():
            user_name_setting = "User"

        for msg in final_matched_messages:
            msg_content = msg.get("content", "")[:char_limit]
            
            display_name = "System"
            if msg.get("role") == "user":
                display_name = msg.get("name", user_name_setting)
                if not display_name or not str(display_name).strip():
                    display_name = "User"
            elif msg.get("role") == "assistant":
                display_name = msg.get("name", _current_active_character_name)
                if not display_name or not str(display_name).strip():
                    display_name = "Assistant"
            else:
                display_name = msg.get("name", msg.get('role', 'System').capitalize())
                if not display_name or not str(display_name).strip():
                    display_name = "System"
            
            source = msg.get('session_source', session_id)
            if source and source != session_id and source != 'Current Session (Recent)':
                source_tag = f" @Session: {source}"
            else:
                source_tag = ""

            ts = msg.get('timestamp')
            if ts:
                try:
                    dt = datetime.fromisoformat(ts)
                    ts_tag = f" | {dt.strftime('%b %d %Y, %I:%M %p')}"
                except Exception:
                    ts_tag = ""
            else:
                ts_tag = ""

            label = f"[{display_name}{source_tag}{ts_tag}]:"
            reflection_lines.append(f"{label}\n{msg_content}")
        # --- END FIX ---

        separator = "\n---\n"
        header = RESONANCE_SETTINGS.get("memory_block_header", "[THIS BLOCK HERE ARE THE RETRIEVED MEMORIES AND PAST CONVERSATIONS]")
        closer = RESONANCE_SETTINGS.get("memory_block_closer", "[/END OF RETRIEVED MEMORIES FROM PAST CONVERSATIONS]")
        reflection_content = f"{header}\n---\n" + separator.join(reflection_lines) + f"\n---\n{closer}"
        return reflection_content

    except Exception as e:
        print(f"ERROR (Faiss): {e}")
        traceback.print_exc()
        return "Error retrieving memories." if force_search else None

def _get_relevant_memories_temporary(user_message, memory, active_memory=None):
    """
    The original temporary index logic, now used as a fallback.
    """
    print("DEBUG (Faiss): Using temporary index method.")
    
    # Filter out active_memory from the search space
    active_message_ids = {id(msg) for msg in active_memory} if active_memory else set()
    ghosted_messages = [
        msg for msg in memory
        if id(msg) not in active_message_ids and
        msg.get("role") in ["user", "assistant"] and
        isinstance(msg.get("content"), str) and 
        msg.get("content", "").strip() and # Ensure content is not empty
        msg.get("content", "").lower() != user_message.lower()
    ]
    
    if len(ghosted_messages) < 1:
        print("DEBUG (Faiss): No ghosted messages to search.")
        return None

    print(f"DEBUG (Faiss): Searching {len(ghosted_messages)} ghosted messages (temporary).")

    try:
        # 1. Prepare texts and encode them
        texts_to_encode = [msg.get("content", "") for msg in ghosted_messages]
        start_time = time.time()
        embeddings = embed_documents(texts_to_encode, show_progress_bar=False)
        embeddings_np = np.array(embeddings).astype('float32')
        embeddings_np = prepare_vectors(embeddings_np)
        print(f"DEBUG (Faiss): Encoding took {time.time() - start_time:.4f} seconds.")

        # 2. Build the Faiss index
        index = create_faiss_index(EMBEDDING_DIM)
        index.add(embeddings_np)
        print(f"DEBUG (Faiss): Built temporary index with {index.ntotal} vectors.")

        # 3. Prepare the query vector
        query_vector = embed_query([user_message], show_progress_bar=False)
        query_vector_np = np.array(query_vector).astype('float32')
        query_vector_np = prepare_vectors(query_vector_np)

        # 4. Search the index
        # Expand retrieval count if reranker or sanity buffer is active
        k_final = RESONANCE_SETTINGS.get("raw_rag_max_recall", 2)
        _reranker_on = RESONANCE_SETTINGS.get("reranker_enabled", False)
        _sanity_on   = SANITY_CHECK_SETTINGS.get("sanity_check_enabled", False)
        if _reranker_on:
            _expansion = RESONANCE_SETTINGS.get("reranker_expansion_factor", 3)
            k = min(k_final * _expansion, index.ntotal)
            print(f"DEBUG (Reranker/Temp): Over-fetching {k} candidates for reranking (final={k_final}).")
        elif _sanity_on:
            _buffer = SANITY_CHECK_SETTINGS.get("sanity_check_buffer_multiplier", 2.0)
            k = min(max(k_final, int(k_final * _buffer)), index.ntotal)
            print(f"DEBUG (Sanity Buffer/Temp): Over-fetching {k} candidates for Stage 3 (final={k_final}).")
        else:
            k = min(k_final, index.ntotal)
        
        print(f"DEBUG (Faiss): Searching for top {k} results (temporary)...")
        distances, indices = index.search(query_vector_np, k)
        
        # 5. Map indices back to messages
        matched_messages = []
        for i in indices[0]:
            if i != -1:
                matched_messages.append(ghosted_messages[i])
        
        if not matched_messages:
            return None

        # --- STAGE 2: SANITY CONTENT PRE-FILTER (cheap — runs BEFORE reranker to prune pool) ---
        # Intent was already pre-gated in get_relevant_memories() before this function
        # was called — always skip Phase 1 here and run content similarity only.
        if _sanity_on and matched_messages:
            print(f"DEBUG (Sanity/Temp): ENABLED - Running content pre-filter on {len(matched_messages)} candidates...")
            matched_messages = sanity_check_filter(user_message, matched_messages, skip_intent_check=True)
            print(f"DEBUG (Sanity/Temp): {len(matched_messages)} candidates passed → feeding to Reranker.")
            if not matched_messages:
                print("DEBUG (Sanity/Temp): All candidates filtered — silent skip.")
                return None
        elif _sanity_on:
            print("DEBUG (Sanity/Temp): ENABLED but no candidates survived previous stages — skipping.")
        else:
            print("DEBUG (Sanity/Temp): DISABLED — skipping content pre-filter.")
        # --- END STAGE 2 ---

        # --- STAGE 3: RERANKER — final authoritative scoring (runs on pruned pool) ---
        if _reranker_on and RERANKER_MODEL and matched_messages:
            reranker_threshold = RESONANCE_SETTINGS.get("reranker_score_threshold", 0.0)
            batch_size = RESONANCE_SETTINGS.get("reranker_batch_size", 32)
            print(f"DEBUG (Reranker/Temp): ENABLED - Reranking {len(matched_messages)} candidates (post content-filter)...")
            reranked = rerank_candidates(
                query=user_message,
                candidates=matched_messages,
                top_k=k_final,
                score_threshold=reranker_threshold,
                batch_size=batch_size
            )
            if reranked:
                matched_messages = reranked
                print(f"DEBUG (Reranker/Temp): {len(matched_messages)} results after reranking.")
            else:
                print("DEBUG (Reranker/Temp): Reranker returned empty — keeping pre-filtered results.")
                matched_messages = matched_messages[:k_final]
        elif _reranker_on and not RERANKER_MODEL:
            print("WARNING (Reranker/Temp): ENABLED but model not loaded — skipping.")
            matched_messages = matched_messages[:k_final]
        else:
            matched_messages = matched_messages[:k_final]
        # --- END STAGE 3 ---

        # --- PAIRED MEMORY: expand hits into full exchanges ---
        if RESONANCE_SETTINGS.get("paired_memory_enabled", False):
            # FIX BUG 27: Pass active_contents_set so the expansion deduplicates
            # against messages already visible in the LLM's active window, same
            # as the permanent-index path does. Without this, paired memory in
            # the temporary fallback path can re-inject live messages.
            _temp_active_set = set()
            if active_memory:
                for _m in active_memory:
                    _c = _m.get("content")
                    if isinstance(_c, str) and _c.strip():
                        _temp_active_set.add(_c.strip())
            matched_messages = _expand_to_exchange(
                matched_messages, ghosted_messages, _temp_active_set
            )
        # --- END PAIRED MEMORY ---

        # 6. Format the final reflection string
        reflection_lines = []
        char_limit = RESONANCE_SETTINGS.get("recalled_message_char_limit", 30000)
        
        # --- FIX: Robust Name Fetching for TEMPORARY MEMORY ---
        user_name_setting = NAME_SETTINGS.get("user_name", "User")
        if not user_name_setting or not str(user_name_setting).strip():
            user_name_setting = "User"

        for msg in matched_messages:
            msg_content = msg.get("content", "")[:char_limit]
            
            display_name = "System"
            if msg.get("role") == "user":
                display_name = msg.get("name", user_name_setting)
                if not display_name or not str(display_name).strip():
                    display_name = "User"
            elif msg.get("role") == "assistant":
                display_name = msg.get("name", _current_active_character_name)
                if not display_name or not str(display_name).strip():
                    display_name = "Assistant"
            else:
                display_name = msg.get("name", msg.get('role', 'System').capitalize())
                if not display_name or not str(display_name).strip():
                    display_name = "System"
                    
            ts = msg.get('timestamp')
            if ts:
                try:
                    dt = datetime.fromisoformat(ts)
                    ts_tag = f" | {dt.strftime('%b %d %Y, %I:%M %p')}"
                except Exception:
                    ts_tag = ""
            else:
                ts_tag = ""

            label = f"[{display_name}{ts_tag}]:"
            reflection_lines.append(f"{label}\n{msg_content}")
        # --- END FIX ---

        separator = "\n---\n"
        # --- FIX: Unified Header Format ---
        header = RESONANCE_SETTINGS.get("memory_block_header", "[THIS BLOCK HERE ARE THE RETRIEVED MEMORIES AND PAST CONVERSATIONS]")
        closer = RESONANCE_SETTINGS.get("memory_block_closer", "[/END OF RETRIEVED MEMORIES FROM PAST CONVERSATIONS]")
        reflection_content = f"{header}\n---\n" + separator.join(reflection_lines) + f"\n---\n{closer}"
        reflection_tokens = count_tokens(reflection_content)
        print(f"DEBUG (Faiss): Generated reflection with {len(matched_messages)} messages, ~{reflection_tokens} tokens (temporary)")
        return reflection_content

    except Exception as e:
        print(f"ERROR (Faiss): An error occurred during temporary Faiss search: {e}")
        return None
# --- END OF FAISS-POWERED get_relevant_memories ---


def _parse_recall_entries(block_content):
    """
    Parse a recall block into a list of (label, content) tuples.
    Skips block header/footer/separators.
    FIX BUG 22: Use RESONANCE_SETTINGS for header/closer detection instead of
    hardcoded "[THIS BLOCK"/"[/END" which breaks when the user renames them.

    FIX BUG 30: State-aware label detection.
    Previously any line matching startswith("[") + endswith("]:") was treated as
    a label — including lines INSIDE message content. This broke for roleplay users
    who write lines like "[Alice]:", "[User]:", "[GM]:" inside their messages,
    causing the parser to silently drop the content above and start a phantom entry.

    Fix: track an `expecting_label` state. A [Name]: line is only promoted to a
    label when the parser is explicitly waiting for one (i.e. right after a ---
    separator, the block header/closer, or at the very start of the block).
    Once we're inside an entry's content, ALL lines — even [Foo]: — are content.
    """
    # Read live configured values so custom headers are handled correctly
    _header = RESONANCE_SETTINGS.get("memory_block_header", "[THIS BLOCK HERE ARE THE RETRIEVED MEMORIES AND PAST CONVERSATIONS]")
    _closer = RESONANCE_SETTINGS.get("memory_block_closer", "[/END OF RETRIEVED MEMORIES FROM PAST CONVERSATIONS]")
    # Build sentinel set — match by prefix of the first word so minor edits still work
    _header_start = _header[:10].strip() if _header else "[THIS BLOC"
    _closer_start = _closer[:10].strip() if _closer else "[/END OF R"

    entries = []
    current_label = None
    current_content_lines = []
    expecting_label = True  # FIX BUG 30: start expecting a label at top of block

    for line in block_content.splitlines():
        stripped = line.strip()

        # --- Separator / header / closer: flush current entry, reset to label-hunt mode ---
        if stripped.startswith(_header_start) or stripped.startswith(_closer_start) or stripped == "---":
            if current_label is not None:
                entries.append((current_label, "\n".join(current_content_lines).strip()))
                current_label = None
                current_content_lines = []
            expecting_label = True  # FIX BUG 30: after any divider we're back to hunting
            continue

        # --- Label line — ONLY accepted when we're actually expecting one ---
        # FIX BUG 30: without the `expecting_label` gate, a roleplay line like
        # "[Alice]: nice to meet you" inside message content would hijack the
        # parser, drop all content above it, and start a phantom new entry.
        if expecting_label and stripped.startswith("[") and stripped.endswith("]:"):
            if current_label is not None:
                entries.append((current_label, "\n".join(current_content_lines).strip()))
            current_label = stripped
            current_content_lines = []
            expecting_label = False  # FIX BUG 30: now reading content — lock out label detection
            continue

        # --- Content line (includes any [Name]: lines mid-content) ---
        if current_label is not None:
            current_content_lines.append(line)
            # expecting_label stays False until the next --- separator

    if current_label is not None:
        entries.append((current_label, "\n".join(current_content_lines).strip()))

    return entries


def _dedup_recall_block(new_reflection, memory, extra_active_contents=None):
    """
    Bouncer: strips entries from new_reflection whose content already exists
    in prior _recall_result blocks — BUT ONLY if those blocks are still in
    the active sliding window (i.e. the LLM can actually see them).

    If a recalled memory has slid out of the active window it no longer exists
    in the LLM's context, so blocking re-injection just causes hallucination.
    The fix: build two sets —
      - active_contents  : recall blocks still visible in the active window
      - ghosted_contents : recall blocks that exist in full memory but NOT in active window
    A memory is blocked only if it's in active_contents.
    If it's only in ghosted_contents it gets re-injected so the LLM can see it again.

    FIX BUG 31: extra_active_contents — ephemeral injection blind spot.
    When persistent_memory_injection=False AND rag_ghost_preservation=False, the
    raw_rag block is ephemeral: it goes into _ephemeral_blocks for this turn's LLM
    payload but is never appended to `memory`. This means when the LLM fires a
    [RECALL:] tool call in the same turn, this function scanned `memory`, found no
    trace of the ephemeral block, and let the same content through — the LLM then
    saw the identical memory twice in one context window.
    Fix: caller passes the parsed contents of any ephemeral blocks via this param
    so the bouncer can treat them as active even though they aren't in memory yet.
    """
    # Compute the active window the same way chat_stream does
    active_window = ghost_memory_if_needed(memory)
    active_ids = {id(msg) for msg in active_window}

    # Build active vs ghosted recall content sets
    active_contents = set()
    ghosted_contents = set()

    for msg in memory:
        if msg.get("_recall_result") and isinstance(msg.get("content"), str):
            entries = _parse_recall_entries(msg["content"])
            if id(msg) in active_ids:
                # Still in active window — LLM can see this
                for _, content in entries:
                    if content:
                        active_contents.add(content)
            else:
                # Slid out of active window — LLM has lost this context
                for _, content in entries:
                    if content:
                        ghosted_contents.add(content)

    # FIX: Also treat ANY content already visible in the active window as "active"
    # This catches cases where FAISS retrieves a message that's still in the live
    # conversation (not yet ghosted) — we should never re-inject what the LLM
    # can already see directly in its context.
    for msg in active_window:
        c = msg.get("content")
        if isinstance(c, str) and c.strip():
            active_contents.add(c.strip())

    # FIX BUG 31: merge in ephemeral block contents from caller (never in memory,
    # but ARE visible to the LLM this turn via the _ephemeral_blocks injection path)
    if extra_active_contents:
        active_contents.update(extra_active_contents)

    incoming_entries = _parse_recall_entries(new_reflection)

    if not incoming_entries:
        return None

    new_entries = []
    skipped_active = 0
    reinstated_ghosted = 0

    for label, content in incoming_entries:
        if not content:
            new_entries.append((label, content))
            continue

        if content in active_contents:
            # Already visible to the LLM right now — safe to skip
            skipped_active += 1
        elif content in ghosted_contents:
            # Was injected before but slid out — re-inject so LLM gets it back
            new_entries.append((label, content))
            reinstated_ghosted += 1
        else:
            # Brand new memory — always inject
            new_entries.append((label, content))

    if skipped_active > 0:
        print(f"DEBUG (Bouncer): {skipped_active} entr{'y' if skipped_active == 1 else 'ies'} still active in window — skipped.")
    if reinstated_ghosted > 0:
        print(f"DEBUG (Bouncer): {reinstated_ghosted} entr{'y' if reinstated_ghosted == 1 else 'ies'} slid out of window — RE-INJECTING.")

    if not new_entries:
        print("DEBUG (Bouncer): All recalled memories still active in window — skipping duplicate injection.")
        return None

    header = RESONANCE_SETTINGS.get("memory_block_header", "[THIS BLOCK HERE ARE THE RETRIEVED MEMORIES AND PAST CONVERSATIONS]")
    closer = RESONANCE_SETTINGS.get("memory_block_closer", "[/END OF RETRIEVED MEMORIES FROM PAST CONVERSATIONS]")
    parts = [header, "---"]
    for label, content in new_entries:
        parts.append(label)
        if content:
            parts.append(content)
        parts.append("---")
    parts.append(closer)

    total_new = len(new_entries)
    print(f"DEBUG (Bouncer): {total_new} entr{'y' if total_new == 1 else 'ies'} passed through ({reinstated_ghosted} reinstated from ghost).")
    return "\n".join(parts)


# --- REFACTORED AND UNIFIED CHAT STREAM ROUTE ---
# --- REFACTORED AGENTIC CHAT STREAM ---
@app.route("/chat_stream", methods=["POST"])
def chat_stream():
    """
    Handles chat requests with agentic search capabilities.
    Supports [SEARCH: query] tool triggers.
    """
    data = request.json
    user_message = data.get("message")
    file_data = data.get("file") 

    if not user_message and not file_data:
        return jsonify({"error": "No message or file provided"}), 400

    session_id = get_current_session_id()
    new_session_id_for_header = None
    memory = load_memory()

    if file_data and 'image' not in file_data.get('type', ''):
        filename = file_data.get("name")
        content = file_data.get("content")
        if filename and content:
            # FIX BUG 29: The latin-1 → unicode-escape decode was designed to handle
            # escape sequences in JS-serialised text, but it corrupts any non-ASCII
            # content (emoji, accented chars) and can produce garbled output for
            # binary-ish files. Just use the string as-is — the frontend already
            # sends UTF-8 text, so no re-encoding is needed.
            cleaned_content = content if isinstance(content, str) else str(content)
            memory.append({"role": "files", "content": f"[uploaded file content: '{filename}']:\n{cleaned_content}", "silent": True})

    user_name = NAME_SETTINGS.get("user_name", "User")
    # --- FIX: Only add name to memory if it's not empty ---
    user_payload = {"role": "user", "timestamp": datetime.now().isoformat(timespec='seconds')}
    if user_name and user_name.strip():
        user_payload["name"] = user_name
    
    llm_content_parts = []
    text_prompt = user_message
    if not text_prompt and file_data and 'image' in file_data.get('type', ''):
        text_prompt = "Describe this image in detail."
    
    if text_prompt:
        llm_content_parts.append({"type": "text", "text": text_prompt})
    if file_data and 'image' in file_data.get('type', ''):
        llm_content_parts.append({"type": "image_url", "image_url": {"url": file_data.get("content")}})

    if llm_content_parts:
        if len(llm_content_parts) == 1 and llm_content_parts[0]['type'] == 'text':
            user_payload['content'] = llm_content_parts[0]['text']
        else:
            user_payload['content'] = llm_content_parts
        memory.append(user_payload)
    
    save_memory(memory)

    # --- ALWAYS INDEX: Immediately index user message if toggle is ON ---
    if RESONANCE_SETTINGS.get("always_index_messages", False) and user_message:
        try:
            _always_index_single_message(user_payload, session_id)
        except Exception as _e:
            print(f"DEBUG (Always Index): Failed to index user message: {_e}")

    user_messages = [m for m in memory if m.get("role") == "user" and not m.get("silent")]
    if len(user_messages) == 1 and user_message:
        # FIX #4: Run session naming in a background thread so the stream starts immediately.
        # The renamed session ID is sent back via the X-Session-Renamed response header
        # once the background thread completes (header is set before streaming begins,
        # so we use an Event + shared list to let the thread pass the result back).
        _rename_result = [None]  # [new_session_id or None]
        _rename_done = threading.Event()

        def _do_rename():
            try:
                sanitized_name = llm_generate_session_name(user_message)
                old_path = os.path.join(SESSION_DIR, session_id)
                new_path = os.path.join(SESSION_DIR, sanitized_name)
                # FIX #7: Collision guard — if a session with that name already exists,
                # append a short unique suffix rather than clobbering it.
                if os.path.exists(new_path) and new_path != old_path:
                    suffix = uuid.uuid4().hex[:6]
                    sanitized_name = f"{sanitized_name}_{suffix}"
                    new_path = os.path.join(SESSION_DIR, sanitized_name)
                if os.path.exists(old_path):
                    os.rename(old_path, new_path)
                    set_current_session_id(sanitized_name)
                    _rename_result[0] = sanitized_name
                    print(f"DEBUG (Async Rename): Session renamed to '{sanitized_name}'")
            except Exception as e:
                print(f"ERROR: Async dynamic session naming failed: {e}")
            finally:
                _rename_done.set()

        rename_thread = threading.Thread(target=_do_rename, daemon=True)
        rename_thread.start()
        # Wait briefly (up to 3s) so the header can still be set before Flask sends it.
        # If the LLM is slow, we skip and the frontend will catch up on next reload.
        _rename_done.wait(timeout=3.0)
        if _rename_result[0]:
            session_id = _rename_result[0]
            new_session_id_for_header = _rename_result[0]

    system_prompt_message, active_character_name = get_system_prompt()
    global _current_active_character_name
    _current_active_character_name = active_character_name

    # --- Inject pinned user persona if set ---
    initial_active_memory_for_recall = ghost_memory_if_needed(memory)
    check_and_update_faiss_index(memory, initial_active_memory_for_recall, session_id)
    
    # FIX #4: Initialize both reflection slots unconditionally so image-only
    # requests (no user_message) never hit a NameError inside the inner generator.
    raw_rag_reflection = None
    resonance_reflection = None

    if user_message:
        # --- RAW RAG: Always-On Passive Recall (runs BEFORE intent-gated recall) ---
        if RESONANCE_SETTINGS.get("raw_rag_enabled", False):
            print("DEBUG (Raw RAG): Running passive ambient recall scan...")
            raw_rag_reflection = get_relevant_memories(
                user_message, memory, initial_active_memory_for_recall, session_id=session_id, raw_mode=True
            )
            if raw_rag_reflection:
                raw_rag_reflection = _dedup_recall_block(raw_rag_reflection, memory)
            if raw_rag_reflection:
                print("DEBUG (Raw RAG): Match found and passed threshold — injecting silently.")
                _persistent = RESONANCE_SETTINGS.get("persistent_memory_injection", False)
                _pinned = RESONANCE_SETTINGS.get("rag_ghost_preservation", False)
                if _persistent or _pinned:
                    _position = RESONANCE_SETTINGS.get("recall_injection_position", "after")
                    if _position == "before":
                        # Higher quality — LLM reads context then question, but block stays
                        # in the window next turn costing ~100-200 extra tokens until it ghosts out
                        _insert_idx = len(memory)
                        for _i in range(len(memory) - 1, -1, -1):
                            if memory[_i].get("role") == "user" and not memory[_i].get("silent"):
                                _insert_idx = _i
                                break
                        memory.insert(_insert_idx, {"role": "system", "content": raw_rag_reflection, "silent": True, "_recall_result": True, "_raw_rag": True, "_pinned": _pinned})
                    else:
                        # After (default) — seamless ~7-24 token cost, slides out next turn cleanly
                        memory.append({"role": "system", "content": raw_rag_reflection, "silent": True, "_recall_result": True, "_raw_rag": True, "_pinned": _pinned})
                    save_memory(memory)
                    raw_rag_reflection = None  # Already persisted, don't inject twice
                else:
                    print("DEBUG (Raw RAG): Both flags OFF — ephemeral only, not saved to history.")
                    # raw_rag_reflection stays set so it gets used this turn, just never saved
            else:
                print("DEBUG (Raw RAG): No match above threshold — silent skip, nothing injected.")

    # --- WRAPPER FUNCTION TO CATCH DISCONNECTS ---
    def agent_response_generator():
        _last_acc = [""]
        _is_saved = [False]
        try:
            yield from _agent_response_generator_inner(_last_acc, _is_saved)
        except GeneratorExit:
            print("DEBUG: Client disconnected mid-stream. Saving partial response.")
            if not _is_saved[0] and _last_acc[0].strip():
                try:
                    mem = load_memory()
                    # FIX BUG 8: Remove orphaned checkpoint if present before saving partial
                    if mem and mem[-1].get("_checkpoint"):
                        mem.pop()
                    clean = re.sub(r'<think>.*?</think>', '', _last_acc[0], flags=re.DOTALL)

                    # FIX BUG 9: Use actual configured search tokens instead of old hardcoded pattern
                    _disc_search_start = SEARCH_SETTINGS.get("search_tool_trigger", "<tool_search>")
                    _disc_search_end   = SEARCH_SETTINGS.get("search_tool_closer", "</tool_search>")
                    _disc_search_re    = re.compile(
                        re.escape(_disc_search_start) + r'\s*(.*?)' + re.escape(_disc_search_end),
                        re.IGNORECASE | re.DOTALL
                    )
                    clean = _disc_search_re.sub(r'\n\n> 🕵️ **SEARCHING:** *\1*\n\n', clean)

                    clean += "\n\n*(Stream interrupted...)*"
                    clean = clean.strip()

                    # ALWAYS append - even on interruption, history is sacred
                    mem.append({
                        "role": "assistant", 
                        "name": _current_active_character_name,
                        "content": clean.strip()
                    })
                    save_memory(mem)
                except Exception as e:
                    print(f"DEBUG: Error saving partial memory: {e}")
            raise # Re-raise so Flask cleanly closes the stream

    # --- INNER FUNCTION WITH TRACKERS ---
    def _agent_response_generator_inner(_last_acc, _is_saved):
        nonlocal memory
        max_agent_turns = 3 
        current_turn = 0
        messages_to_send_to_llm = []
        # FIX: Re-run ghost_memory_if_needed on the current memory instead of reusing
        # initial_active_memory_for_recall. The initial snapshot was taken BEFORE the
        # persistent RAG injection (raw_rag_reflection) was appended/inserted into memory,
        # so the recall block was invisible to the LLM on the turn it was triggered.
        # Re-snapshotting here ensures the just-injected recall block is included.
        active_memory = ghost_memory_if_needed(memory)
        # Keys that are internal app metadata and must NEVER be sent to the LLM.
        # Sending them changes the token structure and kills KV cache hits.
        # FIX BUG 28: Include chunking DLC metadata and always_indexed tag so they
        # are never forwarded to the LLM. Sending these keys corrupts the token
        # structure and breaks KV cache hits on every chunked recall.
        _INTERNAL_MSG_KEYS = {
            "silent", "_search_result", "_recall_result", "_pinned", "_checkpoint",
            "name", "session_source", "timestamp", "_raw_rag",
            # Chunking DLC metadata
            "chunk_source", "chunk_index", "chunk_total", "pinned", "always_indexed",
        }

        messages_to_send_to_llm.append(system_prompt_message)

        for msg in active_memory:
            msg_for_llm = {k: v for k, v in msg.items() if k not in _INTERNAL_MSG_KEYS}
            # Remap non-standard "files" role to "system" — OpenRouter (and strict
            # OpenAI-compatible backends) reject any role outside user/assistant/system.
            if msg_for_llm.get("role") == "files":
                msg_for_llm["role"] = "system"
            role = msg_for_llm.get("role")
            name = msg.get("name")  # Read name from original before strip
            content = msg_for_llm.get("content")
            
            # --- Name Prepending ---
            if role == "user":
                if name and name.strip():
                    if isinstance(content, str): 
                        msg_for_llm["content"] = f"{name} {content}"
                    elif isinstance(content, list): 
                        for part in content:
                            if part.get("type") == "text": 
                                part["text"] = f"{name} {part.get('text', '')}"
            # --------------------------------------------------

            messages_to_send_to_llm.append(msg_for_llm)

        # --- EPHEMERAL INJECTION: inject recalled memory into THIS turn's LLM payload only ---
        # Both flags OFF = ephemeral. Memory was never saved, so we manually inject it here
        # just for this one call. It will not appear in the next turn's context.
        _ephemeral_blocks = []
        if raw_rag_reflection:
            _ephemeral_blocks.append(raw_rag_reflection)
        if resonance_reflection:
            _ephemeral_blocks.append(resonance_reflection)

        # FIX BUG 31: Pre-parse ephemeral block contents so the agentic [RECALL:]
        # bouncer can see them even though they were never saved to `memory`.
        # Without this, _dedup_recall_block scanned `memory`, found no trace of
        # the ephemeral injection, and let duplicate content through — the LLM then
        # saw the same memory twice in one context window on ephemeral-mode turns.
        _ephemeral_active = set()
        for _blk in _ephemeral_blocks:
            for _, _c in _parse_recall_entries(_blk):
                if _c:
                    _ephemeral_active.add(_c)

        if _ephemeral_blocks:
            _position = RESONANCE_SETTINGS.get("recall_injection_position", "after")
            if _position == "before":
                # Find last user message, insert before it
                target_index = len(messages_to_send_to_llm)
                for i in range(len(messages_to_send_to_llm) - 1, -1, -1):
                    if messages_to_send_to_llm[i].get("role") == "user":
                        target_index = i
                        break
                for block in _ephemeral_blocks:
                    messages_to_send_to_llm.insert(target_index, {"role": "system", "content": block})
                    print("DEBUG (Ephemeral Inject): Memory block injected BEFORE user message.")
            else:
                # After (default) — append right after the last user message
                target_index = len(messages_to_send_to_llm)
                for i in range(len(messages_to_send_to_llm) - 1, -1, -1):
                    if messages_to_send_to_llm[i].get("role") == "user":
                        target_index = i + 1
                        break
                for block in _ephemeral_blocks:
                    messages_to_send_to_llm.insert(target_index, {"role": "system", "content": block})
                    print("DEBUG (Ephemeral Inject): Memory block injected AFTER user message.")

        # --- SETUP TRIGGERS ---
        search_start = SEARCH_SETTINGS.get("search_tool_trigger", "<tool_search>")
        search_end = SEARCH_SETTINGS.get("search_tool_closer", "</tool_search>")
        recall_start = SEARCH_SETTINGS.get("recall_tool_trigger", "[RECALL:")
        recall_end = SEARCH_SETTINGS.get("recall_tool_closer", "]")
        
        # Compile search regex
        search_regex = re.compile(r'(?:[:%#]+)?' + re.escape(search_start) + r'\s*(.*?)' + re.escape(search_end), re.IGNORECASE)
        # Compile recall regex
        recall_regex = re.compile(re.escape(recall_start) + r'\s*(.*?)' + re.escape(recall_end), re.IGNORECASE)

        while current_turn < max_agent_turns:
            _is_saved[0] = False  # Track reset per turn
            current_turn += 1
            full_response_acc = ""
            tool_triggered = None # 'search'
            
            yielded_len = 0
            current_temp = TEMPERATURE_SETTINGS.get("chat_stream", 0.4) if current_turn == 1 else 0.5  # FIX: fallback was 0.9 (Opus typo), matches load_temperature_settings default of 0.4
            
            for chunk in _get_llm_response_stream(messages_to_send_to_llm, temperature=current_temp):
                if isinstance(chunk, bytes): chunk = chunk.decode("utf-8")
                if chunk.strip() == "data: [DONE]": continue
                full_response_acc += chunk
                _last_acc[0] = full_response_acc  # Track accumulator for safety
                
                # Check Triggers
                if search_start in full_response_acc:
                    if search_regex.search(full_response_acc):
                         tool_triggered = 'search'
                         break
                    else: continue
                
                if recall_start in full_response_acc:
                    if recall_regex.search(full_response_acc):
                         tool_triggered = 'recall'
                         break
                    else: continue
                
                # Anti-Leak Logic for search trigger
                potential_leak = False
                for i in range(1, len(search_start)): 
                    if full_response_acc.endswith(search_start[:i]):
                        potential_leak = True
                        break
                
                # Anti-Leak Logic for recall trigger
                for i in range(1, len(recall_start)): 
                    if full_response_acc.endswith(recall_start[:i]):
                        potential_leak = True
                        break
                
                if potential_leak: continue 

                to_yield = full_response_acc[yielded_len:]
                if to_yield:
                    yield to_yield
                    yielded_len += len(to_yield)

            # --- EXECUTION ---
            if tool_triggered:
                query = ""
                widget_html = ""
                tool_output = ""
                clean_history_content = ""
                search_block_content = ""  # Initialize here

                if tool_triggered == 'search':
                    match = search_regex.search(full_response_acc)
                    if match:
                        query = match.group(1).strip()
                        print(f"DEBUG: Agent triggered SEARCH: '{query}'")
                        # SENTINEL: injected AFTER full_response_acc is frozen (LLM turn complete,
                        # KV cache already committed). Pure HTTP stream signal to the frontend —
                        # never stored in memory, never seen by the LLM. Frontend uses this to
                        # fold any orphaned mid-think reasoning back into the thought box cleanly.
                        yield ":::TOOL_FIRED:::"
                        yield f"\n\n> 🕵️ **SEARCHING:** *{query}* ...\n\n"
                        
                        try:
                            results = duckduckgo_search(query)
                            if not results: tool_output = "No results found."
                            else:
                                list_res = []
                                for r in results:
                                    title = r.get('title', 'Untitled')
                                    c = r.get('scraped_content') if r.get('scraped_content') else r.get('body', 'No content')
                                    list_res.append(f"- {title}: {c}")
                                tool_output = "\n".join(list_res)
                        except Exception as e:
                            tool_output = f"Error: {e}"

                        # --- SEARCH RESULTS AS SYSTEM ROLE ---
                        # System role for search results - treated as supplementary context
                        # BUG A FIX: was hardcoded "[Search Results]:" — now reads from SEARCH_SETTINGS
                        # so user-configured headers are respected by the frontend stripping regex.
                        _search_header = SEARCH_SETTINGS.get('search_result_header', '[Search Results]:')
                        search_block_content = f"{_search_header}\n{tool_output}\n[/End Search Results]"
                        system_msg = None

                elif tool_triggered == 'recall':
                    match = recall_regex.search(full_response_acc)
                    if match:
                        query = match.group(1).strip()
                        print(f"DEBUG: Agent triggered RECALL: '{query}'")
                        # Same sentinel as search — KV cache untouched, frontend-only signal.
                        yield ":::TOOL_FIRED:::"
                        yield f"\n\n> 🔮 **RECALLING:** *{query}* ...\n\n"  # BUG B FIX: was 🧠 — frontend processWidget looks for 🔮 to apply recall-widget CSS class
                        
                        try:
                            # Use the existing get_relevant_memories function (same as RAW RAG)
                            # It handles both permanent and temporary indexing automatically
                            # FIX (stale snapshot): Use a fresh ghost_memory_if_needed(memory) snapshot
                            # instead of initial_active_memory_for_recall, which was taken before any
                            # persistent RAG injection this turn. Stale snapshot = LLM doesn't see
                            # already-injected recall context = infinite recall loop.
                            recall_result = get_relevant_memories(
                                user_message=query,
                                memory=memory,
                                active_memory=ghost_memory_if_needed(memory),
                                session_id=session_id,
                                force_search=True,
                                raw_mode=False  # Don't apply RAW RAG thresholds
                            )

                            # FIX: Dedup against already-injected recall blocks (same as Raw RAG path)
                            # Prevents the same memory surfacing again in a multi-turn agentic loop.
                            # FIX BUG 31: also pass _ephemeral_active so transient (non-persisted)
                            # raw_rag injections are visible to the bouncer this turn.
                            if recall_result:
                                recall_result = _dedup_recall_block(recall_result, memory, extra_active_contents=_ephemeral_active)

                            if recall_result:
                                tool_output = recall_result
                            else:
                                tool_output = "No relevant memories found for that query."
                                
                        except Exception as e:
                            tool_output = f"Recall Error: {e}"
                            print(f"ERROR (Recall): {e}")
                            import traceback
                            traceback.print_exc()
                        
                        # Format recall results
                        recall_header = SEARCH_SETTINGS.get('recall_result_header', '[Recall Results]:')
                        search_block_content = f"{recall_header}\n{tool_output}\n[/End Recall Results]"
                        system_msg = None

                        # --- AGENTIC RECALL PERSISTENCE ---
                        # Mirror passive RAG behaviour exactly:
                        #   persistent_memory_injection OR rag_ghost_preservation → save to memory
                        #   _pinned driven by rag_ghost_preservation (immortal vs natural decay)
                        #   Both OFF → ephemeral this turn only (injected below into LLM payload)
                        # This prevents the infinite-recall loop: without persistence the next turn's
                        # snapshot has no record the recall happened, so the LLM fires again forever.
                        _recall_persistent = RESONANCE_SETTINGS.get("persistent_memory_injection", False)
                        _recall_pinned     = RESONANCE_SETTINGS.get("rag_ghost_preservation", False)
                        if (_recall_persistent or _recall_pinned) and tool_output and "No relevant memories" not in tool_output:
                            mem_recall = load_memory()
                            mem_recall.append({
                                "role":           "system",
                                "content":        search_block_content,
                                "silent":         True,
                                "_recall_result": True,
                                "_pinned":        _recall_pinned
                            })
                            save_memory(mem_recall)
                            print(f"DEBUG (Recall Persist): Saved agentic recall as SYSTEM entry. Pinned={_recall_pinned}, chars={len(search_block_content)}")
                        else:
                            print("DEBUG (Recall Persist): Both flags OFF — agentic recall is ephemeral this turn only.")

                # --- CHECKPOINT SAVE: RAW, UNMODIFIED OUTPUT (SAVE FIRST!) ---
                # Save the assistant's exact output at the moment the tool triggered.
                # This happens BEFORE search results, maintaining chronological order.
                # NO replacements, NO sanitizing, NO beautifying - just truth.
                try:
                    checkpoint_content = full_response_acc.strip()
                    if checkpoint_content:
                        mem_checkpoint = load_memory()
                        if (mem_checkpoint and
                                mem_checkpoint[-1].get("role") == "assistant" and
                                mem_checkpoint[-1].get("_checkpoint")):
                            mem_checkpoint[-1]["content"] = checkpoint_content
                            mem_checkpoint[-1]["name"] = _current_active_character_name
                        else:
                            mem_checkpoint.append({
                                "role": "assistant",
                                "name": _current_active_character_name,
                                "content": checkpoint_content,
                                "_checkpoint": True
                            })
                        save_memory(mem_checkpoint)
                        print(f"DEBUG (Checkpoint): Saved RAW assistant output ({len(checkpoint_content)} chars). No modifications.")
                except Exception as e:
                    print(f"DEBUG (Checkpoint): Failed to save checkpoint: {e}")
                # --- END CHECKPOINT SAVE ---

                # --- SEARCH RESULT PERSISTENCE (SAVE AFTER checkpoint!) ---
                # FIX #13: Only persist for search tool — recall results are already
                # handled by the RAG/memory system and should NOT be tagged _search_result.
                # Before this fix, recall tool results were persisted here with
                # _search_result=True, landing in the wrong bucket (Manage Memory →
                # "Persisted Search Results") and behaving incorrectly on deletion.
                if tool_triggered == 'search' and SEARCH_SETTINGS.get("persist_search_results", False) and tool_output and tool_output != "No results found.":
                    pinned = SEARCH_SETTINGS.get("search_result_pinned", True)
                    mem_search = load_memory()
                    mem_search.append({
                        "role": "system",
                        "content": search_block_content,
                        "silent": True,
                        "_search_result": True,
                        "_pinned": pinned
                    })
                    save_memory(mem_search)
                    print(f"DEBUG (Search Persist): Saved search results as SYSTEM role. Pinned={pinned}, chars={len(search_block_content)}")
                # --- END SEARCH RESULT PERSISTENCE ---

                # Update LLM context with RAW checkpoint content (NO replacements!)
                # This ensures KV cache consistency - what we save is what the LLM sees.
                messages_to_send_to_llm.append({"role": "assistant", "content": full_response_acc.strip()})
                if search_block_content:
                    messages_to_send_to_llm.append({"role": "system", "content": search_block_content})

                continue # Loop again


            # Flush buffer if no trigger
            if not tool_triggered and len(full_response_acc) > yielded_len:
                remaining = full_response_acc[yielded_len:]
                if remaining: yield remaining

            # Save final — replace checkpoint if one exists, otherwise append
            # PHILOSOPHY: "What happens in the past stays in the past — untouched."
            # Save full_response_acc VERBATIM. No reformatting, no tag stripping,
            # no unclosed-think surgery, no search badge substitution — nothing.
            # Even a dangling [RECALL: or mid-trigger cutoff is saved as-is.
            # The frontend renderer handles all display transforms. History is truth.
            _acc_to_save = full_response_acc

            # ERROR filter — don't save connection errors or exception traces as memories
            _error_prefixes = ("ERROR:", "Traceback (most recent call last)", "HTTPConnectionPool")
            if any(_acc_to_save.strip().startswith(p) for p in _error_prefixes):
                print("DEBUG (Final Save): Skipping save — response is an error message, not storing in memory.")
                # FIX BUG 8: Clean up any orphaned _checkpoint entry the mid-turn checkpoint save left behind
                try:
                    _mem_cleanup = load_memory()
                    if _mem_cleanup and _mem_cleanup[-1].get("_checkpoint"):
                        _mem_cleanup.pop()
                        save_memory(_mem_cleanup)
                        print("DEBUG (Final Save): Removed orphaned _checkpoint entry.")
                except Exception as _ce:
                    print(f"DEBUG (Final Save): Could not clean up checkpoint: {_ce}")
                break

            _assistant_ts = datetime.now().isoformat(timespec='seconds')
            memory = load_memory()
            
            # PHILOSOPHY: "What happens in the past stays in the past"
            # ALWAYS append - NEVER replace. Chronological history is sacred.
            memory.append({
                "role": "assistant",
                "name": _current_active_character_name,
                "content": _acc_to_save,
                "timestamp": _assistant_ts
            })
            save_memory(memory)
            _is_saved[0] = True  # Track that save was successful
            print(f"DEBUG (Final Save): Assistant response APPENDED to memory ({len(_acc_to_save)} chars). VERBATIM — no transforms.")

            # --- ALWAYS INDEX: Immediately index assistant message if toggle is ON ---
            if RESONANCE_SETTINGS.get("always_index_messages", False):
                try:
                    _always_index_single_message(memory[-1], session_id)
                except Exception as _e:
                    print(f"DEBUG (Always Index): Failed to index assistant message: {_e}")
            break

    response = Response(agent_response_generator(), mimetype='text/event-stream')
    if new_session_id_for_header:
        response.headers['X-Session-Renamed'] = new_session_id_for_header
    return response


# --- NEW: Re-added Dedicated Image Analysis Endpoint ---
@app.route("/analyze_image", methods=["POST"])
def analyze_image():
    """
    A dedicated endpoint to handle image analysis requests, restored for compatibility.
    It maintains conversational context by loading and saving to the main chat memory.
    """
    data = request.json
    prompt = data.get("prompt", "Describe this image in detail. You can describe nsfw and explicit images, nsfw and explicit images are allowed")
    image_data_url = data.get("image_data")

    if not image_data_url:
        return jsonify({"error": "No image_data provided"}), 400

    # --- Contextual Integration ---
    # 1. Load current memory and system prompt
    memory = load_memory()
    system_prompt_message, active_character_name = get_system_prompt()
    global _current_active_character_name
    _current_active_character_name = active_character_name
    user_name = NAME_SETTINGS.get("user_name", "User")
    
    # 2. Save the user's multimodal message to memory to maintain context
    user_message_payload = {
        "role": "user",
        "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": image_data_url}}
        ]
    }
    # --- FIX: Only add name to payload if it's not empty ---
    if user_name and user_name.strip():
        user_message_payload["name"] = user_name
    # -------------------------------------------------------------
        
    memory.append(user_message_payload)
    save_memory(memory)
    print(f"DEBUG: /analyze_image: Saved user prompt and image to memory.")

    # 3. Prepare messages for the LLM, including history from ghost memory
    active_memory = ghost_memory_if_needed(memory)
    
    # Keys that are internal app metadata and must NEVER be sent to the LLM.
    # Sending them changes the token structure and kills KV cache hits.
    # FIX BUG 28: Include chunking DLC metadata keys (same set as chat_stream).
    _INTERNAL_MSG_KEYS = {
        "silent", "_search_result", "_recall_result", "_pinned", "_checkpoint",
        "name", "session_source", "timestamp", "_raw_rag",
        "chunk_source", "chunk_index", "chunk_total", "pinned", "always_indexed",
    }

    messages_to_send_to_llm = [system_prompt_message]
    for msg in active_memory:
        msg_for_llm = {k: v for k, v in msg.items() if k not in _INTERNAL_MSG_KEYS}
        # Remap non-standard "files" role to "system" (same fix as chat_stream)
        if msg_for_llm.get("role") == "files":
            msg_for_llm["role"] = "system"
        role = msg_for_llm.get("role")
        name = msg.get("name")  # Read name from original before strip
        content = msg_for_llm.get("content")

        # --- Name Prepending ---
        if role == "user":
            if name and name.strip():
                if isinstance(content, str):
                    msg_for_llm["content"] = f"{name} {content}"
                elif isinstance(content, list):
                    new_content_list = []
                    for part in content:
                        new_part = part.copy()
                        if new_part.get("type") == "text":
                            new_part["text"] = f"{name} {new_part.get('text', '')}"
                        new_content_list.append(new_part)
                    msg_for_llm["content"] = new_content_list
        # --- End Name Prepending ---

        messages_to_send_to_llm.append(msg_for_llm)

    print(f"DEBUG: /analyze_image: Sending image for analysis with prompt: '{prompt}' and chat history.")

    # This generator function will stream the response back to the client.
    def response_generator():
        full_response_acc = ""
        try:
            # Use a lower temperature for more factual image description
            for chunk in _get_llm_response_stream(messages_to_send_to_llm, temperature=0.2):
                full_response_acc += chunk
                yield chunk
        except Exception as e:
            print(f"ERROR: /analyze_image streaming failed: {e}")
            yield "Error: Could not analyze the image."
        finally:
            # 4. Save the assistant's response to memory — but only if there's real content.
            # FIX BUG 21: The finally block runs even on stream errors, so guard against
            # saving empty strings or bare error messages as permanent memory entries.
            _skip_prefixes = ("Error:", "ERROR:")
            if full_response_acc.strip() and not any(full_response_acc.strip().startswith(p) for p in _skip_prefixes):
                assistant_response_payload = {
                    "role": "assistant",
                    "name": active_character_name,
                    "content": full_response_acc
                }
                mem = load_memory()
                mem.append(assistant_response_payload)
                save_memory(mem)
                print("DEBUG: /analyze_image: Saved assistant response to memory.")
            else:
                print("DEBUG: /analyze_image: Skipping save — empty or error response.")

    return Response(response_generator(), mimetype='text/plain')


@app.route("/new_session", methods=["POST"])
def new_session():
    current_session_id = get_current_session_id()
    try:
        # Find all session indices to avoid reusing numbers
        indices = [int(s.split('_')[1]) for s in os.listdir(SESSION_DIR) if s.startswith('session_') and s.split('_')[1].isdigit()]
        index = max(indices) + 1 if indices else 1
    except (IndexError, ValueError):
        index = 1
    # FIX BUG 24: The calculated index might still collide with a renamed session
    # (e.g. a session renamed away from session_NNN frees up that number, but another
    # session using a non-numeric name could have the same folder name coincidentally).
    # Bump index until we find a genuinely free slot.
    new_id = f"session_{index:03d}"
    while os.path.exists(os.path.join(SESSION_DIR, new_id)):
        index += 1
        new_id = f"session_{index:03d}"
    set_current_session_id(new_id)
    save_memory([])
    # FIX Bug 1: Return session_id as an explicit field so frontend doesn't
    # have to parse it out of a human-readable message string
    return jsonify({"message": f"Switched to new session: {new_id}", "session_id": new_id}), 200


# --- FIX: Simplified /history endpoint ---
@app.route("/history", methods=["GET"])
def get_history():
    """
    Returns the chat history.
    FIX BUG 11 (revised): Only strip _checkpoint entries that are ORPHANED — i.e.
    immediately followed by another non-checkpoint assistant message (meaning the
    final save already replaced them). In the agentic case, the pre-search turn1
    response IS saved only as a _checkpoint and must remain visible — it is the
    intentional record of what the LLM said before calling the tool.
    """
    memory = load_memory()

    _search_trigger = SEARCH_SETTINGS.get("search_tool_trigger", "<tool_search>")
    _recall_trigger  = SEARCH_SETTINGS.get("recall_tool_trigger", "[RECALL:")

    filtered = []
    for i, msg in enumerate(memory):
        if not msg.get("_checkpoint"):
            filtered.append(msg)
            continue

        # _checkpoint entries come in two flavours:
        #
        # 1. AGENTIC (intentional): the pre-tool-call response saved mid-stream.
        #    Identifiable because it contains a search or recall trigger token.
        #    This is the ONLY persistent record of that turn — keep it.
        #    The frontend merge logic uses the tool token to stitch turn1+results+turn2.
        #
        # 2. ORPHANED (accidental): saved before a crash / error-filter break.
        #    The final clean save ran afterward and appended a separate clean entry,
        #    making this checkpoint a stale duplicate. Strip it.
        content = msg.get("content", "") or ""
        is_agentic = _search_trigger in content or _recall_trigger in content
        if is_agentic:
            filtered.append(msg)
            continue

        # No tool token → may be orphaned. Orphaned = a clean assistant entry exists
        # later in memory (the final save ran and this checkpoint is now redundant).
        is_orphaned = any(
            m.get("role") == "assistant" and not m.get("_checkpoint") and not m.get("silent")
            for m in memory[i + 1:]
        )
        if is_orphaned:
            print(f"DEBUG (/history): Filtered orphaned _checkpoint at index {i}.")
        else:
            filtered.append(msg)

    return jsonify(filtered)


@app.route("/delete_message", methods=["POST"])
def delete_message():
    data = request.json
    index = data.get("index")
    # Accept explicit session_id from the frontend so a concurrent async rename
    # can never cause save_memory to write to the wrong session.
    req_session_id = data.get("session_id", "").strip() or get_current_session_id()

    # FIX #14: Load and save directly using the resolved filepath instead of
    # temporarily mutating the global current_session pointer via set_current_session_id.
    # The old approach was thread-unsafe — a concurrent request hitting load_memory()
    # or save_memory() mid-redirect would read/write the wrong session file.
    filepath = get_session_filepath(req_session_id)
    try:
        with _memory_lock:
            if os.path.exists(filepath):
                with open(filepath, "r", encoding='utf-8') as f:
                    try:
                        memory = json.load(f)
                    except json.JSONDecodeError:
                        memory = []
            else:
                memory = []

        index = int(index)
        user_message_indices = [i for i, m in enumerate(memory) if m["role"] == "user" and not m.get("silent")]

        if 0 <= index < len(user_message_indices):
            mem_index_to_delete = user_message_indices[index]

            indices_to_delete = [mem_index_to_delete]
            scan = mem_index_to_delete + 1
            while scan < len(memory):
                msg = memory[scan]
                if msg["role"] == "assistant":
                    indices_to_delete.append(scan)
                    break
                elif msg.get("silent") or msg["role"] == "system":
                    indices_to_delete.append(scan)
                    scan += 1
                else:
                    break

            for i in sorted(indices_to_delete, reverse=True):
                del memory[i]

            tmp_path = filepath + ".tmp"
            with _memory_lock:
                with open(tmp_path, "w", encoding='utf-8') as f:
                    json.dump(memory, f, indent=2)
                os.replace(tmp_path, filepath)
            return jsonify({"message": "Messages deleted."})
        else:
            return jsonify({"error": "Invalid message index"}), 400
    except (ValueError, TypeError):
        return jsonify({"error": "Invalid index format"}), 400
    except Exception as e:
        print(f"Error deleting message: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/edit_message", methods=["POST"])
def edit_message():
    data = request.json
    index = data.get("index")
    new_content = data.get("content", "").strip()
    req_session_id = data.get("session_id", "").strip() or get_current_session_id()

    if index is None or not new_content:
        return jsonify({"error": "Index and content are required."}), 400

    # FIX #14: Same as delete_message — operate on the filepath directly rather
    # than mutating the global session pointer.
    filepath = get_session_filepath(req_session_id)
    try:
        with _memory_lock:
            if os.path.exists(filepath):
                with open(filepath, "r", encoding='utf-8') as f:
                    try:
                        memory = json.load(f)
                    except json.JSONDecodeError:
                        memory = []
            else:
                memory = []

        index = int(index)
        if 0 <= index < len(memory):
            msg = memory[index]
            if msg.get("role") in ["user", "assistant"] and not msg.get("silent"):
                memory[index]["content"] = new_content
                tmp_path = filepath + ".tmp"
                with _memory_lock:
                    with open(tmp_path, "w", encoding='utf-8') as f:
                        json.dump(memory, f, indent=2)
                    os.replace(tmp_path, filepath)
                return jsonify({"message": "Message updated."})
            else:
                return jsonify({"error": "Cannot edit this message type."}), 400
        else:
            return jsonify({"error": "Index out of bounds."}), 400
    except (ValueError, TypeError):
        return jsonify({"error": "Invalid index format."}), 400


@app.route("/sessions", methods=["GET"])
def list_sessions():
    sessions = []
    for entry in os.listdir(SESSION_DIR):
        session_path = os.path.join(SESSION_DIR, entry)
        if os.path.isdir(session_path):
            if os.path.exists(os.path.join(session_path, "chat_memory.json")):
                sessions.append(entry)
    return jsonify(sorted(sessions))


@app.route("/rename_session", methods=["POST"])
def rename_session():
    data = request.get_json()
    old_id = data.get("old_id", "").strip()
    new_id = data.get("new_id", "").strip()
    # FIX BUG 16: Reject names that could escape SESSION_DIR via path traversal
    _INVALID_CHARS = ('..', '/', '\\')
    if not old_id or not new_id or any(c in old_id or c in new_id for c in _INVALID_CHARS):
        return jsonify({"error": "Invalid session name."}), 400
    old_folder_path = os.path.join(SESSION_DIR, old_id)
    new_folder_path = os.path.join(SESSION_DIR, new_id)
    if not os.path.exists(old_folder_path):
        return jsonify({"error": "Old session folder not found"}), 404
    if os.path.exists(new_folder_path):
        return jsonify({"error": "New session name already exists"}), 400
    os.rename(old_folder_path, new_folder_path)
    if get_current_session_id() == old_id:
        set_current_session_id(new_id)
    return jsonify({"message": "Session renamed."})


@app.route("/delete_session", methods=["POST"])
def delete_session():
    data = request.get_json()
    session_id = data.get("session_id", "").strip()
    # FIX BUG 16: Reject names that could escape SESSION_DIR via path traversal
    _INVALID_CHARS = ('..', '/', '\\')
    if not session_id or any(c in session_id for c in _INVALID_CHARS):
        return jsonify({"error": "Invalid session name."}), 400
    session_folder_path = os.path.join(SESSION_DIR, session_id)
    if not os.path.exists(session_folder_path):
        return jsonify({"error": "Session not found"}), 404
    try:
        # shutil.rmtree will delete the folder and ALL its contents,
        # including chat_memory.json, faiss_index.idx, and faiss_metadata.json
        shutil.rmtree(session_folder_path)
        print(f"DEBUG: Deleted session folder: {session_folder_path}")
        if get_current_session_id() == session_id:
            remaining_sessions = [
                d for d in os.listdir(SESSION_DIR)
                if os.path.isdir(os.path.join(SESSION_DIR, d)) and
                   os.path.exists(os.path.join(SESSION_DIR, d, "chat_memory.json"))
            ]
            if remaining_sessions:
                set_current_session_id(sorted(remaining_sessions)[0])
            else:
                set_current_session_id("session_001")
                save_memory([])
        return jsonify({"message": "Session deleted."})
    except Exception as e:
        print(f"Error deleting session folder: {e}")
        return jsonify({"error": f"Failed to delete session: {e}"}), 500


@app.route("/switch_session", methods=["POST"])
def switch_session():
    data = request.get_json()
    session_id = data.get("session_id", "").strip()
    session_folder_path = os.path.join(SESSION_DIR, session_id)
    if not session_id or not os.path.exists(session_folder_path):
        return jsonify({"error": "Session not found"}), 404
    set_current_session_id(session_id)
    
    # --- NEW: AUTO-INDEX ON SESSION LOAD ---
    print(f"DEBUG (Faiss): Switched to session {session_id}. Checking index.")
    memory = load_memory()
    active_memory = ghost_memory_if_needed(memory)
    check_and_update_faiss_index(memory, active_memory, session_id)
    # Global index rebuild is handled inside check_and_update_faiss_index
    # --- END NEW ---
    
    return jsonify({"message": f"Switched to {session_id}"})


@app.route("/get_active_character_name", methods=["GET"])
def api_get_active_character_name():
    # Call get_system_prompt to ensure _current_active_character_name is set
    _, active_char_name = get_system_prompt()
    return jsonify({"active_character_name": active_char_name})


# New API endpoint to get the current system prompt content by name
@app.route("/get_system_prompt_content", methods=["GET"])
def api_get_system_prompt_content():
    persona_name = request.args.get("persona_name")
    content = load_system_prompt_content(persona_name)
    return jsonify({
        "prompt_content": content,
        "persona_name": persona_name or get_current_persona_name()
    })


# New API endpoint to set the system prompt content for a specific persona
@app.route("/set_system_prompt", methods=["POST"])
def api_set_system_prompt():
    data = request.json
    prompt_content = data.get("prompt_content", "").strip()
    persona_name = data.get("persona_name", "").strip()
    # FIX BUG 12: frontend can send activate=true to explicitly switch active persona,
    # or activate=false to do a silent save without switching. Defaults to True so
    # existing JS (which always treats save as activate) keeps working until the UI
    # is updated to have a dedicated Activate button.
    make_active = data.get("activate", True)

    if not persona_name:
        return jsonify({"error": "No persona name provided."}), 400

    save_system_prompt_content(persona_name, prompt_content, make_active=make_active)

    # After updating the system prompt, re-evaluate the active character name
    _, active_char_name = get_system_prompt()

    return jsonify({
        "message": f"System prompt '{persona_name}' updated successfully.",
        "active_character_name": active_char_name
    })


# New API endpoint to list all available persona names
@app.route("/list_personas", methods=["GET"])
def api_list_personas():
    return jsonify(list(PERSONAS.keys()))


# NEW: API endpoint to delete a persona
@app.route("/delete_persona", methods=["POST"])
def api_delete_persona():
    data = request.json
    persona_name = data.get("persona_name", "").strip()

    if not persona_name:
        return jsonify({
            "error": "No persona name provided for deletion."
        }), 400

    if persona_name not in PERSONAS:
        return jsonify({"error": f"Persona '{persona_name}' not found."}), 404

    # Prevent deletion of the currently active persona
    if persona_name == get_current_persona_name():
        return jsonify({
            "error": f"Cannot delete the currently active persona "
                     f"'{persona_name}'. Please switch to another first."
        }), 400

    personas_copy = PERSONAS.copy()
    del personas_copy[persona_name]
    save_all_personas(personas_copy)

    return jsonify({"message": f"Persona '{persona_name}' deleted."})


@app.route("/rename_persona", methods=["POST"])
def api_rename_persona():
    data = request.json
    old_name = data.get("old_name", "").strip()
    new_name = data.get("new_name", "").strip()

    if not old_name or not new_name:
        return jsonify({"error": "Both old_name and new_name are required."}), 400

    if old_name not in PERSONAS:
        return jsonify({"error": f"Persona '{old_name}' not found."}), 404

    if new_name in PERSONAS:
        return jsonify({"error": f"A persona named '{new_name}' already exists."}), 409

    personas_copy = PERSONAS.copy()
    # Preserve the full dict (prompt + avatar_image) under the new key
    personas_copy[new_name] = personas_copy.pop(old_name)
    save_all_personas(personas_copy)

    # If this was the active persona, update the active name file too
    if old_name == get_current_persona_name():
        set_current_persona_name(new_name)
        print(f"DEBUG: Active persona renamed '{old_name}' → '{new_name}'.")

    return jsonify({"message": f"Persona renamed to '{new_name}'.", "new_name": new_name})


# --- NEW: API Endpoints for General App Settings ---
@app.route("/get_app_settings", methods=["GET"])
def get_app_settings():
    """Returns the current application settings."""
    return jsonify(APP_SETTINGS)


@app.route("/set_app_settings", methods=["POST"])
def set_app_settings():
    """Receives and saves new application settings."""
    data = request.json
    global APP_SETTINGS
    for key, value in data.items():
        if key in APP_SETTINGS:
            # Add type conversion for safety
            try:
                if isinstance(APP_SETTINGS[key], bool):
                    APP_SETTINGS[key] = bool(value)
                elif isinstance(APP_SETTINGS[key], int):
                    APP_SETTINGS[key] = int(value)
                elif isinstance(APP_SETTINGS[key], float):
                    APP_SETTINGS[key] = float(value)
                else:
                    APP_SETTINGS[key] = str(value)
            except (ValueError, TypeError):
                return jsonify({"error": f"Invalid value type for {key}"}), 400
        else:
            print(f"WARNING (set_app_settings): Unknown key '{key}' ignored.")
    save_app_settings()
    return jsonify({"message": "Application settings updated."})


# --- API Profile Routes ---

@app.route("/get_api_profiles", methods=["GET"])
def get_api_profiles():
    """Returns all API profiles and the currently active one."""
    return jsonify({
        "profiles": API_PROFILES,
        "active": get_current_api_profile_name()
    })


@app.route("/save_api_profile", methods=["POST"])
def save_api_profile():
    """Creates or overwrites an API profile and activates it."""
    global API_PROFILES
    data = request.json
    name = data.get("name", "").strip()
    if not name:
        return jsonify({"error": "Profile name is required."}), 400
    API_PROFILES[name] = {
        "endpoint": data.get("endpoint", "").strip(),
        "model":    data.get("model", "").strip(),
        "api_key":  data.get("api_key", "").strip(),
        "mode":     data.get("mode", "kobold").strip(),
    }
    # Activate the just-saved profile so badge and dropdown stay in sync
    set_current_api_profile_name(name)
    _save_api_profiles()
    return jsonify({"message": f"Profile '{name}' saved.", "active": name})


@app.route("/switch_api_profile", methods=["POST"])
def switch_api_profile():
    """Switches the active profile and immediately applies it to APP_SETTINGS."""
    global APP_SETTINGS
    data = request.json
    name = data.get("name", "").strip()
    if not name or name not in API_PROFILES:
        return jsonify({"error": f"Profile '{name}' not found."}), 404
    profile = API_PROFILES[name]
    APP_SETTINGS["llm_api_endpoint"] = profile.get("endpoint", APP_SETTINGS["llm_api_endpoint"])
    APP_SETTINGS["llm_model_name"]   = profile.get("model", "")
    APP_SETTINGS["llm_api_key"]      = profile.get("api_key", "")
    if "mode" in profile:
        APP_SETTINGS["backend_mode"] = profile["mode"]
    save_app_settings()
    set_current_api_profile_name(name)
    return jsonify({"message": f"Switched to '{name}'.", "profile": profile})


@app.route("/delete_api_profile", methods=["POST"])
def delete_api_profile():
    """Deletes an API profile. Cannot delete the last one."""
    global API_PROFILES
    data = request.json
    name = data.get("name", "").strip()
    if not name or name not in API_PROFILES:
        return jsonify({"error": f"Profile '{name}' not found."}), 404
    if len(API_PROFILES) <= 1:
        return jsonify({"error": "Cannot delete the last profile."}), 400
    if name == get_current_api_profile_name():
        return jsonify({"error": "Cannot delete the active profile. Switch first."}), 400
    del API_PROFILES[name]
    _save_api_profiles()
    return jsonify({"message": f"Profile '{name}' deleted."})

# --- END API Profile Routes ---


# --- NEW: API Endpoints for Temperature Settings ---
@app.route("/get_temperatures", methods=["GET"])
def get_temperatures():
    """Returns the current temperature settings."""
    return jsonify(TEMPERATURE_SETTINGS)


@app.route("/set_temperatures", methods=["POST"])
def set_temperatures():
    """Receives and saves new temperature settings."""
    data = request.json
    global TEMPERATURE_SETTINGS
    # Update only the keys provided in the request
    for key, value in data.items():
        if key in TEMPERATURE_SETTINGS:
            try:
                TEMPERATURE_SETTINGS[key] = float(value)
            except (ValueError, TypeError):
                return jsonify({"error": f"Invalid value for {key}"}), 400
    save_temperature_settings()
    return jsonify({"message": "Temperature settings updated."})


# --- NEW: API Endpoints for Token Settings ---
@app.route("/get_tokens", methods=["GET"])
def get_tokens():
    """Returns the current token settings."""
    return jsonify(TOKEN_SETTINGS)


@app.route("/set_tokens", methods=["POST"])
def set_tokens():
    """Receives and saves new token settings."""
    data = request.json
    global TOKEN_SETTINGS
    # Update only the keys provided in the request
    for key, value in data.items():
        if key in TOKEN_SETTINGS:
            try:
                # Handle booleans for toggles
                if key in ["resonance_enabled", "coding_mode_enabled", "ghost_memory_enabled", "precision_mode_enabled"]:
                    TOKEN_SETTINGS[key] = bool(value)
                else:
                    TOKEN_SETTINGS[key] = int(value)
            except (ValueError, TypeError):
                return jsonify({"error": f"Invalid value for {key}"}), 400
    # FIX BUG 23: Enforce even number for max_chat_messages on save, same as on load.
    # An odd window size can leave a dangling user message with no assistant reply visible,
    # causing confusing UI state.
    if TOKEN_SETTINGS.get("max_chat_messages", 16) % 2 != 0:
        TOKEN_SETTINGS["max_chat_messages"] += 1
    save_token_settings()
    return jsonify({"message": "Token settings updated."})


# --- NEW: API Endpoints for Resonance Settings ---
@app.route("/get_resonance_settings", methods=["GET"])
def get_resonance_settings():
    """Returns the current resonance settings."""
    return jsonify(RESONANCE_SETTINGS)


@app.route("/set_resonance_settings", methods=["POST"])
def set_resonance_settings():
    """Receives and saves new resonance settings."""
    data = request.json
    global RESONANCE_SETTINGS
    # Update only the keys provided in the request
    for key, value in data.items():
        if key in RESONANCE_SETTINGS:
            # IMPORTANT: bool check MUST come before int — bool is a subclass of int in Python
            if isinstance(RESONANCE_SETTINGS[key], bool):
                try:
                    RESONANCE_SETTINGS[key] = bool(value)
                except (ValueError, TypeError):
                    return jsonify({"error": f"Invalid value type for {key}, expected a boolean"}), 400
            elif isinstance(RESONANCE_SETTINGS[key], list):
                if isinstance(value, list):
                    RESONANCE_SETTINGS[key] = value
                else:
                    return jsonify({"error": f"Invalid value type for {key}, expected a list"}), 400
            elif isinstance(RESONANCE_SETTINGS[key], int):
                try:
                    RESONANCE_SETTINGS[key] = int(value)
                except (ValueError, TypeError):
                    return jsonify({"error": f"Invalid value type for {key}, expected an integer"}), 400
            elif isinstance(RESONANCE_SETTINGS[key], float):
                try:
                    RESONANCE_SETTINGS[key] = float(value)
                except (ValueError, TypeError):
                    return jsonify({"error": f"Invalid value type for {key}, expected a float"}), 400
            else:
                RESONANCE_SETTINGS[key] = value
    save_resonance_settings()
    return jsonify({"message": "Resonance settings updated."})


# --- CHUNKING DLC API Endpoints ---
@app.route("/get_chunking_settings", methods=["GET"])
def get_chunking_settings():
    """Returns all chunking DLC settings from RESONANCE_SETTINGS."""
    keys = [
        "chunking_enabled", "chunk_token_limit", "chunk_overlap_tokens",
        "chunk_unit", "chunk_retrieval_mode", "long_context_threshold",
        "chunk_pinned_always_reassemble", "chunk_index_system_immediately"
    ]
    return jsonify({k: RESONANCE_SETTINGS.get(k) for k in keys})


@app.route("/set_chunking_settings", methods=["POST"])
def set_chunking_settings():
    """Saves chunking DLC settings into RESONANCE_SETTINGS and persists to disk."""
    data = request.json
    global RESONANCE_SETTINGS
    type_map = {
        "chunking_enabled":              bool,
        "chunk_token_limit":             int,
        "chunk_overlap_tokens":          int,
        "chunk_unit":                    str,
        "chunk_retrieval_mode":          str,
        "long_context_threshold":        int,
        "chunk_pinned_always_reassemble": bool,
        "chunk_index_system_immediately": bool,
    }
    for key, cast in type_map.items():
        if key in data:
            try:
                RESONANCE_SETTINGS[key] = cast(data[key])
            except (ValueError, TypeError):
                return jsonify({"error": f"Invalid value for {key}"}), 400
    save_resonance_settings()
    return jsonify({"message": "Chunking settings saved."})
# --- END CHUNKING DLC API ---


# --- NEW: API Endpoints for Sampler Settings ---
@app.route("/get_sampler_settings", methods=["GET"])
def get_sampler_settings():
    """Returns the current sampler settings."""
    return jsonify(SAMPLER_SETTINGS)


@app.route("/set_sampler_settings", methods=["POST"])
def set_sampler_settings():
    """Receives and saves new sampler settings."""
    data = request.json
    global SAMPLER_SETTINGS
    for key, value in data.items():
        if key in SAMPLER_SETTINGS:
            try:
                if isinstance(SAMPLER_SETTINGS[key], bool):
                    SAMPLER_SETTINGS[key] = bool(value)
                else:
                    SAMPLER_SETTINGS[key] = float(value)
            except (ValueError, TypeError):
                return jsonify({"error": f"Invalid value for {key}"}), 400
    save_sampler_settings()
    return jsonify({"message": "Sampler settings updated."})

# --- NEW: API Endpoints for Search Settings ---
@app.route("/get_search_settings", methods=["GET"])
def get_search_settings():
    """Returns the current search settings."""
    return jsonify(SEARCH_SETTINGS)


@app.route("/set_search_settings", methods=["POST"])
def set_search_settings():
    """Receives and saves new search settings."""
    data = request.json
    global SEARCH_SETTINGS
    allowed = set(SEARCH_SETTINGS.keys())
    for key, value in data.items():
        if key in allowed:
            SEARCH_SETTINGS[key] = value
        else:
            print(f"WARNING (set_search_settings): Unknown key '{key}' ignored.")
    save_search_settings()
    return jsonify({"message": "Search settings updated."})


# --- NEW: API Endpoints for Name Settings ---
@app.route("/get_name_settings", methods=["GET"])
def get_name_settings():
    """Returns the current name settings."""
    # Dynamically update the assistant name based on the current persona before returning
    _, assistant_name = get_system_prompt()
    current_settings = NAME_SETTINGS.copy()
    current_settings["assistant_name"] = assistant_name
    return jsonify(current_settings)


@app.route("/set_name_settings", methods=["POST"])
def set_name_settings():
    """Receives and saves new name settings."""
    data = request.json
    global NAME_SETTINGS
    # FIX BUG 2: previously used `if key in NAME_SETTINGS` which silently dropped any
    # new fields (e.g. a future nametag key) that weren't already in the defaults dict.
    # Now we accept any string-valued key the frontend sends, same pattern as
    # set_appearance_settings. assistant_name is intentionally excluded here because
    # it is always derived live from the active persona via get_system_prompt().
    for key, value in data.items():
        if key == "assistant_name":
            continue  # read-only — always driven by active persona, never stored here
        NAME_SETTINGS[key] = str(value)
    save_name_settings()
    return jsonify({"message": "Name settings updated."})


@app.route("/get_appearance_settings", methods=["GET"])
def get_appearance_settings():
    """Returns the current appearance settings."""
    return jsonify(APPEARANCE_SETTINGS)


@app.route("/set_appearance_settings", methods=["POST"])
def set_appearance_settings():
    """Saves appearance settings."""
    global APPEARANCE_SETTINGS
    data = request.json
    if not data:
        return jsonify({"error": "No data provided."}), 400
    # Merge nested colors dict carefully
    if "colors" in data and isinstance(data["colors"], dict):
        if "colors" not in APPEARANCE_SETTINGS:
            APPEARANCE_SETTINGS["colors"] = {}
        APPEARANCE_SETTINGS["colors"].update(data["colors"])
        data_without_colors = {k: v for k, v in data.items() if k != "colors"}
        APPEARANCE_SETTINGS.update(data_without_colors)
    else:
        APPEARANCE_SETTINGS.update(data)
    save_appearance_settings()
    return jsonify({"status": "ok"})


@app.route("/get_streaming_settings", methods=["GET"])
def get_streaming_settings():
    """Returns the current streaming settings."""
    return jsonify(STREAMING_SETTINGS)


@app.route("/set_streaming_settings", methods=["POST"])
def set_streaming_settings():
    """Saves streaming settings. Coerces streamingDelayEnabled to a proper bool."""
    global STREAMING_SETTINGS
    data = request.json
    if not data:
        return jsonify({"error": "No data provided."}), 400
    # Coerce streamingDelayEnabled to bool in case the client sends 0/1 or a string
    if "streamingDelayEnabled" in data:
        data["streamingDelayEnabled"] = bool(data["streamingDelayEnabled"])
    STREAMING_SETTINGS.update(data)
    save_streaming_settings()
    return jsonify({
        "status": "ok",
        "streamingDelayEnabled": STREAMING_SETTINGS.get("streamingDelayEnabled", True)
    })


# --- NEW: DEDICATED FAISS SETTINGS ENDPOINTS ---

@app.route("/get_faiss_settings", methods=["GET"])
def get_faiss_settings():
    """Returns only the Faiss-related settings from the resonance config."""
    faiss_settings = {
        "faiss_permanent_indexing": RESONANCE_SETTINGS.get("faiss_permanent_indexing", True),
        "faiss_indexing_quota": RESONANCE_SETTINGS.get("faiss_indexing_quota", 50),
        "faiss_distance_metric": RESONANCE_SETTINGS.get("faiss_distance_metric", "l2"),
        "embedding_model": RESONANCE_SETTINGS.get("embedding_model", "nomic"),
        "embedding_ctx_length": RESONANCE_SETTINGS.get("embedding_ctx_length", EMBEDDING_CTX_LENGTH),
        "global_memory_enabled": RESONANCE_SETTINGS.get("global_memory_enabled", False),
        "global_memory_session_limit": RESONANCE_SETTINGS.get("global_memory_session_limit", 3),
        "always_index_messages": RESONANCE_SETTINGS.get("always_index_messages", False),
        "embedding_prefix_mode": RESONANCE_SETTINGS.get("embedding_prefix_mode", "auto"),
    }
    return jsonify(faiss_settings)

def _wipe_all_faiss_indexes(reason=""):
    """
    Helper: deletes all per-session and global FAISS index files.
    Called ONLY when the embedding model or distance metric actually changes —
    never on a plain settings save, to protect potato CPUs from surprise rebuilds.
    """
    tag = f" ({reason})" if reason else ""
    wiped = 0
    for session_folder in os.listdir(SESSION_DIR):
        session_path = os.path.join(SESSION_DIR, session_folder)
        if os.path.isdir(session_path):
            for fname in ["faiss_index.idx", "faiss_metadata.json"]:
                fpath = os.path.join(session_path, fname)
                if os.path.exists(fpath):
                    os.remove(fpath)
                    wiped += 1
                    print(f"DEBUG (Faiss wipe{tag}): Removed {fname} from {session_folder}")
    for fname in ["global.faiss", "global.meta.json"]:
        fpath = os.path.join(GLOBAL_MEMORY_DIR, fname)
        if os.path.exists(fpath):
            os.remove(fpath)
            wiped += 1
            print(f"DEBUG (Faiss wipe{tag}): Removed global {fname}")
    print(f"DEBUG (Faiss wipe{tag}): Done — {wiped} file(s) removed. Fresh indexes will build automatically.")
    return wiped


@app.route("/set_faiss_settings", methods=["POST"])
def set_faiss_settings():
    """
    Receives and saves new Faiss settings to the resonance config.

    POTATO-SAFE SAVE LOGIC:
    - Structural changes (embedding model swap, distance metric swap) →
      wipe stale indexes so they don't cause dimension/metric mismatches.
      Rebuilding happens lazily on the next chat message, NOT here.
    - Non-structural changes (quota, toggles, ctx length, session limit) →
      ONLY write the config file. Zero index work. Zero CPU spike.
      The user can trigger a manual rebuild via /api/rebuild_index if needed.
    """
    data = request.json
    global RESONANCE_SETTINGS, EMBEDDING_CTX_LENGTH, EMBEDDING_MODEL

    save_needed = False
    structural_change = False  # tracks if indexes must be wiped

    # ------------------------------------------------------------------ #
    # 1. EMBEDDING MODEL SWITCH  (structural — different vector dimensions)
    # ------------------------------------------------------------------ #
    if "embedding_model" in data:
        old_model = RESONANCE_SETTINGS.get("embedding_model", "nomic")
        new_model = str(data["embedding_model"])
        RESONANCE_SETTINGS["embedding_model"] = new_model
        save_needed = True

        if old_model != new_model:
            structural_change = True
            initialize_embedding_model(new_model)
            EMBEDDING_CTX_LENGTH = EMBEDDING_MODEL.max_seq_length if EMBEDDING_MODEL else None
            print(f"DEBUG (Faiss): Model switched {old_model} → {new_model}. Stale indexes will be wiped.")

    # ------------------------------------------------------------------ #
    # 2. CONTEXT LENGTH  (non-structural — same model, just token window)
    # ------------------------------------------------------------------ #
    if "embedding_ctx_length" in data:
        try:
            requested_ctx = int(data["embedding_ctx_length"])
            current_model = RESONANCE_SETTINGS.get("embedding_model", "nomic")
            if current_model in HIGH_CTX_MODELS and EMBEDDING_MODEL is not None:
                EMBEDDING_MODEL.max_seq_length = requested_ctx
                EMBEDDING_CTX_LENGTH = requested_ctx
                RESONANCE_SETTINGS["embedding_ctx_length"] = requested_ctx
                save_needed = True
                print(f"DEBUG: Embedding context length updated to {requested_ctx} tokens.")
            else:
                print(f"DEBUG: ctx_length change ignored — model '{current_model}' is not a high-ctx model.")
        except (ValueError, TypeError):
            return jsonify({"error": "Invalid value for embedding_ctx_length, expected an integer"}), 400

    # ------------------------------------------------------------------ #
    # 3. DISTANCE METRIC SWITCH  (structural — L2 vs cosine scores are
    #    incomparable, so existing vectors would produce garbage results)
    # ------------------------------------------------------------------ #
    if "faiss_distance_metric" in data:
        new_metric = str(data["faiss_distance_metric"])
        if new_metric in ("l2", "cosine"):
            old_metric = RESONANCE_SETTINGS.get("faiss_distance_metric", "l2")
            RESONANCE_SETTINGS["faiss_distance_metric"] = new_metric
            save_needed = True
            if old_metric != new_metric:
                structural_change = True
                print(f"DEBUG (Faiss): Distance metric changed {old_metric} → {new_metric}. Stale indexes will be wiped.")
        else:
            return jsonify({"error": "Invalid value for faiss_distance_metric, expected 'l2' or 'cosine'"}), 400

    # ------------------------------------------------------------------ #
    # 4. NON-STRUCTURAL TOGGLES / SCALARS  (config-only, zero CPU cost)
    # ------------------------------------------------------------------ #
    if "faiss_permanent_indexing" in data:
        try:
            RESONANCE_SETTINGS["faiss_permanent_indexing"] = bool(data["faiss_permanent_indexing"])
            save_needed = True
        except (ValueError, TypeError):
            return jsonify({"error": "Invalid value type for faiss_permanent_indexing, expected a boolean"}), 400

    if "faiss_indexing_quota" in data:
        try:
            RESONANCE_SETTINGS["faiss_indexing_quota"] = int(data["faiss_indexing_quota"])
            save_needed = True
        except (ValueError, TypeError):
            return jsonify({"error": "Invalid value type for faiss_indexing_quota, expected an integer"}), 400

    if "always_index_messages" in data:
        RESONANCE_SETTINGS["always_index_messages"] = bool(data["always_index_messages"])
        save_needed = True
        print(f"DEBUG: always_index_messages set to {RESONANCE_SETTINGS['always_index_messages']}")

    # ------------------------------------------------------------------ #
    # 5b. EMBEDDING PREFIX MODE  (structural when model supports asymmetric)
    #     Changing the mode on nomic/jina means old vectors were encoded
    #     without the prefix — those vectors are now mismatched against
    #     queries that DO use the prefix (or vice versa).  Wipe + rebuild.
    # ------------------------------------------------------------------ #
    ASYMMETRIC_MODELS = {"nomic", "jina-small", "jina-base"}

    if "embedding_prefix_mode" in data:
        new_prefix_mode = str(data["embedding_prefix_mode"]).strip().lower()
        if new_prefix_mode not in ("auto", "off"):
            return jsonify({"error": "Invalid value for embedding_prefix_mode, expected 'auto' or 'off'"}), 400
        old_prefix_mode = RESONANCE_SETTINGS.get("embedding_prefix_mode", "auto")
        RESONANCE_SETTINGS["embedding_prefix_mode"] = new_prefix_mode
        save_needed = True
        if old_prefix_mode != new_prefix_mode:
            current_model = RESONANCE_SETTINGS.get("embedding_model", "nomic")
            if current_model in ASYMMETRIC_MODELS:
                structural_change = True
                print(f"DEBUG (Faiss): Prefix mode changed {old_prefix_mode} → {new_prefix_mode} "
                      f"on asymmetric model '{current_model}'. Stale indexes will be wiped.")

    # ------------------------------------------------------------------ #
    # 5. GLOBAL MEMORY TOGGLE / SESSION LIMIT
    #    Turning global ON for the first time triggers a background rebuild
    #    (it has no index yet, so this isn't a surprise rebuild).
    #    Changing session_limit does NOT auto-rebuild — use manual rebuild.
    # ------------------------------------------------------------------ #
    global_was_off = not RESONANCE_SETTINGS.get("global_memory_enabled", False)

    if "global_memory_enabled" in data:
        RESONANCE_SETTINGS["global_memory_enabled"] = bool(data["global_memory_enabled"])
        save_needed = True

    if "global_memory_session_limit" in data:
        RESONANCE_SETTINGS["global_memory_session_limit"] = int(data["global_memory_session_limit"])
        save_needed = True
        # Do NOT auto-rebuild here — user can use manual rebuild button.
        print("DEBUG (Global Index): global_memory_session_limit changed. "
              "Use Manual Rebuild to apply to existing index.")

    # ------------------------------------------------------------------ #
    # 6. APPLY STRUCTURAL CHANGES — wipe stale indexes now (no rebuild)
    #    Rebuild happens lazily on next message, or via manual rebuild.
    # ------------------------------------------------------------------ #
    if structural_change:
        _wipe_all_faiss_indexes(reason="structural change")

    # ------------------------------------------------------------------ #
    # 7. SAVE CONFIG + handle first-time global enable
    # ------------------------------------------------------------------ #
    if save_needed:
        save_resonance_settings()
        print(f"DEBUG: Faiss settings saved (structural_change={structural_change}).")

        # Only trigger a background global rebuild if global was just turned ON
        # and there is no index yet (first enable, not accidental re-save).
        if (RESONANCE_SETTINGS.get("global_memory_enabled", False)
                and global_was_off
                and not os.path.exists(get_global_faiss_filepath())):
            print("DEBUG (Global Index): global_memory_enabled turned ON for first time — "
                  "launching background rebuild.")
            threading.Thread(
                target=rebuild_global_index,
                kwargs={"force": True},
                daemon=True
            ).start()
            return jsonify({
                "message": "Faiss settings saved. Global index is being built in the background.",
                "structural_change": structural_change
            })

        action_hint = " Indexes wiped — will rebuild automatically on next message." if structural_change else ""
        return jsonify({
            "message": f"Faiss settings saved.{action_hint}",
            "structural_change": structural_change
        })

    return jsonify({"message": "No Faiss settings provided to update.", "structural_change": False})


@app.route("/api/abort", methods=["POST"])
def abort_generation():
    """Forwards an abort request to KoboldCPP to stop ongoing generation."""
    try:
        llm_base = APP_SETTINGS.get("llm_api_endpoint", "http://127.0.0.1:5001/v1/chat/completions")
        from urllib.parse import urlparse
        parsed = urlparse(llm_base)
        base_url = f"{parsed.scheme}://{parsed.netloc}"
        abort_url = f"{base_url}/api/extra/abort"
        response = requests.post(abort_url, json={}, timeout=5)
        return jsonify({"success": True}), 200
    except Exception as e:
        print(f"DEBUG (Abort): Failed to abort — {e}")
        return jsonify({"success": False, "error": str(e)}), 500    


@app.route("/clear_global_memory", methods=["POST"])
def clear_global_memory():
    """Deletes all files inside the global_memory folder, resetting the vector index."""
    try:
        deleted = []
        if os.path.exists(GLOBAL_MEMORY_DIR):
            for fname in os.listdir(GLOBAL_MEMORY_DIR):
                fpath = os.path.join(GLOBAL_MEMORY_DIR, fname)
                if os.path.isfile(fpath):
                    os.remove(fpath)
                    deleted.append(fname)
        print(f"DEBUG (Global Memory): Cleared {len(deleted)} file(s): {deleted}")
        return jsonify({"message": f"Global memory cleared. Deleted {len(deleted)} file(s).", "deleted": deleted})
    except Exception as e:
        print(f"ERROR (Global Memory): Failed to clear: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/rebuild_index", methods=["POST"])
def api_rebuild_index():
    """
    Manual index rebuild endpoint.

    Behaviour:
    - scope="local"  → rebuilds only the current session's FAISS index.
                       Useful after corruption or after importing old messages.
    - scope="global" → rebuilds the unified global index across all sessions
                       (only meaningful when global_memory_enabled is ON).
    - scope="all"    → rebuilds local first, then global (if enabled).

    The rebuild runs in a background daemon thread so the HTTP response
    returns immediately — the frontend should poll /api/rebuild_status
    or just wait a sensible amount of time before trying a recall.

    The endpoint is intentionally idempotent: calling it multiple times is
    safe because check_and_update_faiss_index and rebuild_global_index both
    have their own staleness / debounce guards.
    """
    data = request.json or {}
    scope = data.get("scope", "all").strip().lower()

    if scope not in ("local", "global", "all"):
        return jsonify({"error": "Invalid scope. Use 'local', 'global', or 'all'."}), 400

    if scope == "global" and not RESONANCE_SETTINGS.get("global_memory_enabled", False):
        return jsonify({
            "error": "Global memory is not enabled. Turn it on in Vector Indexing settings first."
        }), 400

    session_id = get_current_session_id()

    def _do_rebuild():
        global _rebuild_running
        with _rebuild_status_lock:
            _rebuild_running = True
        try:
            if scope in ("local", "all"):
                print(f"DEBUG (Manual Rebuild): Rebuilding local index for session '{session_id}'...")
                # FIX BUG 30: Load memory directly from the session file captured at request
                # time, not via load_memory() which re-reads get_current_session_id() at
                # thread runtime. If the user switched sessions before the thread ran,
                # load_memory() would load the wrong session's data while the index files
                # being wiped/rebuilt belong to session_id.
                _session_filepath = os.path.join(SESSION_DIR, session_id, "chat_memory.json")
                try:
                    with open(_session_filepath, "r", encoding="utf-8") as _f:
                        memory = json.load(_f)
                except Exception:
                    memory = []
                active_memory = ghost_memory_if_needed(memory)
                # Force a fresh build by wiping just this session's index files first
                for fname in ["faiss_index.idx", "faiss_metadata.json"]:
                    fpath = os.path.join(SESSION_DIR, session_id, fname)
                    if os.path.exists(fpath):
                        os.remove(fpath)
                        print(f"DEBUG (Manual Rebuild): Wiped stale local {fname}")
                check_and_update_faiss_index(memory, active_memory, session_id)
                print(f"DEBUG (Manual Rebuild): Local index rebuild complete.")

            if scope in ("global", "all"):
                if RESONANCE_SETTINGS.get("global_memory_enabled", False):
                    print("DEBUG (Manual Rebuild): Rebuilding global index...")
                    rebuild_global_index(force=True)
                    print("DEBUG (Manual Rebuild): Global index rebuild complete.")
                else:
                    print("DEBUG (Manual Rebuild): Global memory disabled, skipping global rebuild.")
        except Exception as e:
            print(f"ERROR (Manual Rebuild): Rebuild failed: {e}")
            traceback.print_exc()
        finally:
            with _rebuild_status_lock:
                _rebuild_running = False

    thread = threading.Thread(target=_do_rebuild, daemon=True)
    thread.start()

    scope_desc = {
        "local": f"Local index for session '{session_id}'",
        "global": "Global index",
        "all": f"Local index for session '{session_id}' + Global index"
    }[scope]

    return jsonify({
        "message": f"{scope_desc} rebuild started in the background. "
                   "This may take a moment depending on history size.",
        "scope": scope,
        "session_id": session_id
    })


@app.route("/api/rebuild_status", methods=["GET"])
def api_rebuild_status():
    """
    Returns the current status of the manual rebuild operation.
    Frontend polls this endpoint to know when the rebuild completes.
    """
    with _rebuild_status_lock:
        running = _rebuild_running
    
    return jsonify({
        "running": running,
        "message": "Rebuilding index..." if running else "Rebuild complete."
    })


@app.route("/api/rebuild_cancel", methods=["POST"])
def api_rebuild_cancel():
    """
    Stub endpoint for canceling an in-progress rebuild.
    
    NOTE: Actual cancellation is not implemented yet. The rebuild runs in a
    daemon thread with no interrupt mechanism. This endpoint exists so the
    frontend "Cancel" button doesn't throw 404 errors.
    
    Future implementation could use threading.Event() to signal the rebuild
    thread to stop early.
    """
    return jsonify({
        "message": "Cancel not implemented yet. Rebuild will complete in background."
    }), 200




# =============================================================================
# --- PERSONA AVATAR ENDPOINTS ---
# =============================================================================

@app.route("/upload_persona_avatar", methods=["POST"])
def upload_persona_avatar():
    """Uploads and saves an avatar image for a specific persona."""
    data = request.json
    persona_name = data.get("persona_name", "").strip()
    image_data = data.get("image")  # base64 string

    if not persona_name or not image_data:
        return jsonify({"error": "persona_name and image are required."}), 400
    if persona_name not in PERSONAS:
        return jsonify({"error": f"Persona '{persona_name}' not found."}), 404

    try:
        if "base64," in image_data:
            image_data = image_data.split("base64,")[1]
        safe_name = re.sub(r'[^\w]', '_', persona_name)
        # FIX BUG 25: Include a short hash of the original name so two personas
        # whose names map to the same safe_name don't clobber each other's avatar.
        _name_hash = format(hash(persona_name) & 0xFFFFFF, '06x')
        filename = f"persona_{safe_name}_{_name_hash}.png"
        filepath = os.path.join(PERSONA_IMAGES_DIR, filename)
        with open(filepath, "wb") as f:
            f.write(base64.b64decode(image_data))

        # Save filename reference into the persona dict
        personas_copy = PERSONAS.copy()
        entry = personas_copy.get(persona_name, {})
        if isinstance(entry, str):
            entry = {'prompt': entry, 'avatar_image': None}
        entry['avatar_image'] = filename
        personas_copy[persona_name] = entry
        save_all_personas(personas_copy)

        print(f"DEBUG: Persona avatar saved for '{persona_name}': {filename}")
        return jsonify({"status": "success", "filename": filename})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/get_persona_avatar/<persona_name>")
def get_persona_avatar(persona_name):
    """Serves the avatar image for a given persona."""
    entry = PERSONAS.get(persona_name)
    if not entry:
        return jsonify({"error": "Persona not found"}), 404
    filename = entry.get('avatar_image') if isinstance(entry, dict) else None
    if not filename:
        return jsonify({"avatar_image": None})
    return send_from_directory(PERSONA_IMAGES_DIR, filename)


@app.route("/get_persona_avatar_info/<persona_name>")
def get_persona_avatar_info(persona_name):
    """Returns avatar metadata (filename) for a persona."""
    entry = PERSONAS.get(persona_name)
    if not entry:
        return jsonify({"avatar_image": None, "avatar_url": None})
    filename = entry.get('avatar_image') if isinstance(entry, dict) else None
    url = f"/get_persona_avatar/{persona_name}" if filename else None
    return jsonify({"avatar_image": filename, "avatar_url": url})


@app.route("/delete_persona_avatar", methods=["POST"])
def delete_persona_avatar():
    """Removes the avatar image for a persona."""
    data = request.json
    persona_name = data.get("persona_name", "").strip()
    if not persona_name or persona_name not in PERSONAS:
        return jsonify({"error": "Persona not found."}), 404
    personas_copy = PERSONAS.copy()
    entry = personas_copy.get(persona_name, {})
    if isinstance(entry, dict):
        old_file = entry.get('avatar_image')
        if old_file:
            fpath = os.path.join(PERSONA_IMAGES_DIR, old_file)
            if os.path.exists(fpath):
                os.remove(fpath)
        entry['avatar_image'] = None
        personas_copy[persona_name] = entry
        save_all_personas(personas_copy)
    return jsonify({"message": "Persona avatar removed."})


# =============================================================================
# --- USER LOADOUT ENDPOINTS ---
# =============================================================================

@app.route("/get_user_loadouts", methods=["GET"])
def get_user_loadouts():
    """Returns all user loadouts and the currently active one."""
    return jsonify({
        "loadouts": USER_LOADOUTS,
        "active": get_current_user_loadout_name()
    })


@app.route("/set_user_loadout", methods=["POST"])
def set_user_loadout():
    """Creates or updates a user loadout."""
    data = request.json
    loadout_name = data.get("loadout_name", "").strip()
    if not loadout_name:
        return jsonify({"error": "loadout_name is required."}), 400
    loadouts_copy = USER_LOADOUTS.copy()
    existing = loadouts_copy.get(loadout_name, {})
    existing['display_name'] = data.get("display_name", existing.get("display_name", "User"))
    existing['avatar_emoji'] = data.get("avatar_emoji", existing.get("avatar_emoji", "❄️"))
    existing['persona_prompt'] = data.get("persona_prompt", existing.get("persona_prompt", ""))
    if 'avatar_image' not in existing:
        existing['avatar_image'] = None
    loadouts_copy[loadout_name] = existing
    save_all_user_loadouts(loadouts_copy)
    return jsonify({"message": f"Loadout '{loadout_name}' saved."})


@app.route("/switch_user_loadout", methods=["POST"])
def switch_user_loadout():
    """Switches the active user loadout."""
    data = request.json
    loadout_name = data.get("loadout_name", "").strip()
    if not loadout_name or loadout_name not in USER_LOADOUTS:
        return jsonify({"error": "Loadout not found."}), 404
    set_current_user_loadout_name(loadout_name)
    loadout = USER_LOADOUTS[loadout_name]
    NAME_SETTINGS["user_name"] = loadout.get("display_name", "")
    save_name_settings()
    print(f"DEBUG: Switched to user loadout '{loadout_name}'")
    return jsonify({"message": f"Switched to '{loadout_name}'.", "loadout": loadout})


@app.route("/delete_user_loadout", methods=["POST"])
def delete_user_loadout():
    """Deletes a user loadout. Cannot delete the last one."""
    data = request.json
    loadout_name = data.get("loadout_name", "").strip()
    if loadout_name == get_current_user_loadout_name():
        return jsonify({"error": "Cannot delete the active loadout. Switch first."}), 400
    if loadout_name not in USER_LOADOUTS:
        return jsonify({"error": "Loadout not found."}), 404
    if len(USER_LOADOUTS) <= 1:
        return jsonify({"error": "Cannot delete the last loadout."}), 400
    loadouts_copy = USER_LOADOUTS.copy()
    old_img = loadouts_copy[loadout_name].get('avatar_image')
    if old_img:
        fpath = os.path.join(USER_LOADOUT_IMAGES_DIR, old_img)
        if os.path.exists(fpath):
            os.remove(fpath)
    del loadouts_copy[loadout_name]
    save_all_user_loadouts(loadouts_copy)
    return jsonify({"message": f"Loadout '{loadout_name}' deleted."})


@app.route("/upload_user_loadout_avatar", methods=["POST"])
def upload_user_loadout_avatar():
    """Uploads and saves an avatar image for a user loadout."""
    data = request.json
    loadout_name = data.get("loadout_name", "").strip()
    image_data = data.get("image")
    if not loadout_name or not image_data:
        return jsonify({"error": "loadout_name and image are required."}), 400
    if loadout_name not in USER_LOADOUTS:
        return jsonify({"error": f"Loadout '{loadout_name}' not found."}), 404
    try:
        if "base64," in image_data:
            image_data = image_data.split("base64,")[1]
        safe_name = re.sub(r'[^\w]', '_', loadout_name)
        # FIX BUG 25: Include a short hash so same-safe_name loadouts don't clobber each other.
        _name_hash = format(hash(loadout_name) & 0xFFFFFF, '06x')
        filename = f"user_{safe_name}_{_name_hash}.png"
        filepath = os.path.join(USER_LOADOUT_IMAGES_DIR, filename)
        with open(filepath, "wb") as f:
            f.write(base64.b64decode(image_data))
        loadouts_copy = USER_LOADOUTS.copy()
        loadouts_copy[loadout_name]['avatar_image'] = filename
        save_all_user_loadouts(loadouts_copy)
        return jsonify({"status": "success", "filename": filename})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/get_user_loadout_avatar/<loadout_name>")
def get_user_loadout_avatar(loadout_name):
    """Serves the avatar image for a user loadout."""
    entry = USER_LOADOUTS.get(loadout_name)
    if not entry:
        return jsonify({"error": "Loadout not found"}), 404
    filename = entry.get('avatar_image')
    if not filename:
        return jsonify({"avatar_image": None})
    return send_from_directory(USER_LOADOUT_IMAGES_DIR, filename)


@app.route("/delete_user_loadout_avatar", methods=["POST"])
def delete_user_loadout_avatar():
    """Removes the avatar image for a user loadout."""
    data = request.json
    loadout_name = data.get("loadout_name", "").strip()
    if not loadout_name or loadout_name not in USER_LOADOUTS:
        return jsonify({"error": "Loadout not found."}), 404
    loadouts_copy = USER_LOADOUTS.copy()
    old_file = loadouts_copy[loadout_name].get('avatar_image')
    if old_file:
        fpath = os.path.join(USER_LOADOUT_IMAGES_DIR, old_file)
        if os.path.exists(fpath):
            os.remove(fpath)
    loadouts_copy[loadout_name]['avatar_image'] = None
    save_all_user_loadouts(loadouts_copy)
    return jsonify({"message": "User loadout avatar removed."})


@app.route("/get_active_loadout_info", methods=["GET"])
def get_active_loadout_info():
    """Returns full info about the active user loadout for the frontend."""
    name = get_current_user_loadout_name()
    loadout = get_current_user_loadout()
    avatar_url = f"/get_user_loadout_avatar/{name}" if loadout.get('avatar_image') else None
    return jsonify({
        "loadout_name": name,
        "display_name": loadout.get('display_name', 'User'),
        "avatar_emoji": loadout.get('avatar_emoji', '❄️'),
        "avatar_image": loadout.get('avatar_image'),
        "avatar_url": avatar_url,
        "persona_prompt": loadout.get('persona_prompt', '')
    })


# =============================================================================
# --- RECALL SEARCH ENDPOINT (NEW) ---
# =============================================================================

@app.route("/api/search_memory", methods=["POST"])
def search_memory():
    """
    Search memories using FAISS vector similarity + optional reranking + optional sanity check.
    
    Three-stage retrieval flow:
    1. FAISS fast retrieval (get top_k × expansion_factor candidates)
    2. Content pre-filter — cheap cosine recheck prunes pool before reranker [optional]
    3. Reranker — precise cross-encoder scoring, final authoritative ranking [optional]
    
    Request JSON:
    {
        "query": str,               # Required: what to search for
        "session_id": str,          # Optional: defaults to current session
        "top_k": int,               # Optional: how many results (default 5)
        "score_threshold": float    # Optional: minimum score filter
    }
    
    Response JSON:
    {
        "results": [
            {
                "content": str,
                "role": str,
                "timestamp": str,
                "faiss_score": float,
                "reranker_score": float,    # Only if reranker enabled
                "intent_score": float,      # Only if sanity check enabled
                "sanity_score": float,      # Only if sanity check enabled
                "score": float              # Final score used
            }
        ]
    }
    """
    data = request.json
    query = data.get("query")
    session_id = data.get("session_id", get_current_session_id())
    top_k = data.get("top_k", 5)
    score_threshold = data.get("score_threshold")
    
    if not query:
        return jsonify({"error": "Query parameter is required"}), 400
    
    try:
        # FIX BUG: Use helper paths so this endpoint uses the same index files
        # as the rest of the app, not a hardcoded path.
        index_path    = get_faiss_index_filepath(session_id)
        metadata_path = get_faiss_metadata_filepath(session_id)

        if not os.path.exists(index_path) or not os.path.exists(metadata_path):
            print(f"DEBUG (Search): No FAISS index found for {session_id}.")
            return jsonify({"results": []})

        # Load index + metadata
        # FIX BUG: Results must be resolved against the METADATA list, NOT active_memory.
        # FAISS indices map into the metadata array (ghosted messages). Using active_memory
        # here caused wrong/crashed lookups because it's a completely different array.
        try:
            index = _safe_faiss_read(index_path)
        except Exception as e:
            print(f"ERROR (Search): Failed to load index: {e}")
            return jsonify({"error": f"Failed to load index: {str(e)}"}), 500

        indexed_messages, index_model = load_faiss_metadata(metadata_path)
        if not indexed_messages:
            return jsonify({"results": []})

        # Validate model match
        current_model = RESONANCE_SETTINGS.get("embedding_model", "nomic")
        if index_model is not None and index_model != current_model:
            print(f"DEBUG (Search): Index model mismatch ('{index_model}' vs '{current_model}'). Returning empty.")
            return jsonify({"results": []})
        if index.d != EMBEDDING_DIM:
            print(f"DEBUG (Search): Index dimension mismatch ({index.d} vs {EMBEDDING_DIM}). Returning empty.")
            return jsonify({"results": []})

        # Generate query embedding
        query_emb = embed_query([query], convert_to_tensor=False)
        query_np = np.array(query_emb).astype('float32')
        query_np = prepare_vectors(query_np)

        # --- STAGE 3 PRE-GATE: Intent check BEFORE any FAISS work ---
        # /api/search_memory is only called by the explicit [RECALL:] tool, so intent
        # is already confirmed by the LLM invoking the tool. Skip the gate by default.
        # If you ever call this endpoint from passive RAG, pass force_search=False in body.
        sanity_enabled = SANITY_CHECK_SETTINGS.get("sanity_check_enabled", False)
        intent_enabled = SANITY_CHECK_SETTINGS.get("sanity_check_intent_enabled", True)
        force_search = data.get("force_search", True)  # Explicit tool calls skip intent gate
        if sanity_enabled and intent_enabled and not force_search:
            intent_threshold = SANITY_CHECK_SETTINGS.get("sanity_check_intent_threshold", 0.40)
            has_intent, intent_score = check_recall_intent(query, intent_threshold)
            if not has_intent:
                print(f"DEBUG (Sanity Pre-Gate/Search): No recall intent — returning empty (score: {intent_score:.3f})")
                return jsonify({"results": []})
            print(f"DEBUG (Sanity Pre-Gate/Search): Intent confirmed (score: {intent_score:.3f})")
        # --- END STAGE 3 PRE-GATE ---

        # --- STAGE 1: FAISS Fast Retrieval ---
        reranker_enabled = RESONANCE_SETTINGS.get("reranker_enabled", False)
        expansion_factor = RESONANCE_SETTINGS.get("reranker_expansion_factor", 3)

        # Retrieval count: reranker gets expansion, sanity check gets buffer, vanilla gets exact k.
        if reranker_enabled:
            retrieval_k = top_k * expansion_factor
            print(f"DEBUG (FAISS): Fetching {retrieval_k} candidates (reranker expansion)")
        elif sanity_enabled:
            buffer_mult = SANITY_CHECK_SETTINGS.get("sanity_check_buffer_multiplier", 2.0)
            retrieval_k = max(top_k, int(top_k * buffer_mult))
            print(f"DEBUG (FAISS): Fetching {retrieval_k} candidates (sanity buffer expansion)")
        else:
            retrieval_k = top_k
            print(f"DEBUG (FAISS): Fetching {retrieval_k} candidates (vanilla mode)")

        retrieval_k = min(retrieval_k, index.ntotal)
        if retrieval_k == 0:
            return jsonify({"results": []})

        distances, indices = index.search(query_np, retrieval_k)

        # Build candidate list — resolved from metadata, NOT active_memory
        candidates = []
        metric = RESONANCE_SETTINGS.get("faiss_distance_metric", "l2")

        for i, idx in enumerate(indices[0]):
            if idx < 0 or idx >= len(indexed_messages):
                continue

            memory_entry = indexed_messages[idx]
            distance = float(distances[0][i])

            # Convert distance to similarity score
            if metric == "cosine":
                similarity = distance  # Already a similarity (higher = better)
            else:  # L2
                similarity = 1.0 / (1.0 + distance)  # Convert to 0-1 range

            candidates.append({
                "content": memory_entry.get("content", ""),
                "role": memory_entry.get("role", ""),
                "timestamp": memory_entry.get("timestamp", ""),
                "faiss_score": similarity,
                "score": similarity  # Will be overridden if reranker enabled
            })
        
        # --- STAGE 2: Content Pre-Filter (cheap — prunes pool before expensive reranker) ---
        # Intent was already checked in the pre-gate above — skip Phase 1 here.
        if sanity_enabled and candidates:
            print(f"DEBUG (Sanity/Search): ENABLED - Running content pre-filter on {len(candidates)} candidates...")
            candidates = sanity_check_filter(query, candidates, skip_intent_check=True)
            print(f"DEBUG (Sanity/Search): Content pre-filter passed {len(candidates)} candidates → feeding to Reranker")
        elif sanity_enabled and not candidates:
            print(f"DEBUG (Sanity/Search): ENABLED but no candidates from Stage 1")
        else:
            print(f"DEBUG (Sanity/Search): DISABLED - Skipping content pre-filter")
        # --- END STAGE 2 ---

        # --- STAGE 3: Reranker — final authoritative scoring (runs on pruned pool) ---
        if reranker_enabled and RERANKER_MODEL and candidates:
            print(f"DEBUG (Reranker): ENABLED - Reranking {len(candidates)} candidates (post content-filter)...")
            reranker_threshold = RESONANCE_SETTINGS.get("reranker_score_threshold", 0.0)
            batch_size = RESONANCE_SETTINGS.get("reranker_batch_size", 32)
            reranked = rerank_candidates(
                query=query,
                candidates=candidates,
                top_k=top_k,
                score_threshold=reranker_threshold,
                batch_size=batch_size
            )
            # Update final score to reranker score
            for item in reranked:
                item["score"] = item.get("reranker_score", item["faiss_score"])
            print(f"DEBUG (Reranker): Stage 3 returned {len(reranked)} results after reranking.")
            results = reranked
        else:
            # VANILLA MODE: Reranker disabled or not loaded - use pure FAISS scores
            if not reranker_enabled:
                print(f"DEBUG (Reranker): DISABLED - Using vanilla FAISS retrieval (top {top_k}).")
            elif not RERANKER_MODEL:
                print(f"WARNING (Reranker): ENABLED but model not loaded - falling back to FAISS.")
            results = candidates[:top_k]
        # --- END STAGE 3 ---
        
        # Apply threshold if specified (for FAISS-only mode)
        if score_threshold is not None and not reranker_enabled:
            results = [r for r in results if r["score"] >= score_threshold]
        
        return jsonify({"results": results})
        
    except Exception as e:
        print(f"ERROR (Search): {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# =============================================================================
# --- RERANKER SETTINGS API ROUTES ---
# =============================================================================

@app.route("/get_reranker_settings", methods=["GET"])
def get_reranker_settings():
    """Returns current reranker settings."""
    return jsonify({
        "reranker_enabled": RESONANCE_SETTINGS.get("reranker_enabled", False),
        "reranker_model": RESONANCE_SETTINGS.get("reranker_model", "ms-marco-mini-v2"),
        "reranker_expansion_factor": RESONANCE_SETTINGS.get("reranker_expansion_factor", 3),
        "reranker_ctx_length": RESONANCE_SETTINGS.get("reranker_ctx_length"),
        "reranker_score_threshold": RESONANCE_SETTINGS.get("reranker_score_threshold", 0.0),
        "reranker_batch_size": RESONANCE_SETTINGS.get("reranker_batch_size", 32)
    })


@app.route("/set_reranker_settings", methods=["POST"])
def set_reranker_settings():
    """
    Updates reranker settings and hot-reloads model if changed.
    
    Master Toggle Behavior:
    - OFF: Reranker completely bypassed, FAISS → Content Filter (if on) → done
    - ON: FAISS → Content Filter (pre-prune) → Reranker (final word)
    
    Request JSON:
    {
        "reranker_enabled": bool,
        "reranker_model": str,
        "reranker_expansion_factor": int,
        "reranker_ctx_length": int | None,
        "reranker_score_threshold": float,
        "reranker_batch_size": int
    }
    """
    data = request.json
    
    # Detect if model changed (requires reload)
    old_model = RESONANCE_SETTINGS.get("reranker_model", "ms-marco-mini-v2")
    new_model = data.get("reranker_model", old_model)
    old_ctx = RESONANCE_SETTINGS.get("reranker_ctx_length")
    # FIX BUG 3: Cast to int (or keep None) — JSON may deliver a string from
    # form inputs, which would later crash min(str, int) in the in-place patch path.
    _raw_ctx = data.get("reranker_ctx_length")
    try:
        new_ctx = int(_raw_ctx) if _raw_ctx is not None else None
    except (ValueError, TypeError):
        return jsonify({"error": "Invalid value for reranker_ctx_length, expected an integer or null"}), 400
    old_enabled = RESONANCE_SETTINGS.get("reranker_enabled", False)
    new_enabled = data.get("reranker_enabled", False)
    
    model_changed = new_model != old_model
    ctx_changed = new_ctx != old_ctx
    toggle_changed = new_enabled != old_enabled
    
    # Update settings
    RESONANCE_SETTINGS["reranker_enabled"] = new_enabled
    RESONANCE_SETTINGS["reranker_model"] = new_model
    RESONANCE_SETTINGS["reranker_expansion_factor"] = data.get("reranker_expansion_factor", 3)
    RESONANCE_SETTINGS["reranker_ctx_length"] = new_ctx
    RESONANCE_SETTINGS["reranker_score_threshold"] = data.get("reranker_score_threshold", 0.0)
    RESONANCE_SETTINGS["reranker_batch_size"] = data.get("reranker_batch_size", 32)
    
    # Save to disk
    save_resonance_settings()
    
    # Handle model loading based on toggle state
    if new_enabled:
        if model_changed or toggle_changed:
            # Full reload only when model name changed or toggled ON
            print(f"DEBUG (Reranker): ENABLED - Loading model '{new_model}'...")
            initialize_reranker_model(new_model, new_ctx)
            message = f"Reranker ENABLED with model '{new_model}'."
        elif ctx_changed and RERANKER_MODEL is not None and new_model in HIGH_CTX_RERANKERS:
            # In-place ctx patch — no reload, no double model in RAM 💀
            max_supported = HIGH_CTX_RERANKERS[new_model]
            capped_ctx = min(new_ctx, max_supported) if new_ctx else max_supported
            RERANKER_MODEL.max_length = capped_ctx
            RERANKER_CTX_LENGTH = capped_ctx
            print(f"DEBUG (Reranker): Context length updated in-place to {capped_ctx} tokens (no reload).")
            message = f"Reranker context updated to {capped_ctx} tokens."
        else:
            message = "Reranker settings saved (no reload needed)."
    else:
        if toggle_changed:
            print(f"DEBUG (Reranker): DISABLED - Switching to vanilla FAISS mode.")
            message = "Reranker DISABLED. Using vanilla FAISS retrieval."
        else:
            message = "Reranker settings saved (currently disabled)."
    
    return jsonify({
        "status": "success",
        "message": message,
        "model_reloaded": (model_changed or toggle_changed) and new_enabled,
        "ctx_patched_inplace": ctx_changed and not model_changed and new_enabled,
        "reranker_active": new_enabled
    })


# =============================================================================
# --- STAGE 3: SANITY CHECK SETTINGS API ROUTES ---
# =============================================================================

@app.route("/get_sanity_check_settings", methods=["GET"])
def get_sanity_check_settings():
    """Returns current Stage 3 sanity check settings."""
    return jsonify(SANITY_CHECK_SETTINGS)


@app.route("/set_sanity_check_settings", methods=["POST"])
def set_sanity_check_settings():
    """
    Updates Stage 3 sanity check settings.
    Reloads intent embeddings if recall phrases changed.
    
    Request JSON:
    {
        "sanity_check_enabled": bool,
        "sanity_check_intent_enabled": bool,
        "sanity_check_intent_threshold": float,
        "sanity_check_content_threshold": float,
        "sanity_check_recall_phrases": str,
        "sanity_check_buffer_multiplier": float,  # FAISS over-fetch multiplier when sanity ON, reranker OFF
        "sanity_check_show_scores": bool           # gate per-item score DEBUG prints
    }
    """
    global ZEROSHOT_MODEL, ZEROSHOT_CTX_LENGTH
    data = request.json

    # Detect if recall phrases changed (requires re-embedding)
    old_phrases = SANITY_CHECK_SETTINGS.get("sanity_check_recall_phrases", "")
    new_phrases = data.get("sanity_check_recall_phrases", old_phrases)
    phrases_changed = new_phrases != old_phrases

    # Detect if negative phrases changed (also requires re-embedding)
    old_neg_phrases = SANITY_CHECK_SETTINGS.get("sanity_check_negative_phrases", "")
    new_neg_phrases = data.get("sanity_check_negative_phrases", old_neg_phrases)
    neg_phrases_changed = new_neg_phrases != old_neg_phrases
    
    # Detect toggle changes
    old_enabled = SANITY_CHECK_SETTINGS.get("sanity_check_enabled", False)
    new_enabled = data.get("sanity_check_enabled", False)
    toggle_changed = new_enabled != old_enabled

    # --- ZEROSHOT DLC: detect relevant changes ---
    old_zs_enabled = SANITY_CHECK_SETTINGS.get("zeroshot_intent_enabled", False)
    new_zs_enabled = data.get("zeroshot_intent_enabled", old_zs_enabled)
    old_zs_model   = SANITY_CHECK_SETTINGS.get("zeroshot_model", "nli-minilm-l6")
    new_zs_model   = data.get("zeroshot_model", old_zs_model)
    old_zs_ctx     = SANITY_CHECK_SETTINGS.get("zeroshot_ctx_length", 512)
    new_zs_ctx     = int(data.get("zeroshot_ctx_length", old_zs_ctx))
    zs_toggle_changed = new_zs_enabled != old_zs_enabled
    zs_model_changed  = new_zs_model  != old_zs_model
    zs_ctx_changed    = new_zs_ctx    != old_zs_ctx
    
    # Update settings
    # Coerce zeroshot_hypotheses — UI may send as newline-separated string or list
    if "zeroshot_hypotheses" in data:
        raw = data["zeroshot_hypotheses"]
        if isinstance(raw, str):
            data["zeroshot_hypotheses"] = [h.strip() for h in raw.split("\n") if h.strip()]

    # Coerce advanced zeroshot settings with type safety
    if "zeroshot_score_floor" in data:
        try:
            data["zeroshot_score_floor"] = float(data["zeroshot_score_floor"])
        except (ValueError, TypeError):
            data.pop("zeroshot_score_floor")
    if "zeroshot_top_n_hypotheses" in data:
        try:
            data["zeroshot_top_n_hypotheses"] = int(data["zeroshot_top_n_hypotheses"])
        except (ValueError, TypeError):
            data.pop("zeroshot_top_n_hypotheses")
    # Never let the frontend overwrite model presets — they are server-side constants
    data.pop("zeroshot_model_presets", None)

    SANITY_CHECK_SETTINGS.update(data)
    
    # Save to disk
    save_sanity_check_settings()
    
    # --- ZEROSHOT DLC: model lifecycle ---
    zs_message = ""
    if new_enabled and new_zs_enabled:
        if zs_toggle_changed or zs_model_changed:
            print(f"DEBUG (Zeroshot): Loading model '{new_zs_model}'...")
            initialize_zeroshot_model(new_zs_model, new_zs_ctx)
            zs_message = f" Zeroshot model '{new_zs_model}' loaded."
        elif zs_ctx_changed and ZEROSHOT_MODEL is not None:
            ZEROSHOT_MODEL.max_length = new_zs_ctx
            ZEROSHOT_CTX_LENGTH = new_zs_ctx
            print(f"DEBUG (Zeroshot): ctx patched in-place to {new_zs_ctx}.")
            zs_message = f" Zeroshot ctx updated to {new_zs_ctx}."
    elif zs_toggle_changed and not new_zs_enabled:
        if ZEROSHOT_MODEL is not None:
            del ZEROSHOT_MODEL
            ZEROSHOT_MODEL = None
            gc.collect()
            print("DEBUG (Zeroshot): DLC disabled — model unloaded.")
        zs_message = " Zeroshot DLC disabled — vanilla phrase embeddings active."

    # Handle vanilla intent embeddings reload (only needed when zeroshot is OFF)
    if not new_zs_enabled:
        if (phrases_changed or neg_phrases_changed) and new_enabled:
            print(f"DEBUG (Sanity): Phrases changed - re-embedding...")
            initialize_recall_intent_embeddings()
            message = "Sanity check settings saved. Intent phrases re-embedded."
        elif zs_toggle_changed and new_enabled:
            # Zeroshot just turned OFF — vanilla path has no embeddings, reload them now
            print(f"DEBUG (Sanity): Zeroshot DLC disabled — re-embedding vanilla intent phrases...")
            initialize_recall_intent_embeddings()
            message = "Zeroshot DLC disabled. Vanilla phrase embeddings reloaded."
        elif toggle_changed and new_enabled:
            print(f"DEBUG (Sanity): ENABLED - Initializing intent embeddings...")
            initialize_recall_intent_embeddings()
            message = "Sanity check ENABLED."
        elif toggle_changed and not new_enabled:
            print(f"DEBUG (Sanity): DISABLED.")
            message = "Sanity check DISABLED."
        else:
            message = "Sanity check settings saved."
    else:
        message = "Sanity check settings saved (Zeroshot DLC active — phrase embeddings skipped)."

    message += zs_message
    
    return jsonify({
        "status": "success",
        "message": message,
        "sanity_check_active": new_enabled,
        "intent_check_active": SANITY_CHECK_SETTINGS.get("sanity_check_intent_enabled", True),
        "phrases_reloaded": phrases_changed or neg_phrases_changed,
        "zeroshot_active": new_enabled and new_zs_enabled,
        "zeroshot_model_loaded": ZEROSHOT_MODEL is not None,
    })


# =============================================================================
# --- ZEROSHOT ADVANCED: PRESET ENDPOINT ---
# =============================================================================

@app.route("/api/zeroshot_preset", methods=["GET", "POST"])
def api_zeroshot_preset():
    """
    GET  → returns all model presets so the UI can populate a "Load Preset" button.
    POST → applies a named preset to the live SANITY_CHECK_SETTINGS.
           Body: { "model": "deberta-base" }
           Also accepts { "reset": true } to restore all advanced settings to defaults.
    """
    presets = SANITY_CHECK_SETTINGS.get("zeroshot_model_presets", {
        "deberta-base":    {"threshold": 0.80, "aggregation": "avg", "score_floor": 0.10},
        "deberta-xsmall":  {"threshold": 0.65, "aggregation": "max", "score_floor": 0.10},
        "nli-minilm-l6":   {"threshold": 0.75, "aggregation": "max", "score_floor": 0.10},
        "distilbert-mnli": {"threshold": 0.50, "aggregation": "max", "score_floor": 0.05},
        "bart-large-mnli": {"threshold": 0.80, "aggregation": "avg", "score_floor": 0.10},
    })

    if request.method == "GET":
        return jsonify({
            "presets": presets,
            "current_model": SANITY_CHECK_SETTINGS.get("zeroshot_model", "nli-minilm-l6")
        })

    # POST
    data = request.json or {}

    # Reset all advanced settings back to defaults
    if data.get("reset"):
        SANITY_CHECK_SETTINGS["zeroshot_score_floor"]       = 0.10
        SANITY_CHECK_SETTINGS["zeroshot_top_n_hypotheses"]  = 0
        SANITY_CHECK_SETTINGS["zeroshot_entailment_threshold"] = 0.70
        SANITY_CHECK_SETTINGS["zeroshot_aggregation"]       = "max"
        SANITY_CHECK_SETTINGS["zeroshot_threshold_enabled"] = True
        save_sanity_check_settings()
        print("DEBUG (Zeroshot Preset): Reset all advanced settings to defaults.")
        return jsonify({
            "message": "Advanced zeroshot settings reset to defaults.",
            "applied": {
                "threshold": 0.70,
                "aggregation": "max",
                "score_floor": 0.10,
                "top_n_hypotheses": 0
            }
        })

    # Apply named preset
    model_name = data.get("model", "").strip()
    if not model_name:
        return jsonify({"error": "Provide 'model' name or 'reset': true"}), 400

    preset = presets.get(model_name)
    if not preset:
        return jsonify({
            "error": f"No preset for '{model_name}'. Available: {list(presets.keys())}"
        }), 404

    SANITY_CHECK_SETTINGS["zeroshot_entailment_threshold"] = preset["threshold"]
    SANITY_CHECK_SETTINGS["zeroshot_aggregation"]          = preset["aggregation"]
    SANITY_CHECK_SETTINGS["zeroshot_score_floor"]          = preset["score_floor"]
    save_sanity_check_settings()

    print(f"DEBUG (Zeroshot Preset): Applied preset for '{model_name}': {preset}")
    return jsonify({
        "message": f"Preset for '{model_name}' applied.",
        "applied": preset
    })

# =============================================================================
# --- END ZEROSHOT ADVANCED PRESET ENDPOINT ---
# =============================================================================


if __name__ == "__main__":
    # Load all settings at startup
    load_app_settings()
    load_temperature_settings()
    load_token_settings()
    load_resonance_settings()
    load_sampler_settings()
    load_search_settings()
    load_name_settings()
    load_appearance_settings()
    load_streaming_settings()
    load_sanity_check_settings()  # NEW: Load Stage 3 settings
    initialize_embedding_model(RESONANCE_SETTINGS.get("embedding_model", "nomic"), RESONANCE_SETTINGS.get("embedding_ctx_length", None))

    # Initialize recall intent embeddings for Stage 3 (if enabled)
    if SANITY_CHECK_SETTINGS.get("sanity_check_enabled", False):
        if SANITY_CHECK_SETTINGS.get("zeroshot_intent_enabled", False):
            initialize_zeroshot_model(
                SANITY_CHECK_SETTINGS.get("zeroshot_model", "nli-minilm-l6"),
                SANITY_CHECK_SETTINGS.get("zeroshot_ctx_length", 512)
            )
        else:
            initialize_recall_intent_embeddings()

    # Initialize reranker if enabled
    if RESONANCE_SETTINGS.get("reranker_enabled", False):
        initialize_reranker_model(
            RESONANCE_SETTINGS.get("reranker_model", "ms-marco-mini-v2"),
            RESONANCE_SETTINGS.get("reranker_ctx_length")
        )

    # Initialize personas and set the current active persona
    initialize_personas()
    # Initialize user loadouts
    initialize_user_loadouts()
    # Initialize API profiles
    initialize_api_profiles()
    # Ensure a current persona is set at startup
    if not os.path.exists(CURRENT_SYSTEM_PROMPT_NAME_FILE):
        # Default to Vela if no current persona file exists
        set_current_persona_name("Vela")

    # Load the initial system prompt to set _current_active_character_name
    get_system_prompt()

    # load_character_locations()  # Load locations at application startup
    os.makedirs(SESSION_DIR, exist_ok=True)
    os.makedirs(GLOBAL_MEMORY_DIR, exist_ok=True)
    # if not os.path.exists(RP_MODE_TOGGLE_FILE):  # NEW: Initialize RP mode toggle
    #     set_rp_mode_state(True)  # Default to ON

    current_session = get_current_session_id()
    if not os.path.exists(get_session_filepath(current_session)):
        print(f"DEBUG: Initializing default session: {current_session}")
        set_current_session_id(current_session)
        save_memory([])
    
    # --- NEW: AUTO-INDEX ON APP START ---
    print(f"DEBUG (Faiss): App started. Checking index for {current_session}.")
    memory = load_memory()
    active_memory = ghost_memory_if_needed(memory)
    check_and_update_faiss_index(memory, active_memory, current_session)
    # --- END NEW ---
    
    print("Application started.")
    
    # --- FALLBACK 2: Changed server host to 0.0.0.0 ---
    # NOTE: debug=False — running debug=True on 0.0.0.0 exposes the Werkzeug
    # interactive debugger to the network (remote code execution via PIN).
    # Set USE_DEBUG=1 in your environment only for local-only development.
    _debug = os.environ.get("USE_DEBUG", "0") == "1"
    app.run(host="0.0.0.0", port=5000, debug=_debug)
