import os, json, numpy as np
import streamlit as st
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import torch
from pathlib import Path
import re, unicodedata, difflib

from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    SearchParams, Filter, FieldCondition, MatchAny, MatchValue, Range, PointStruct, VectorParams, Distance
)


# ---------------------
# INIT
# ---------------------
load_dotenv()
if "GROQ_API_KEY" not in os.environ:
    os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
COLL_NAME = os.getenv("QDRANT_COLLECTION", "movies")

# Пути к данным для импорта при пустой коллекции
IDS_PATH  = Path("data/id_map.json")
EMB_PATH  = Path("data/embeddings.npy")
PAYL_PATH = Path("data/payload.jsonl")

# Корень локального стора Qdrant (внутри должен лежать каталог `collection/`)
DB_PATH = os.getenv("QDRANT_PATH", "db/qdrant_db")

# Настройка обрезки для fallback-описаний (0 = не обрезать)
FALLBACK_OVERVIEW_CHARS = 0  # 0 = не обрезать fallback-описания вовсе

st.set_page_config(page_title="MoviesRAG", layout="wide")

# Lazy imports for LLM only if key present
_llm = None
if GROQ_API_KEY:
    from langchain_groq import ChatGroq
    from langchain.prompts import ChatPromptTemplate

    _PITCH_PROMPT = ChatPromptTemplate.from_messages([
        ("system", """
Отвечай ТОЛЬКО валидным JSON строго такого вида:
{"items": {"<id>": {"pitch": "2–3 коротких, естественных предложения на живом русском без канцелярита и штампов; без спойлеров; допустима одна уместная метафора", "why": ["кому может зайти (1 короткая причина)", "настроение/темп (1 короткая причина)", "ещё одна конкретика: жанр/сеттинг/актёр"]}}}
Требования к стилю:
- Разговорно и по‑человечески, вместо общих слов — конкретика.
- Не повторяй название фильма в pitch.
- Если данных мало — честно напиши: «мало данных, ориентируйся по жанрам».
- Язык: русский. ВНЕ JSON — ничего.
"""),
        ("human", "Данные фильмов (JSON):\n{items_json}\nСформируй pitch и why по каждому фильму.")
    ])
    _llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0.6, max_tokens=1600)

@st.cache_resource(show_spinner=False)
def get_client() -> QdrantClient:
    path = DB_PATH
    # если указали .../collection — подняться на 1 уровень к корню
    if os.path.basename(path.rstrip("/")) == "collection":
        path = os.path.dirname(path.rstrip("/"))
    # если случайно передали путь к storage.sqlite — подняться на 2 уровня
    if os.path.isfile(os.path.join(path, "storage.sqlite")):
        path = os.path.dirname(os.path.dirname(path))
    try:
        return QdrantClient(path=path)
    except RuntimeError as e:
        # лок ожидает другой процесс
        st.error("Qdrant локальный стор уже открыт другим процессом. Закрой другой запуск или перезапусти Python. Детали: " + str(e))
        st.stop()
    except Exception as e:
        emsg = repr(e)
        # несовместимый формат хранилища → создаём новый путь и работаем с ним
        if ("ValidationError" in emsg) or ("CreateCollection" in emsg):
            base = os.path.abspath(path)
            fresh = base + "_fresh"
            os.makedirs(fresh, exist_ok=True)
            st.warning(f"Локальная БД Qdrant несовместима с текущим клиентом. Использую новый стор: {fresh}. Ниже можно переимпортировать коллекцию из data/*.")
            return QdrantClient(path=fresh)
        st.error("Не удалось открыть локальный Qdrant: " + str(e))
        st.stop()

@st.cache_resource(show_spinner=False)
def get_embedder() -> SentenceTransformer:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return SentenceTransformer("intfloat/multilingual-e5-base", device=device)

@st.cache_resource(show_spinner=False)
def get_vector_size() -> int:
    return get_embedder().get_sentence_embedding_dimension()

# --- Fuzzy title index ---

def _norm_text(s: str) -> str:
    if not s:
        return ""
    s = str(s).lower().replace("ё", "е")
    s = unicodedata.normalize("NFKD", s)
    # оставить буквы/цифры (включая кириллицу), заменить прочее пробелом
    s = re.sub(r"[^\w]+", " ", s, flags=re.UNICODE)
    return " ".join(s.split())

@st.cache_resource(show_spinner=False)
def build_title_index(name: str):
    """Загружаем (кешируем) индекс заголовков для лексического поиска.
    Возвращает список кортежей (id, norm_title, orig_title).
    """
    _client = get_client()
    out = []
    next_offset = None
    # ограничим до 100k на всякий случай
    fetched = 0
    LIMIT = 4096
    HARD_MAX = 100_000
    while True:
        pts, next_offset = _client.scroll(
            collection_name=name,
            with_payload=True,
            limit=LIMIT,
            offset=next_offset,
        )
        for p in pts:
            pl = p.payload or {}
            t = pl.get("title_ru") or pl.get("original_title") or pl.get("title")
            if t:
                out.append((p.id, _norm_text(t), t))
        fetched += len(pts)
        if not next_offset or fetched >= HARD_MAX or len(pts) == 0:
            break
    return out

def fuzzy_candidates(query: str, index: list, top_k: int = 5):
    """Возвращает top_k кандидатов по нечеткому совпадению.
    Использует максимальный из трёх скорингов: полное сходство, partial ratio по окнам и token-set overlap.
    """
    nq = _norm_text(query)
    if not nq:
        return []

    def _partial_ratio(nq: str, nt: str) -> float:
        la, lb = len(nq), len(nt)
        if la == 0 or lb == 0:
            return 0.0
        if la >= lb:
            return difflib.SequenceMatcher(None, nq, nt).ratio()
        best = 0.0
        # скользящее окно по символам
        for i in range(0, lb - la + 1):
            sub = nt[i : i + la]
            r = difflib.SequenceMatcher(None, nq, sub).ratio()
            if r > best:
                best = r
        return best

    def _token_set_ratio(nq: str, nt: str) -> float:
        a, b = set(nq.split()), set(nt.split())
        if not a or not b:
            return 0.0
        inter = a & b
        if not inter:
            return 0.0
        # Дайс на токенах
        return 2 * len(inter) / (len(a) + len(b))

    scored = []
    for pid, nt, ot in index:
        if not nt:
            continue
        full = difflib.SequenceMatcher(None, nq, nt).ratio()
        part = _partial_ratio(nq, nt)
        tset = _token_set_ratio(nq, nt)
        score = max(full, part, tset)
        scored.append((score, pid, ot))

    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[:top_k]

# --- Data loaders + importer ---

def _fail(msg: str):
    st.error(msg)
    st.stop()

@st.cache_resource(show_spinner=False)
def _load_ids(path: Path) -> list[int]:
    if not path.exists():
        _fail(f"Не найден файл id_map: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    ids: list = []
    if isinstance(data, list):
        ids = data
    elif isinstance(data, dict):
        # частый вариант: {"ids": [...]} -> берём список
        if "ids" in data and isinstance(data["ids"], list):
            ids = data["ids"]
        else:
            # попытка: словарь с числовыми ключами
            try:
                items = sorted(((int(k), v) for k, v in data.items()), key=lambda x: x[0])
                ids = [v for _, v in items]
            except Exception:
                # если единственное поле содержит список — используем его
                if len(data) == 1:
                    only_val = next(iter(data.values()))
                    if isinstance(only_val, list):
                        ids = only_val
                if not ids:
                    _fail("Неподдержимый формат id_map.json — ожидается list, dict с ключом 'ids' или dict с числовыми ключами")
    else:
        _fail("Неподдержимый формат id_map.json — ожидается list или dict")

    # приведение к int
    try:
        return [int(x) for x in ids]
    except Exception:
        _fail("id_map.json: не удалось привести значения ids к int")

@st.cache_resource(show_spinner=False)
def _load_embeddings(path: Path) -> np.ndarray:
    if not path.exists():
        _fail(f"Не найден файл эмбеддингов: {path}")
    arr = np.load(path)
    if arr.dtype != np.float32:
        arr = arr.astype(np.float32)
    return arr

@st.cache_resource(show_spinner=False)
def _load_payloads(path: Path) -> list[dict]:
    if not path.exists():
        _fail(f"Не найден payload JSONL: {path}")
    out = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out

def _maybe_l2_normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    mask = norms.squeeze(-1) > 1e-12
    out = vectors.copy()
    out[mask] = out[mask] / norms[mask]
    return out

def import_points(client: QdrantClient, name: str,
                  ids_path: Path = IDS_PATH,
                  emb_path: Path = EMB_PATH,
                  payl_path: Path = PAYL_PATH):
    ids = _load_ids(ids_path)
    vecs = _load_embeddings(emb_path)
    payloads = _load_payloads(payl_path)

    n_ids, n_vecs = len(ids), vecs.shape[0]
    if n_ids != n_vecs:
        _fail(f"Несовпадение размеров: ids={n_ids}, emb={n_vecs}")

    # Приводим к float32 и L2-нормализуем для COSINE
    if vecs.dtype != np.float32:
        vecs = vecs.astype(np.float32)
    vecs = _maybe_l2_normalize(vecs)

    # Строим маппинг payload по tmdb_id / meta.tmdb_id
    payload_by_id: dict[int, dict] = {}
    for rec in payloads:
        tmdb_id = rec.get("tmdb_id")
        if tmdb_id is None:
            tmdb_id = _get(rec, "meta.tmdb_id")
        if tmdb_id is None:
            continue
        try:
            payload_by_id[int(tmdb_id)] = rec
        except Exception:
            continue

    # Выравниваем payload по порядку ids
    aligned_payload = [payload_by_id.get(i, {"meta": {"tmdb_id": i}}) for i in ids]

    st.info(f"Импорт {n_vecs} точек в коллекцию '{name}'…")
    client.upload_collection(
        collection_name=name,
        vectors=vecs,
        payload=aligned_payload,
        ids=ids,
        batch_size=1000,
        parallel=2,
        wait=True,
    )
    st.success(f"Импорт завершён: {n_vecs} точек.")

# --- Helpers ---

def ensure_collection(client: QdrantClient, name: str):
    """Гарантирует существование коллекции. Если нет — создаёт с нужной размерностью."""
    try:
        cols = client.get_collections().collections
        names = [getattr(c, "name", None) for c in cols]
    except Exception:
        names = []
    if name in names:
        return
    dim = get_vector_size()
    client.create_collection(
        collection_name=name,
        vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
    )

def embed_query(text: str) -> np.ndarray:
    vec = get_embedder().encode([f"query: {text}"], normalize_embeddings=True)
    return vec[0].astype("float32")

# Multiple query variants for better recall (e5 family)
def make_query_vectors(text: str) -> list[np.ndarray]:
    e = get_embedder()
    variants = [f"query: {text}", text, f"passage: {text}"]
    vecs = e.encode(variants, normalize_embeddings=True)
    return [v.astype("float32") for v in vecs]

# Extract years from user query
def extract_years_from_query(q: str) -> list[int]:
    if not q:
        return []
    yrs = re.findall(r"(?<!\d)(19\d{2}|20\d{2})(?!\d)", q)
    years = []
    for y in yrs:
        try:
            yi = int(y)
            if 1900 <= yi <= 2100:
                years.append(yi)
        except Exception:
            pass
    return years

# Build a Qdrant filter for year range (supports float payloads and two possible keys)
def make_year_filter(y_min: float, y_max: float) -> Filter:
    return Filter(
        should=[
            FieldCondition(key="meta.year", range=Range(gte=float(y_min), lte=float(y_max))),
            FieldCondition(key="year",      range=Range(gte=float(y_min), lte=float(y_max))),
        ]
    )

# Merge results from multiple vectors; keep max score per id
def knn_search_merge(client: QdrantClient,
                     collection: str,
                     vectors: list[np.ndarray],
                     yr_filter: Filter | None,
                     limit_each: int = 80) -> list:
    pool: dict = {}
    for v in vectors:
        kwargs = dict(
            collection_name=collection,
            query=v,
            limit=limit_each,
            with_payload=True,
            search_params=SearchParams(exact=True),
        )
        if yr_filter is not None:
            kwargs["query_filter"] = yr_filter
        res = _qp(client, **kwargs)
        for sp in res.points:
            pid = sp.id
            score = getattr(sp, "score", 0.0) or 0.0
            if (pid not in pool) or (score > pool[pid][0]):
                pool[pid] = (score, sp)
    ranked = sorted(pool.values(), key=lambda x: x[0], reverse=True)
    return [sp for _, sp in ranked]

# --- Qdrant compatibility wrapper for query_points keyword ---

def _qp(client: QdrantClient, **kwargs):
    """Call query_points with correct filter keyword across client versions.
    Prefers `query_filter`. Falls back to `filter` if needed, and vice versa.
    """
    try:
        # Try with whatever kwargs are passed in
        return client.query_points(**kwargs)
    except AssertionError as e:
        msg = str(e)
        # Unknown arguments handling
        if "['query_filter']" in msg and 'query_filter' in kwargs:
            # Try replacing with legacy 'filter'
            k2 = dict(kwargs)
            k2['filter'] = k2.pop('query_filter')
            return client.query_points(**k2)
        if "['filter']" in msg and 'filter' in kwargs:
            # Try replacing with modern 'query_filter'
            k2 = dict(kwargs)
            k2['query_filter'] = k2.pop('filter')
            return client.query_points(**k2)
        raise
    except TypeError as e:
        smsg = str(e)
        if 'query_filter' in smsg and 'query_filter' in kwargs:
            k2 = dict(kwargs)
            k2['filter'] = k2.pop('query_filter')
            return client.query_points(**k2)
        if 'filter' in smsg and 'filter' in kwargs:
            k2 = dict(kwargs)
            k2['query_filter'] = k2.pop('filter')
            return client.query_points(**k2)
        raise

# Optional typed filter builder (на будущее)
def build_filter(genres: list[str] | None = None,
                 year_range: tuple[int, int] | None = None,
                 language: str | None = None) -> Filter | None:
    must = []
    if genres:
        must.append(FieldCondition(key="genres", match=MatchAny(any=genres)))
    if year_range:
        gte, lte = year_range
        must.append(FieldCondition(key="meta.year", range=Range(gte=gte, lte=lte)))
    if language:
        must.append(FieldCondition(key="original_language", match=MatchValue(value=language)))
    return Filter(must=must) if must else None

# Helper to get nested field

def _get(payload: dict, path: str, default=None):
    if path in payload:
        return payload.get(path, default)
    cur = payload
    for part in path.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return default
    return cur

# ---------------
# LLM summaries
# ---------------

def llm_pitches(items: list[dict]) -> dict[str, dict]:
    def _fallback_why(it: dict) -> str:
        parts = []
        g = (it.get("genres") or [])[:3]
        if g: parts.append("жанры: " + ", ".join(g))
        y = it.get("year")
        if y: parts.append(f"год: {y}")
        return "; ".join(parts) or "похожее по эмбеддингам"

    def _fallback_pitch(it: dict) -> str:
        ov = it.get("overview_ru") or it.get("overview") or ""
        s = (ov or "").strip()
        if not s:
            return "Фильм без описания."
        if FALLBACK_OVERVIEW_CHARS and len(s) > FALLBACK_OVERVIEW_CHARS:
            return s[:FALLBACK_OVERVIEW_CHARS] + "…"
        return s

    def _fallback(it: dict) -> dict:
        return {"pitch": _fallback_pitch(it), "why": _fallback_why(it)}

    if not GROQ_API_KEY or _llm is None:
        return {str(it.get("tmdb_id") or it.get("id") or i): _fallback(it) for i, it in enumerate(items)}

    safe_items = [{
        "id": it.get("id") or it.get("tmdb_id"),
        "title": it.get("title_ru") or it.get("original_title") or it.get("title") or "",
        "year": _get(it, "meta.year") or it.get("year"),
        "genres": it.get("genres") or [],
        "overview": it.get("overview_ru") or it.get("overview") or "",
    } for it in items]

    try:
        msgs = _PITCH_PROMPT.format_messages(items_json=json.dumps(safe_items, ensure_ascii=False))
        raw = _llm.invoke(msgs).content
        data = json.loads(raw)
        items_map = data.get("items", {})
        result = {}
        for k, v in items_map.items():
            pitch_text = str(v.get("pitch", ""))
            why_val = v.get("why", "")
            if isinstance(why_val, list):
                why_text = " • ".join(str(x) for x in why_val if x)
            else:
                why_text = str(why_val) if why_val is not None else ""
            result[str(k)] = {"pitch": pitch_text, "why": why_text}
        return result
    except Exception:
        return {str(it.get("id") or i): _fallback(it) for i, it in enumerate(items)}

# ---------------------
# UI
# ---------------------
st.markdown("### Введите запрос: ")
query = st.text_input("Запрос", placeholder="например: корейские триллеры 2025 без супергероев", label_visibility="collapsed")


hits = []




if query:
    client = get_client()
    # Проверяем наличие коллекции, не создаём
    ensure_collection(client, COLL_NAME)

    # Готовим индекс заголовков для лексического фолбэка (кешируется)
    title_index = build_title_index(COLL_NAME)

    # Year filter from query (e.g., "2025" -> only year 2025)
    years = extract_years_from_query(query)
    yr_filter = None
    if years:
        y_min, y_max = float(min(years)), float(max(years))
        yr_filter = make_year_filter(y_min, y_max)
        st.caption(f"Фильтр по году: {int(y_min)}-{int(y_max)}")

    # Если коллекция есть, но пустая — предложить импорт
    try:
        total_points = client.count(COLL_NAME, exact=True).count
    except Exception:
        total_points = None
    if total_points == 0:
        st.warning("Коллекция существует, но пуста. Можно импортировать данные из data/*.")
        col_a, col_b = st.columns([1, 2])
        with col_a:
            do_import = st.button("Импортировать сейчас", type="primary")
        with col_b:
            st.caption(f"IDS: {IDS_PATH} · EMB: {EMB_PATH} · PAYLOAD: {PAYL_PATH}")
        if do_import:
            import_points(client, COLL_NAME)
            try:
                total_points = client.count(COLL_NAME, exact=True).count
            except Exception:
                total_points = None
        if not total_points:
            st.stop()

    # Поиск (мультивекторный для большего recall)
    vectors = make_query_vectors(query)
    candidates = knn_search_merge(client, COLL_NAME, vectors, yr_filter, limit_each=80)
    hits = candidates[:5]

    # Client-side year enforcement in case payload year is float/str
    if years and hits:
        y_min, y_max = float(min(years)), float(max(years))
        _filtered = []
        for p in hits:
            pay = p.payload or {}
            yv = _get(pay, "meta.year")
            if yv is None:
                yv = pay.get("year")
            try:
                yr = float(yv)
            except Exception:
                continue
            if (y_min - 1e-6) <= yr <= (y_max + 1e-6):
                _filtered.append(p)
        hits = _filtered

    # Если строго по году ничего не нашлось, мягко расширим диапазон на ±1 год
    if years and not hits:
        y_min, y_max = float(min(years)), float(max(years))
        yr_filter_relaxed = make_year_filter(y_min - 1.0, y_max + 1.0)
        candidates = knn_search_merge(client, COLL_NAME, vectors, yr_filter_relaxed, limit_each=80)
        hits = candidates[:5]
        if hits:
            st.caption(f"Фильтр по году временно расширен до {int(y_min)-1}–{int(y_max)+1} для добора кандидатов")

    # Fuzzy-фолбэк по названию (на случай опечаток)
    fuzzy = fuzzy_candidates(query, title_index, top_k=5)
    if fuzzy and (not hits or fuzzy[0][0] >= 0.60):  # порог снижен для длинных названий
        f_ids = [pid for _, pid, _ in fuzzy]
        f_points = client.retrieve(collection_name=COLL_NAME, ids=f_ids, with_payload=True)
        # Учитываем фильтр по годам, если он задан
        if years:
            y_min, y_max = float(min(years)), float(max(years))
            filtered = []
            for p in f_points:
                pay = p.payload or {}
                yv = _get(pay, "meta.year")
                if yv is None:
                    yv = pay.get("year")
                try:
                    yr = float(yv)
                except Exception:
                    continue
                if (y_min - 1e-6) <= yr <= (y_max + 1e-6):
                    filtered.append(p)
            f_points = filtered
        if f_points:
            st.info("Найдено по лексическому совпадению в названиях (исправление опечаток).")
            hits = f_points

    # Сбор карточек
    items = []
    for r in hits:
        p = r.payload or {}
        items.append({
            "id": r.id,
            "title_ru": p.get("title_ru"),
            "original_title": p.get("original_title"),
            "year": _get(p, "meta.year") or p.get("year"),
            "poster_url": _get(p, "meta.poster_url") or p.get("poster_url"),
            "overview_ru": p.get("overview_ru"),
            "overview": p.get("overview"),
            "genres": p.get("genres") or [],
        })

    # Питчи + краткое объяснение
    data_map = llm_pitches(items)

    cols = st.columns(5, gap="small")
    for col, it in zip(cols, items):
        with col:
            title = it.get("title_ru") or it.get("original_title") or "Без названия"
            raw_year = it.get("year")
            year = None
            if raw_year is not None:
                try:
                    yf = float(raw_year)
                    yi = int(yf)
                    year = yi if abs(yf - yi) < 1e-6 else yi
                except Exception:
                    year = raw_year
            poster = it.get("poster_url")
            caption = f"{title} ({year})" if year else title
            if poster:
                st.image(poster, caption=caption, width='stretch')
            else:
                st.markdown(f"**{caption}**")
            d = data_map.get(str(it.get("id")), {})
            pitch = d.get("pitch") or ""
            why = d.get("why") or ""
            if pitch:
                st.caption(pitch)
            if why:
                parts = [p.strip() for p in str(why).replace("•", "|").split("|") if p.strip()]
                if parts:
                    st.markdown("<div style='font-size:0.85em;opacity:0.85'>Почему:</div>", unsafe_allow_html=True)
                    st.markdown("\n".join(f"- {p}" for p in parts))

if query and not hits:
    st.info("Ничего не найдено по запросу. Попробуй переформулировать или ослабить условия.")