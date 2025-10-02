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

client = QdrantClient(path="db/qdrant_db")
CN = "movies"

# 1) ids
with open("data/id_map.json", "r", encoding="utf-8") as f:
    ids = [int(x) for x in json.load(f)["ids"]]

# 2) vectors
emb = np.load("data/embeddings.npy").astype("float32")
assert emb.shape[0] == len(ids) and emb.shape[1] == 768

# 3) payload (JSONL -> dict по tmdb_id)
payload_by_id = {}
with open("data/payload.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue
        rec = json.loads(line)
        tmdb_id = (
            rec.get("tmdb_id")
            or rec.get("meta.tmdb_id")
            or (rec.get("meta") or {}).get("tmdb_id")
        )
        if tmdb_id is not None:
            payload_by_id[int(tmdb_id)] = rec

# 4) выравниваем payload по порядку ids
aligned_payload = [payload_by_id.get(i, {"meta": {"tmdb_id": i}}) for i in ids]

# 5) заливаем коллекцию
client.upload_collection(
    collection_name=CN,
    vectors=emb,
    payload=aligned_payload,
    ids=ids,
    batch_size=1000,
    parallel=2,
    wait=True,
)

print("OK, points:", client.count(CN, exact=True).count)
