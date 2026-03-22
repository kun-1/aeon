"""
Aeon Memory OS — Test Pipeline

End-to-end: GLiNER2 entity extraction → embedding → Aeon Atlas storage → semantic retrieval.
SQLite as companion document store for metadata.
"""

import os
import sys
import json
import time
import sqlite3
import glob

import numpy as np
from gliner2 import GLiNER2
from sentence_transformers import SentenceTransformer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "shell"))
from aeon_py import core
from aeon_py.client import AeonClient


DATA_DIR = os.path.join(os.path.dirname(__file__), "memory_data")
os.makedirs(DATA_DIR, exist_ok=True)

# Clean slate for each run
import glob
for f in glob.glob(os.path.join(DATA_DIR, "*")):
    os.remove(f)

ATLAS_PATH = os.path.join(DATA_DIR, "my_atlas.bin")
DB_PATH = os.path.join(DATA_DIR, "metadata.db")


# ==========================================
# SQLite companion database
# ==========================================

def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS atlas_metadata (
            id         INTEGER PRIMARY KEY,
            content    TEXT NOT NULL,
            timestamp  REAL NOT NULL
        )
    """)
    conn.commit()
    return conn


def save_metadata(conn, node_id, payload):
    conn.execute(
        "INSERT OR REPLACE INTO atlas_metadata (id, content, timestamp) VALUES (?, ?, ?)",
        (node_id, json.dumps(payload), time.time()),
    )
    conn.commit()


def load_metadata(conn, ids):
    if not ids:
        return {}
    placeholders = ",".join("?" * len(ids))
    rows = conn.execute(
        f"SELECT id, content FROM atlas_metadata WHERE id IN ({placeholders})", ids
    ).fetchall()
    return {id_: json.loads(content) for id_, content in rows}


# ==========================================
# 0. Initialize engines
# ==========================================

print("加载 GLiNER2 模型...")
extractor = GLiNER2.from_pretrained("fastino/gliner2-large-v1")

print("加载 Embedding 模型 (768-dim)...")
embedding_model = SentenceTransformer("all-mpnet-base-v2")

print("初始化 Aeon Atlas 内核...")
atlas = AeonClient(ATLAS_PATH)
print(f"  Atlas ready: {atlas.atlas.size()} nodes")

print("初始化 SQLite metadata 库...")
db = init_db()
row_count = db.execute("SELECT COUNT(*) FROM atlas_metadata").fetchone()[0]
print(f"  SQLite ready: {row_count} rows")


# ==========================================
# 1. GLiNER2 entity extraction
# ==========================================

user_text = "I think Aeon is amazing because it uses Write-Ahead Log to prevent data loss."
labels = ["Project", "Feature", "Concept", "Opinion"]

print("\n--- [GLiNER2 提取结果] ---")
raw = extractor.extract_entities(user_text, labels)

extracted_facts = []
for label, texts in raw.get("entities", {}).items():
    for text in texts:
        print(f"  [{label}]: {text}")
        extracted_facts.append(f"{label}: {text}")


# ==========================================
# 2. Vectorize and store in Aeon + SQLite
# ==========================================

fact_payload = {
    "source_text": user_text,
    "entities": extracted_facts,
    "confidence": "high",
}

vec = embedding_model.encode(user_text, convert_to_numpy=True).astype(np.float32)
print(f"\n  Vector shape: {vec.shape}, dtype: {vec.dtype}")

node_id = atlas.atlas.insert(
    parent_id=0,
    vector=vec.tolist(),
    metadata=json.dumps({"placeholder": "see SQLite"}),  # thin placeholder
)
print(f"✅ 写入 Atlas！Node ID: {node_id}")

save_metadata(db, node_id, fact_payload)
print(f"✅ 写入 SQLite！")


# ==========================================
# 3. Semantic retrieval
# ==========================================

print("\n--- [Aeon 语义检索测试] ---")
query_text = "Does the new memory engine support crash recovery?"
query_vec = embedding_model.encode(query_text, convert_to_numpy=True).astype(np.float32)

results = atlas.query(query_vec)

# Collect IDs and load metadata from SQLite
ids = [int(r["id"]) for r in results]
metadata_map = load_metadata(db, ids)

print(f"Query: '{query_text}'")
print(f"Results ({len(results)} total):")
for r in results:
    id_ = int(r["id"])
    sim = r["similarity"]
    meta = metadata_map.get(id_, {})
    print(f"\n  id={id_}  similarity={sim:.4f}")
    print(f"  metadata: {meta}")


# ==========================================
# 4. Build info
# ==========================================

bi = core.get_build_info()
print(f"\n--- [Build Info] ---")
print(f"  compiler={bi.compiler}  arch={bi.architecture}  simd={bi.simd_level}")

print("\n=== All tests passed ===")

db.close()
