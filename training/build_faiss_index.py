import os, json, pickle
import numpy as np
from PIL import Image
import torch
import faiss
from huggingface_hub import login
from transformers import AutoModel, AutoProcessor

# ── STEP 1: Login FIRST — before any from_pretrained call ──
HF_TOKEN = "REDACTED_HF_TOKEN"
login(token=HF_TOKEN, add_to_git_credential=False)
print("HuggingFace login successful.")

# ── Config ──
CASE_DIR   = "data/faiss_case_library"
META_FILE  = f"{CASE_DIR}/case_metadata.json"
INDEX_FILE = f"{CASE_DIR}/case_embeddings.faiss"
EMBED_FILE = f"{CASE_DIR}/case_embeddings.npy"

os.makedirs(CASE_DIR, exist_ok=True)

# ── STEP 2: Load MedSigLIP with token passed explicitly ──
print("Loading MedSigLIP...")   
model_id = "google/medsiglip-448"   # correct model ID (not so400m — that doesn't exist)

processor = AutoProcessor.from_pretrained(
    model_id,
    token=HF_TOKEN          # pass token directly as extra safety
)

model = AutoModel.from_pretrained(
    model_id,
    token=HF_TOKEN,
    dtype=torch.float32   # CPU-safe
)
model.eval()
print("MedSigLIP loaded on CPU.")

# ── STEP 3: Load case metadata ──
if not os.path.exists(META_FILE):
    print(f"ERROR: {META_FILE} not found. Create it first.")
    exit(1)

with open(META_FILE) as f:
    cases = json.load(f)

print(f"Building FAISS index for {len(cases)} reference cases...")

embeddings = []

for case in cases:
    img_path = os.path.join(CASE_DIR, case["image_file"])

    if not os.path.exists(img_path):
        print(f"  WARNING: {img_path} not found — using random placeholder embedding")
        # Use a deterministic placeholder (same seed = reproducible)
        rng = np.random.RandomState(hash(case["case_id"]) % (2**31))
        placeholder = rng.randn(768).astype(np.float32)
        placeholder = placeholder / (np.linalg.norm(placeholder) + 1e-8)
        embeddings.append(placeholder)
        continue

    image = Image.open(img_path).convert("RGB")
    text  = f"Medical image: {case['modality']}. Diagnosis: {case['diagnosis']}"

    inputs = processor(
        text=[text],
        images=[image],
        return_tensors="pt",
        padding=True
    )

    with torch.no_grad():
        outputs = model(**inputs)
        # SigLIP outputs image_embeds directly
        img_emb = outputs.image_embeds.squeeze(0).numpy()

    # L2-normalize for cosine similarity via inner product
    img_emb = img_emb / (np.linalg.norm(img_emb) + 1e-8)
    img_emb = img_emb.astype(np.float32)
    embeddings.append(img_emb)
    print(f"  Embedded: {case['case_id']} ({case['modality']}) — dim: {img_emb.shape[0]}")

# ── STEP 4: Build FAISS index ──
embeddings_arr = np.stack(embeddings)
dim = embeddings_arr.shape[1]
print(f"\nEmbedding matrix shape: {embeddings_arr.shape}")

index = faiss.IndexFlatIP(dim)   # inner product = cosine similarity (normalized vectors)
index.add(embeddings_arr)

faiss.write_index(index, INDEX_FILE)
np.save(EMBED_FILE, embeddings_arr)
print(f"FAISS index saved: {INDEX_FILE}")
print(f"Index size: {index.ntotal} vectors, dim: {dim}")

# ── STEP 5: Test query ──
print("\nTesting query with case 0 as query...")
query = embeddings_arr[0:1]
D, I = index.search(query, k=3)
print("Top-3 similar cases:")
for rank, (dist, idx) in enumerate(zip(D[0], I[0])):
    print(f"  Rank {rank+1}: {cases[idx]['case_id']} — {cases[idx]['diagnosis']} (score: {dist:.4f})")
