import json
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer  # Added normalization
from sklearn.metrics import silhouette_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from collections import Counter

# reproducibility for scikit-learn estimators
RANDOM_STATE = 42

def load_data(json_path: str) -> list[dict]:
    with open(json_path, 'r') as f:
        data = json.load(f)

    documents = []
    if isinstance(data, dict):
        for doc_id, kw in data.items():
            documents.append({
                "doc_id": doc_id,
                "keywords": kw if isinstance(kw, dict) else {}
            })
    elif isinstance(data, list):
        for e in data:
            doc_id = e.get("file_path") or e.get("doc_id")
            kw     = e.get("keyword_frequency") or e.get("keywords") or {}
            documents.append({
                "doc_id": doc_id,
                "keywords": kw if isinstance(kw, dict) else {}
            })
    else:
        raise ValueError(f"Unexpected JSON format: {type(data)}")
    return documents

def load_tagged_data(json_path: str) -> list[dict]:
    with open(json_path, 'r') as f:
        data = json.load(f)

    documents = []
    for e in data:
        kw = e.get("keyword_frequency") or {}
        documents.append({
            "doc_id": e.get("file_path"),
            "keywords": kw if isinstance(kw, dict) else {},
            "tags": e.get("tags", [])
        })
    return documents

def compute_tag_centroids(
    tagged_docs: list[dict],
    lsa_vectors: np.ndarray,
    num_untagged: int
) -> dict[str, np.ndarray]:
    tag_sums, tag_counts = {}, {}
    for idx, doc in enumerate(tagged_docs):
        vec = lsa_vectors[num_untagged + idx]
        for tag in doc.get("tags", []):
            tag_sums.setdefault(tag, np.zeros_like(vec))
            tag_sums[tag] += vec
            tag_counts[tag] = tag_counts.get(tag, 0) + 1
    return {tag: tag_sums[tag] / tag_counts[tag] for tag in tag_sums}

def assign_tags_to_clusters(
    cluster_centers: np.ndarray,
    tag_centroids: dict[str, np.ndarray]
) -> dict[int, str]:
    tags = list(tag_centroids.keys())
    centroids = np.vstack([tag_centroids[t] for t in tags])
    distances = cdist(cluster_centers, centroids, metric='euclidean')
    closest = np.argmin(distances, axis=1)
    return {i: tags[closest[i]] for i in range(cluster_centers.shape[0])}

def plot_clusters(lsa_vectors: np.ndarray, labels: np.ndarray, k: int) -> None:
    reduced = lsa_vectors[:, :2]
    plt.figure(figsize=(10, 6))
    cmap = plt.get_cmap("tab10")
    for cid in range(k):
        pts = reduced[labels == cid]
        plt.scatter(pts[:, 0], pts[:, 1],
                    color=cmap(cid % 10),
                    label=f"Cluster {cid}",
                    s=50, alpha=0.7)
    plt.title("Clusters (LSA: first two components)")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    #plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def select_optimal_k(
    data: np.ndarray,
    k_min: int,
    k_max: int
) -> int:
    """
    Compute SSE, silhouette, and singleton counts for k = k_min..k_max,
    plot metrics, and return k that maximizes (silhouette, -singleton_count).
    """
    ks = list(range(k_min, min(k_max, data.shape[0] - 1) + 1))
    sses, sil_scores, zero_counts = [], [], []

    for k in ks:
        km = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE)
        labels = km.fit_predict(data)
        sses.append(km.inertia_)
        sil = silhouette_score(data, labels) if k >= 2 else 0
        sil_scores.append(sil)
        sizes = np.bincount(labels, minlength=k)
        zero_counts.append(int(np.sum(sizes == 1)))
        print(f"k={k}: SSE={km.inertia_:.1f}, silhouette={sil:.4f}, singletons={zero_counts[-1]}")

    # plot all three metrics
    plt.figure(figsize=(15, 4))

    plt.subplot(1, 3, 1)
    plt.plot(ks, sses, 'o-', markerfacecolor='none')
    plt.title("Elbow: SSE vs. k")
    plt.xlabel("k"); plt.ylabel("SSE"); plt.grid(True)

    plt.subplot(1, 3, 2)
    plt.plot(ks, sil_scores, 's-', markerfacecolor='none')
    plt.title("Silhouette vs. k")
    plt.xlabel("k"); plt.ylabel("Silhouette"); plt.grid(True)

    plt.subplot(1, 3, 3)
    plt.plot(ks, zero_counts, 'd-', markerfacecolor='none')
    plt.title("Singleton Clusters vs. k")
    plt.xlabel("k"); plt.ylabel("Count of SSE=0"); plt.grid(True)

    plt.tight_layout()
    plt.show()

    # choose best k
    best_idx = max(range(len(ks)), key=lambda i: (sil_scores[i], -zero_counts[i]))
    optimal_k = ks[best_idx]
    print(f"\nSelected k = {optimal_k}  (silhouette={sil_scores[best_idx]:.4f}, singletons={zero_counts[best_idx]})\n")
    return optimal_k

def evaluate_multi_label_threshold(
    tagged_docs: list[dict],
    lsa_vectors: np.ndarray,
    num_untagged: int,
    centroids: dict[str, np.ndarray]
) -> float:
    """
    Split tagged_docs into train/test, compute centroids on train,
    then for test docs compute precision/recall/F1 across thresholds.
    """
    n_tag = len(tagged_docs)
    idxs = np.arange(n_tag)
    train_idx, test_idx = train_test_split(idxs, test_size=0.2, random_state=RANDOM_STATE)

    # build true label matrix
    tag_list = list(centroids.keys())
    y_true = np.zeros((len(test_idx), len(tag_list)), dtype=int)
    for i, doc_i in enumerate(test_idx):
        for t in tagged_docs[doc_i]["tags"]:
            if t in tag_list:
                y_true[i, tag_list.index(t)] = 1

    # compute test vectors and distances to centroids
    test_vecs = lsa_vectors[num_untagged + np.array(test_idx)]
    cent_mat  = np.vstack([centroids[t] for t in tag_list])
    dists     = cdist(test_vecs, cent_mat, metric="euclidean")

    # grid of thresholds
    ths = np.linspace(dists.min(), dists.max(), 30)
    precisions, recalls, f1s = [], [], []

    for thr in ths:
        y_pred = (dists <= thr).astype(int)
        p, r, f, _ = precision_recall_fscore_support(
            y_true, y_pred, average="micro", zero_division=0
        )
        precisions.append(p); recalls.append(r); f1s.append(f)

    # plot P/R/F1 vs threshold
    plt.figure(figsize=(8, 5))
    plt.plot(ths, precisions, label="Precision", marker="o")
    plt.plot(ths, recalls,    label="Recall",    marker="s")
    plt.plot(ths, f1s,        label="F1",        marker="d")
    plt.xlabel("Distance threshold")
    plt.ylabel("Score")
    plt.title("Multi‑label P/R/F1 vs. Threshold")
    plt.legend(); plt.grid(True); plt.tight_layout()
    plt.show()

    best_thr = ths[int(np.argmax(f1s))]
    print(f"Best threshold = {best_thr:.3f} → P={max(precisions):.3f}, R={max(recalls):.3f}, F1={max(f1s):.3f}")
    return best_thr


def main():
    untagged_json = "/home/jno/Curie_Test/Post_Processing/Untagged.json"
    tagged_json   = "/home/jno/Curie_Test/Post_Processing/processed_tagged_pdfs.json"

    untagged_docs = load_data(untagged_json)
    tagged_docs   = load_tagged_data(tagged_json)
    all_docs      = untagged_docs + tagged_docs
    num_untagged  = len(untagged_docs)
    

    
    # Updated pipeline with normalization and dynamic SVD components
    pipeline = Pipeline([
        ("vect", DictVectorizer(sparse=True)),
        ("tfidf", TfidfTransformer(smooth_idf=True)),
        ("svd", TruncatedSVD(n_components=300, random_state=RANDOM_STATE)),  
        ("normalize", Normalizer(norm='l2'))  # L2 normalization
    ])
    keyword_dicts = [d["keywords"] for d in all_docs]
    lsa_vectors   = pipeline.fit_transform(keyword_dicts)


    vect: DictVectorizer = pipeline.named_steps["vect"]
    svd_model: TruncatedSVD = pipeline.named_steps["svd"]
    feature_names = vect.get_feature_names_out()


    # Variance analysis to determine number of components
    expl_var = svd_model.explained_variance_ratio_.cumsum()
    plt.figure()
    plt.plot(range(1, len(expl_var)+1), expl_var, marker='o')
    plt.axhline(0.80, color='grey', linestyle='--', label='80% var')
    plt.xlabel('n_components'); plt.ylabel('Cumulative explained variance')
    plt.title('Choosing LSA n_components')
    plt.legend(); plt.grid(True); plt.show()
    
    # Print explained variance ratios
    print("Explained variance ratios:")
    for i, ratio in enumerate(svd_model.explained_variance_ratio_[:2], start=1):
        print(f"  Component {i}: {ratio:.4f}")
    print()

    # Print top 10 terms for each of the first two components
    for comp_idx in (0, 1):
        comp_num = comp_idx + 1
        comp_vector = svd_model.components_[comp_idx]
        top_indices = np.argsort(np.abs(comp_vector))[::-1][:10]
        print(f"Component {comp_num} top terms (by |weight|):")
        for idx in top_indices:
            term = feature_names[idx]
            weight = comp_vector[idx]
            print(f"  {term:20s} {weight: .4f}")
        print()

    # Print rare clusters
    tag_counts = Counter(tag for doc in tagged_docs for tag in doc["tags"])
    tags, counts = zip(*tag_counts.most_common())
    plt.figure(figsize=(10,4))
    plt.bar(range(len(tags)), counts)
    plt.yscale('log')
    plt.xlabel("Cluster corresponding to tag(s) (sorted by frequency)")
    plt.ylabel("Document count (log scale)")
    plt.title("Cluster Frequency Distribution")
    plt.tight_layout()
    plt.show()
    
    # Highlight small clusters
    rare = [t for t,c in tag_counts.items() if c < 5]
    print(f"Small clustesr (<5 docs): {rare}")



    
    # Compute tag centroids early to initialize K-means
    tag_centroids = compute_tag_centroids(tagged_docs, lsa_vectors, num_untagged)
    distinct_tags = list(tag_centroids.keys())
    k = len(distinct_tags)  # Set k equal to the number of distinct tags

    # print tag index
    print("Tag Index Mapping:")
    for idx, tag in enumerate(distinct_tags):
        print(f"  {idx:3d} → {tag}")
    print() 
    
    # Initialize K-means with tag centroids
    kmeans = KMeans(
        n_clusters=k,
        init=np.vstack([tag_centroids[tag] for tag in distinct_tags]),  # Use tag centroids as initial centers
        n_init=1,  # Custom initialization, no need for multiple runs
        random_state=RANDOM_STATE
    )
    untagged_vecs = lsa_vectors[:num_untagged]
    labels = kmeans.fit_predict(untagged_vecs)

    
    # Plot clusters (optional)
    plot_clusters(untagged_vecs, labels, k)

    # Recompute tag centroids after clustering (optional, if tagged data is included in refinement)
    tag_centroids = compute_tag_centroids(tagged_docs, lsa_vectors, num_untagged)

    # Evaluate and find best threshold
    best_threshold = evaluate_multi_label_threshold(
        tagged_docs, lsa_vectors, num_untagged, tag_centroids
    )

if __name__ == "__main__":
    main()
