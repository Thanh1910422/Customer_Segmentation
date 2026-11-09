"""Huấn luyện lại mô hình bằng pipeline của dự án và ghi tham số vào index.html.
Chạy trong thư mục dự án (cùng chỗ với main.py, CC GENERAL.csv, results/):
    python build_web.py
"""
import json, re, io, contextlib
import numpy as np, pandas as pd
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from scipy.optimize import linear_sum_assignment
import data_preprocessing as dp

CSV, HTML = "CC GENERAL.csv", "index.html"
K, COV = 5, "spherical"          # Khớp cấu hình GMM có Silhouette tốt nhất tại K=5.

with contextlib.redirect_stdout(io.StringIO()):
    df, Xs, Xp, feats = dp.load_and_preprocess(CSV)
raw = pd.read_csv(CSV)
imp = raw[feats].fillna(raw[feats].median())

LOG = ["BALANCE","PURCHASES","CASH_ADVANCE","CREDIT_LIMIT","PAYMENTS","MINIMUM_PAYMENTS","PURCHASES_TRX"]
params, Mx = [], imp.copy()
for c in feats:
    lo, hi = Mx[c].quantile([.01, .99]); Mx[c] = Mx[c].clip(lo, hi)
    lg = c in LOG and Mx[c].skew() > 1
    if lg: Mx[c] = np.log1p(Mx[c])
    params.append(dict(name=c, lo=float(lo), hi=float(hi), log=bool(lg)))
mu, sd = Mx.mean().values, Mx.std(ddof=0).values
Z = (Mx.values - mu) / sd
assert np.abs(Z - Xs.values).max() < 1e-9, "Tiền xử lý không khớp data_preprocessing.py"
pca = PCA(n_components=0.95, svd_solver="full").fit(Z); P = pca.transform(Z)[:, :2]

km = KMeans(n_clusters=K, init="k-means++", n_init=50, max_iter=500, random_state=42).fit(P)
gm = GaussianMixture(n_components=K, covariance_type=COV, reg_covar=1e-4, n_init=10,
                     max_iter=500, init_params="kmeans", tol=1e-3, random_state=42).fit(P)
kl, gl = km.labels_, gm.predict(P)

ct = pd.crosstab(gl, kl).values
r, c = linear_sum_assignment(-ct); gmap = [int(c[list(r).index(j)]) for j in range(K)]

presets = []
for k in range(K):
    idx = np.where(kl == k)[0]
    j = idx[np.argmin(((P[idx] - km.cluster_centers_[k])**2).sum(1))]
    presets.append({f: round(float(imp.loc[j, f]), 2) for f in feats})
sidx = np.random.RandomState(7).choice(len(P), 240, replace=False)
pts = [[round(float(P[i,0]),2), round(float(P[i,1]),2), int(kl[i]), int(gl[i])] for i in sidx]

cmp_ = pd.read_csv("results/model_comparison.csv", encoding="utf-8-sig")
def row(i): r_ = cmp_.loc[i]; return dict(sil=float(r_.Silhouette), db=float(r_.Davies_Bouldin),
    ch=float(r_.Calinski_Harabasz), ari=float(r_.Stability_ARI_Mean))
M = dict(feats=params, mu=mu.round(6).tolist(), sd=sd.round(6).tolist(),
    pca=dict(components=pca.components_[:2].round(6).tolist(), mean=pca.mean_.round(8).tolist(),
             ev=pca.explained_variance_ratio_[:2].round(4).tolist()),
    km=dict(centers=km.cluster_centers_.round(5).tolist(), sizes=np.bincount(kl).tolist()),
    gmm=dict(weights=gm.weights_.round(6).tolist(), means=gm.means_.round(5).tolist(),
             var=gm.covariances_.round(6).tolist(), sizes=np.bincount(gl).tolist()),
    gmap=gmap, presets=presets, pts=pts,
    mx=[round(float(raw[f].quantile(.99))) if i < 5 else 1 for i, f in enumerate(feats)],
    medians=[[round(v,2) for v in raw.loc[kl==k, feats].median()] for k in range(K)],
    metrics=dict(kmeans=row(2), gmm=row(3), agree_ari=float(cmp_.loc[2,"Agreement_ARI"]), n=int(len(raw))))

js = json.dumps(M, separators=(",",":"), ensure_ascii=False)
html = open(HTML, encoding="utf-8").read()
html, n = re.subn(r"const M=\{.*?\};\n", lambda _: "const M=" + js + ";\n", html, count=1, flags=re.S)
assert n == 1, "Không tìm thấy dòng 'const M=...;' trong index.html"
open(HTML, "w", encoding="utf-8").write(html)
print("Đã cập nhật", HTML, "| GMM->K-Means:", gmap)
