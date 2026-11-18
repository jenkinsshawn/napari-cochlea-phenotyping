"""Random forest prediction on the GPU.

wrap(forest) returns an object with predict_proba(X) that gives the same values as the
scikit-learn forest, computed with PyTorch on an NVIDIA GPU. The trees are copied into
arrays (feature, threshold, left, right, class-1 share per node) and every sample walks all
trees at once. Without a CUDA GPU, or with BUNDLE_GPU=0, the forest is returned unchanged.
"""
import os
import numpy as np


class TorchForest:
    def __init__(self, forest, device="cuda", chunk=200_000):
        import torch
        self.torch, self.device, self.chunk = torch, device, chunk
        self.classes_ = forest.classes_
        trees = [e.tree_ for e in forest.estimators_]
        n = max(t.node_count for t in trees)
        T = len(trees)
        feat = np.zeros((T, n), np.int64); thr = np.zeros((T, n), np.float64)
        left = np.zeros((T, n), np.int64); right = np.zeros((T, n), np.int64)
        val = np.zeros((T, n), np.float64)
        for i, t in enumerate(trees):
            k = t.node_count; leaf = t.children_left[:k] < 0; idx = np.arange(k)
            feat[i, :k] = np.where(leaf, 0, t.feature[:k])
            thr[i, :k] = t.threshold[:k]
            left[i, :k] = np.where(leaf, idx, t.children_left[:k])       # leaves point to themselves
            right[i, :k] = np.where(leaf, idx, t.children_right[:k])
            v = t.value[:k, 0, :]; val[i, :k] = v[:, 1] / v.sum(1)
        self.depth = max(t.max_depth for t in trees)
        off = (np.arange(T) * n)[None, :]
        to = lambda a: torch.as_tensor(a.reshape(-1), device=device)
        self.feat, self.thr, self.left, self.right, self.val = map(to, (feat, thr, left, right, val))
        self.off = torch.as_tensor(off, device=device)
        self.T = T

    def predict_proba(self, X):
        torch = self.torch
        X = np.asarray(X, np.float32)      # scikit-learn compares float32 features with float64 thresholds
        out = np.empty(len(X), np.float64)
        with torch.no_grad():
            for a in range(0, len(X), self.chunk):
                x = torch.as_tensor(X[a:a + self.chunk], device=self.device).double()
                m = len(x); node = self.off.expand(m, self.T).clone()
                for _ in range(self.depth):
                    f = self.feat[node]
                    go_left = torch.gather(x, 1, f) <= self.thr[node]
                    node = torch.where(go_left, self.left[node], self.right[node]) + self.off
                out[a:a + m] = self.val[node].mean(1).cpu().numpy()
        return np.column_stack([1 - out, out])


def wrap(forest):
    if os.environ.get("BUNDLE_GPU", "1") == "0":
        return forest
    try:
        import torch
        if not torch.cuda.is_available():
            return forest
        return TorchForest(forest)
    except Exception as e:                   # any problem: fall back to the processor
        print(f"  GPU forest not used ({e}); predicting on the processor", flush=True)
        return forest
