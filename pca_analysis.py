"""
PCA on Gene Expression Data - Breast Cancer (GSE5325)
Reproducing Figure 1a and 1c from the Nature Primer paper.
"""

import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gzip

# ---- load data ----

# load the expression matrix
# rows = patients (105), columns = gene IDs (16174)
df = pd.read_csv('data/filtered.tsv.gz', sep='\t', compression='gzip')
df.columns = df.columns.str.strip()
print(f"Expression matrix shape: {df.shape}")

# load class labels: 1 = ER+, 0 = ER-
import os
class_file = 'data/class.tsv' if os.path.exists('data/class.tsv') else 'data/class.csv'
labels = pd.read_csv(class_file, header=None, names=['class'])
labels = labels['class'].values
print(f"Labels shape: {labels.shape}")
print(f"ER+ count: {np.sum(labels == 1)}, ER- count: {np.sum(labels == 0)}")

# load column mapping to get gene names
cols = pd.read_csv('data/columns.tsv.gz', sep='\t', compression='gzip', comment='#')
print(f"Column mapping shape: {cols.shape}")

# ---- find XBP1 and GATA3 ----

# from the column mapping:
# XBP1 -> ID 4404
# GATA3 -> ID 4359
xbp1_id = '4404'
gata3_id = '4359'

# extract expression values for these two genes
xbp1_expr = df[xbp1_id].values
gata3_expr = df[gata3_id].values

print(f"XBP1 range: [{xbp1_expr.min():.3f}, {xbp1_expr.max():.3f}]")
print(f"GATA3 range: [{gata3_expr.min():.3f}, {gata3_expr.max():.3f}]")

# ---- Figure 1a: scatter plot of XBP1 vs GATA3 ----

fig1, ax1 = plt.subplots(figsize=(6, 5))

# separate by class
er_pos = labels == 1
er_neg = labels == 0

# plot ER- first (black), then ER+ (red) so red shows on top
ax1.scatter(gata3_expr[er_neg], xbp1_expr[er_neg], c='black', s=30, marker='s', label='ER−', zorder=2)
ax1.scatter(gata3_expr[er_pos], xbp1_expr[er_pos], c='red', s=30, marker='s', label='ER+', zorder=3)

ax1.set_xlabel('GATA3', fontsize=13, fontstyle='italic')
ax1.set_ylabel('XBP1', fontsize=13, fontstyle='italic')
ax1.legend(loc='lower right', fontsize=10)
ax1.set_title('(a) XBP1 vs GATA3 Expression', fontsize=13)

plt.tight_layout()
plt.savefig('figure_1a.png', dpi=150, bbox_inches='tight')
print("Saved figure_1a.png")

# ---- PCA on the 2D matrix (GATA3, XBP1) ----

# build the 2D matrix: columns are GATA3 and XBP1
X = np.column_stack([gata3_expr, xbp1_expr])
print(f"\n2D matrix shape: {X.shape}")

# center the data (subtract mean)
X_mean = X.mean(axis=0)
X_centered = X - X_mean
print(f"Mean: GATA3={X_mean[0]:.4f}, XBP1={X_mean[1]:.4f}")

# compute covariance matrix
cov_matrix = np.cov(X_centered, rowvar=False)
print(f"Covariance matrix:\n{cov_matrix}")

# eigendecomposition
eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

# sort by descending eigenvalue
idx = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]

print(f"\nEigenvalues: {eigenvalues}")
print(f"PC1 direction: {eigenvectors[:, 0]}")
print(f"PC2 direction: {eigenvectors[:, 1]}")

# variance explained
var_explained = eigenvalues / eigenvalues.sum() * 100
print(f"Variance explained: PC1={var_explained[0]:.1f}%, PC2={var_explained[1]:.1f}%")

# project onto PC1
pc1_projections = X_centered @ eigenvectors[:, 0]

# flip sign so ER+ is on the right (positive) side, matching reference figure
# eigenvector sign is arbitrary, so this is just a convention choice
if np.mean(pc1_projections[er_pos]) < np.mean(pc1_projections[er_neg]):
    pc1_projections = -pc1_projections

# ---- Figure 1b: scatter plot with PC axes overlaid ----

fig1b, ax1b = plt.subplots(figsize=(6, 5))

# same scatter as 1a
ax1b.scatter(gata3_expr[er_neg], xbp1_expr[er_neg], c='black', s=30, marker='s', label='ER−', zorder=2)
ax1b.scatter(gata3_expr[er_pos], xbp1_expr[er_pos], c='red', s=30, marker='s', label='ER+', zorder=3)

# draw PC1 and PC2 lines through the mean
# each eigenvector gives the direction, we draw a line through (mean_GATA3, mean_XBP1)
line_len = 5  # how far the line extends from center
for i, (pc_label, style) in enumerate(zip(['PC1', 'PC2'], ['-', '-'])):
    direction = eigenvectors[:, i]
    # flip PC1 direction if we flipped projections earlier
    if i == 0 and np.mean(pc1_projections[er_pos]) > np.mean(pc1_projections[er_neg]):
        direction = -direction if eigenvectors[:, 0][0] < 0 else direction
    start = X_mean - line_len * direction
    end = X_mean + line_len * direction
    ax1b.plot([start[0], end[0]], [start[1], end[1]], 'k-', linewidth=1.2, zorder=4)
    # add arrow at the positive end
    arrow_pos = X_mean + (line_len * 0.85) * direction
    arrow_dir = direction * 0.4
    ax1b.annotate(pc_label, xy=(arrow_pos[0], arrow_pos[1]),
                  xytext=(arrow_pos[0] - arrow_dir[0]*1.5, arrow_pos[1] - arrow_dir[1]*1.5),
                  fontsize=12, fontstyle='italic', fontweight='bold',
                  arrowprops=dict(arrowstyle='->', lw=1.5, color='black'),
                  zorder=5)

ax1b.set_xlabel('GATA3', fontsize=13, fontstyle='italic')
ax1b.set_ylabel('XBP1', fontsize=13, fontstyle='italic')
ax1b.legend(loc='lower right', fontsize=10)
ax1b.set_title('(b) PCA Axes on XBP1 vs GATA3', fontsize=13)

plt.tight_layout()
plt.savefig('figure_1b.png', dpi=150, bbox_inches='tight')
print("Saved figure_1b.png")

# ---- Figure 1c: projection onto PC1 ----

fig2, ax2 = plt.subplots(figsize=(8, 4))

y_all = 3
y_er_neg = 2
y_er_pos = 1

# all samples
ax2.scatter(pc1_projections[er_neg], np.full(np.sum(er_neg), y_all), c='black', s=20, marker='s', zorder=2)
ax2.scatter(pc1_projections[er_pos], np.full(np.sum(er_pos), y_all), c='red', s=20, marker='s', zorder=3)

# ER- only
ax2.scatter(pc1_projections[er_neg], np.full(np.sum(er_neg), y_er_neg), c='black', s=20, marker='s', zorder=2)

# ER+ only
ax2.scatter(pc1_projections[er_pos], np.full(np.sum(er_pos), y_er_pos), c='red', s=20, marker='s', zorder=3)

# horizontal lines
for y in [y_all, y_er_neg, y_er_pos]:
    ax2.axhline(y=y, color='gray', linewidth=0.5, zorder=1)

ax2.set_yticks([y_pos for y_pos in [y_all, y_er_neg, y_er_pos]])
ax2.set_yticklabels(['All', 'ER−', 'ER+'], fontsize=12)
ax2.set_xlabel('Projection onto PC1', fontsize=13)
ax2.set_ylim(0.3, 3.7)
ax2.set_title('(c) PC1 Projections', fontsize=13)

plt.tight_layout()
plt.savefig('figure_1c.png', dpi=150, bbox_inches='tight')
print("Saved figure_1c.png")

# ---- combined figure (both plots side by side) ----

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# panel a
ax = axes[0]
ax.scatter(gata3_expr[er_neg], xbp1_expr[er_neg], c='black', s=30, marker='s', label='ER−', zorder=2)
ax.scatter(gata3_expr[er_pos], xbp1_expr[er_pos], c='red', s=30, marker='s', label='ER+', zorder=3)
ax.set_xlabel('GATA3', fontsize=13, fontstyle='italic')
ax.set_ylabel('XBP1', fontsize=13, fontstyle='italic')
ax.legend(loc='lower right', fontsize=10)
ax.set_title('a', fontsize=16, fontweight='bold', loc='left')

# panel c
ax = axes[1]
ax.scatter(pc1_projections[er_neg], np.full(np.sum(er_neg), y_all), c='black', s=20, marker='s', zorder=2)
ax.scatter(pc1_projections[er_pos], np.full(np.sum(er_pos), y_all), c='red', s=20, marker='s', zorder=3)
ax.scatter(pc1_projections[er_neg], np.full(np.sum(er_neg), y_er_neg), c='black', s=20, marker='s', zorder=2)
ax.scatter(pc1_projections[er_pos], np.full(np.sum(er_pos), y_er_pos), c='red', s=20, marker='s', zorder=3)
for y in [y_all, y_er_neg, y_er_pos]:
    ax.axhline(y=y, color='gray', linewidth=0.5, zorder=1)
ax.set_yticks([y_all, y_er_neg, y_er_pos])
ax.set_yticklabels(['All', 'ER−', 'ER+'], fontsize=12)
ax.set_xlabel('Projection onto PC1', fontsize=13)
ax.set_ylim(0.3, 3.7)
ax.set_title('c', fontsize=16, fontweight='bold', loc='left')

plt.tight_layout()
plt.savefig('figure_1a_1c_combined.png', dpi=150, bbox_inches='tight')
print("Saved figure_1a_1c_combined.png")
print("\nDone!")
