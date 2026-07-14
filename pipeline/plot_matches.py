import csv
import os

import matplotlib
matplotlib.use('Agg')  # non-interactive backend since on WSL
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch 
import matplotlib.image as mpimg
import numpy as np

def read_matches(csv_path):
    matches = []
    with open(csv_path, newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if not row or row[0].startswith('#'):
                continue
            if len(row) < 4:
                raise ValueError(f'Invalid match row: {row}')
            x1, y1, x2, y2 = map(float, row[:4])
            matches.append((x1, y1, x2, y2))
    return matches


def load_image(path):
    img = mpimg.imread(path)
    img = np.rot90(img, k=-1)
    return img


def plot_matches(image1_path, image2_path, matches, corrected_matches=None, marker_size=8, line_alpha=0.35, output_path=None):
    img1 = load_image(image1_path)
    img2 = load_image(image2_path)

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    axes[0].imshow(img1)
    axes[1].imshow(img2)

    axes[0].set_title(os.path.basename(image1_path))
    axes[1].set_title(os.path.basename(image2_path))

    for ax in axes:
        ax.axis('off')

    for i, (x1, y1, x2, y2) in enumerate(matches, start=1):
        axes[0].scatter([x1], [y1], s=marker_size**2, c='orange', edgecolors='black', linewidths=0.8, zorder=3)
        axes[1].scatter([x2], [y2], s=marker_size**2, c='orange', edgecolors='black', linewidths=0.8, zorder=3)
        axes[0].text(x1 + 4, y1 - 6, str(i), color='white', fontsize=8, weight='bold', zorder=4)
        axes[1].text(x2 + 4, y2 - 6, str(i), color='white', fontsize=8, weight='bold', zorder=4)

        con = ConnectionPatch(
            xyA=(x2, y2), coordsA=axes[1].transData,
            xyB=(x1, y1), coordsB=axes[0].transData,
            axesA=axes[1], axesB=axes[0], color='yellow', alpha=line_alpha, linewidth=0.9, zorder=1
        )
        fig.add_artist(con)
    
    if corrected_matches is not None:
        for i, (x1, y1, x2, y2) in enumerate(corrected_matches, start=1):
            axes[0].scatter([x1], [y1], s=marker_size**2, c='lime', edgecolors='black', linewidths=0.8, zorder=3)
            axes[1].scatter([x2], [y2], s=marker_size**2, c='lime', edgecolors='black', linewidths=0.8, zorder=3)
            axes[0].text(x1 + 4, y1 - 6, str(i), color='white', fontsize=8, weight='bold', zorder=4)
            axes[1].text(x2 + 4, y2 - 6, str(i), color='white', fontsize=8, weight='bold', zorder=4)

            con = ConnectionPatch(
                xyA=(x2, y2), coordsA=axes[1].transData,
                xyB=(x1, y1), coordsB=axes[0].transData,
                axesA=axes[1], axesB=axes[0], color='lime', alpha=line_alpha*0.7, linewidth=1.5, zorder=2
            )
            fig.add_artist(con)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f'Saved match plot to {output_path}')

plot_matches(
    image1_path='./images/table1.jpeg',
    image2_path='./images/table2.jpeg',
    matches=read_matches('./data/table_matches.csv'),
    corrected_matches=read_matches('./data/table_corrected_correspondences.csv'),
    marker_size=8,
    line_alpha=0.35,
    output_path="./matches.pdf"
)

