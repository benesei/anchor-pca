"""Plot the covariance perturbation path and its optimal-method intervals."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle, ConnectionPatch
import numpy as np
import pandas as pd
from anchorpca.reproducibility import publication_style

source_method_ids = ['poolPCA', 'maxRCS', 'maxRegret', 'norm-maxRegret', 'AnchorPCA_lambda=25', 'AnchorPCA_infty']
colors = ['tab:blue', '#2ca02c', '0.5', '#E69F00', '#6A3D9A', '#D62728']
labels = ['poolPCA', 'maxRCS', 'maxRegret', 'norm-maxRegret', r'AnchorPCA$_{25}$', r'AnchorPCA$_\infty$']
styles = ['-', '--', ':', '-.', '-', '-']
rx = lambda x: x
yr = lambda x: x

def plot_from_results(results_dir: Path, figures_dir: Path):
    curves_table = pd.read_csv(results_dir / 'perturbation_path_curves.csv')
    intervals = pd.read_csv(results_dir / 'perturbation_path_best_intervals.csv')
    cross = pd.read_csv(results_dir / 'perturbation_path_crossovers.csv')['rho_star'].tolist()
    bounds = [0, *cross, 500]
    winners = [source_method_ids.index(m) for m in intervals['best_method_id']]
    publication_style()
    plt.rcParams.update({'font.size': 11.04, 'axes.labelsize': 12.419999999999998, 'axes.titlesize': 12.419999999999998, 'xtick.labelsize': 11.04, 'ytick.labelsize': 11.04, 'legend.fontsize': 10.35})
    fig = plt.figure(figsize=(8, 3.05))
    az = fig.add_axes([0.065, 0.3180327868852459, 0.255, 0.5895081967213115])
    af = fig.add_axes([0.415, 0.3180327868852459, 0.43, 0.5895081967213115])
    al = fig.add_axes([0.87, 0.39704918032786873, 0.16, 0.41606557377049186])
    al.axis('off')
    for ax, limit, title in [(az, 20, 'Origin region: $0\\leq\\rho\\leq20$'), (af, 500, 'Full perturbation path')]:
        for i, method in enumerate(source_method_ids):
            v = curves_table[curves_table.method_id == method]
            ax.plot(v['rho'], v['avg_reconstruction_error'], color=colors[i], ls=styles[i], lw=1.55 if i in [0, 4, 5] else 1.1)
        ax.set_xlim(rx(0), rx(limit))
        ax.set_ylim((yr(91), yr(126)) if limit == 20 else (yr(68), yr(620)))
        xt = [0, 5, 10, 15, 20] if limit == 20 else [0, 100, 200, 300, 400, 500]
        yt = [100, 110, 120] if limit == 20 else [100, 200, 300, 400, 500, 600]
        ax.set_xticks([rx(x) for x in xt], list(map(str, xt)))
        ax.set_yticks([yr(y) for y in yt], list(map(str, yt)))
        ax.set_title(title, pad=5)
        ax.grid(False)
        ax.text(1.055, -0.11109566184649612, '$\\rho$', transform=ax.transAxes)
    az.set_ylabel('Avg. reconstruction error')
    af.axvline(rx(150), color='0.5', ls=':', lw=1)
    af.text(rx(157), yr(592), '$\\rho_0=2E\\lambda=150$', fontsize=10.35, va='top', color='0.35')
    zoom_rect = Rectangle((rx(0), yr(91)), rx(20) - rx(0), yr(126) - yr(91), fill=False, ec='0.55', lw=0.65, ls='--')
    af.add_patch(zoom_rect)
    for yy in [91, 126]:
        fig.add_artist(ConnectionPatch((rx(20), yr(yy)), (rx(0), yr(yy)), coordsA=az.transData, coordsB=af.transData, color='0.65', lw=0.6, ls='--', zorder=0))
    order = [5, 4, 0, 1, 2, 3]
    al.legend([Line2D([], [], color=colors[i], ls=styles[i], lw=1.6) for i in order], [labels[i] for i in order], loc='center left', frameon=False, handlelength=2.2, borderaxespad=0, fontsize=11.04, labelspacing=0.8)
    bar_labels = {0: 'poolPCA', 4: 'AnchorPCA$_{\\mathbf{25}}$', 5: 'AnchorPCA$_{\\mathbf{\\infty}}$'}

    def centered_25(bar, x):
        from matplotlib.textpath import TextPath
        from matplotlib.font_manager import FontProperties
        from matplotlib.patches import PathPatch
        from matplotlib.transforms import Affine2D, ScaledTranslation
        prop = FontProperties(family='DejaVu Serif', weight='bold', math_fontfamily='dejavuserif')
        label = TextPath((0, 0), bar_labels[4], size=8.4, prop=prop)
        box = label.get_extents()
        transform = Affine2D().translate(-(box.x0 + box.x1) / 2, -(box.y0 + box.y1) / 2).scale(1 / 72) + fig.dpi_scale_trans + ScaledTranslation(x, 0.5, bar.transData)
        bar.add_patch(PathPatch(label, transform=transform, facecolor='white', edgecolor='none', clip_on=False))
        placed = label.get_extents(transform)
        assert np.allclose([(placed.x0 + placed.x1) / 2, (placed.y0 + placed.y1) / 2], bar.transData.transform((x, 0.5)), rtol=0, atol=1e-08)
    for ax, limit in [(az, 20), (af, 500)]:
        pos = ax.get_position()
        bar = fig.add_axes([pos.x0, 0.1359016393442623, pos.width, 0.07213114754098361])
        bar.set_xlim(0, limit)
        bar.set_ylim(0, 1)
        bar.axis('off')
        for j, i in enumerate(winners):
            lo, hi = (max(bounds[j], 0), min(bounds[j + 1], limit))
            if hi > lo:
                bar.add_patch(Rectangle((lo, 0), hi - lo, 1, color=colors[i], lw=0))
        if limit == 20:
            for j, i in enumerate(winners):
                lo, hi = (bounds[j], min(bounds[j + 1], limit))
                if i == 4 and hi > lo:
                    centered_25(bar, (lo + hi) / 2)
                elif i == 0 and hi > lo:
                    bar.text((lo + hi) / 2, 0.5, bar_labels[i], ha='center', va='center', color='white', fontsize=8.4, fontweight='bold', fontfamily='DejaVu Serif', math_fontfamily='dejavuserif')
            bar.text(-0.025, 0.5, 'Best:', transform=bar.transAxes, ha='right', va='center', fontsize=10.35)
        else:
            for j, i in enumerate(winners):
                if i == 4:
                    centered_25(bar, (bounds[j] + bounds[j + 1]) / 2)
                elif i == 5:
                    from matplotlib.textpath import TextPath
                    from matplotlib.font_manager import FontProperties
                    from matplotlib.path import Path as MplPath
                    from matplotlib.patches import PathPatch
                    from matplotlib.transforms import Affine2D, ScaledTranslation
                    prop = FontProperties(family='DejaVu Serif', weight='bold')
                    word = TextPath((0, 0), 'AnchorPCA', size=8.4, prop=prop)
                    digits = TextPath((0, 0), '25', size=5.88, prop=prop)
                    symbol = TextPath((0, 0), '∞', size=5.88, prop=prop)
                    factor = digits.get_extents().height / symbol.get_extents().height
                    symbol = Affine2D().scale(factor).transform_path(symbol)
                    symbol = Affine2D().translate(word.get_extents().xmax + 0.42 - symbol.get_extents().xmin, -1.38 - symbol.get_extents().ymin).transform_path(symbol)
                    label = MplPath.make_compound_path(word, symbol)
                    box = label.get_extents()
                    transform = Affine2D().translate(-(box.x0 + box.x1) / 2, -(box.y0 + box.y1) / 2).scale(1 / 72) + fig.dpi_scale_trans + ScaledTranslation((bounds[j] + bounds[j + 1]) / 2, 0.5, bar.transData)
                    bar.add_patch(PathPatch(label, transform=transform, facecolor='white', edgecolor='none', clip_on=False))
            for val, ha in [(cross[-2], 'left'), (cross[-1], 'left')]:
                bar.plot([val, val], [0, -0.4090909090909092], color='0.5', lw=0.6, clip_on=False)
                bar.text(val, -0.5795454545454546, f'$\\rho\\approx{val:.2f}$', ha=ha, va='top', fontsize=10.2)
    for label in af.get_yticklabels():
        label.set_bbox({'facecolor': 'white', 'edgecolor': 'none', 'pad': 0.1})
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figures_dir / 'perturbation_path.pdf', bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--figures-dir", type=Path, required=True)
    args = parser.parse_args()
    plot_from_results(args.results_dir, args.figures_dir)
