import matplotlib.pyplot as plt
import numpy as np
from typing import List

def show_gc(GC, GC_est, best_accuracy_gc, save_to: str):
    # Make figures for best loss and also best accuracy
    fig, axarr = plt.subplots(1, 3, figsize=(15, 5))
    if GC is not None:
        axarr[0].imshow(GC, cmap="Blues")
        axarr[0].set_title("Ground Truth GC")
        axarr[0].set_ylabel("Affected series")
        axarr[0].set_xlabel("Causal series")
        axarr[0].set_xticks([])
        axarr[0].set_yticks([])

    axarr[1].imshow(
        GC_est, cmap="Blues", vmin=0, vmax=1, extent=(0, len(GC_est), len(GC_est), 0)
    )
    axarr[1].set_title("Estimated GC")
    axarr[1].set_ylabel("Affected series")
    axarr[1].set_xlabel("Causal series")
    axarr[1].set_xticks([])
    axarr[1].set_yticks([])

    axarr[2].imshow(
        best_accuracy_gc,
        cmap="Blues",
        vmin=0,
        vmax=1,
        extent=(0, len(best_accuracy_gc), len(best_accuracy_gc), 0),
    )
    axarr[2].set_title("GC estimated (best accuracy)")
    axarr[2].set_ylabel("Affected series")
    axarr[2].set_xlabel("Causal series")
    axarr[2].set_xticks([])
    axarr[2].set_yticks([])

    # Mark disagreements in best model
    if GC is not None:
        for i in range(len(GC_est)):
            for j in range(len(GC_est)):
                if GC[i, j] != GC_est[i, j]:
                    rect = plt.Rectangle(
                        (j, i - 0.05), 1, 1, facecolor="none", edgecolor="red", linewidth=1
                    )
                    axarr[1].add_patch(rect)

        # Mark disagreements in best accuracy
        for i in range(len(best_accuracy_gc)):
            for j in range(len(best_accuracy_gc)):
                if GC[i, j] != best_accuracy_gc[i, j]:
                    rect = plt.Rectangle(
                        (j, i - 0.05), 1, 1, facecolor="none", edgecolor="red", linewidth=1
                    )
                    axarr[2].add_patch(rect)

    plt.savefig(save_to, dpi=200)

def show_gc_actual_and_estimated(GC, GC_est, save_to: str):
    # Make figures for GC and estimated GC
    fig, axarr = plt.subplots(1, 2, figsize=(10, 5))
    if GC is not None:
        axarr[0].imshow(GC, cmap="Blues")
        axarr[0].set_title("Ground Truth GC", fontsize=25)
        axarr[0].set_ylabel("Affected series", fontsize=25)
        axarr[0].set_xlabel("Causal series", fontsize=25)
        axarr[0].set_xticks([])
        axarr[0].set_yticks([])

    axarr[1].imshow(
        GC_est, cmap="Blues", vmin=0, vmax=1, extent=(0, len(GC_est), len(GC_est), 0)
    )
    axarr[1].set_title("Estimated GC", fontsize=25)
    # axarr[1].set_ylabel("Affected series", fontsize=25)
    axarr[1].set_xlabel("Causal series", fontsize=25)
    axarr[1].set_xticks([])
    axarr[1].set_yticks([])

    # Mark disagreements in estimated GC
    if GC is not None:
        for i in range(len(GC_est)):
            for j in range(len(GC_est)):
                if GC[i, j] != GC_est[i, j]:
                    rect = plt.Rectangle(
                        (j, i - 0.05), 1, 1, facecolor="none", edgecolor="red", linewidth=1
                    )
                    axarr[1].add_patch(rect)

    plt.savefig(save_to, dpi=200, bbox_inches="tight")

def show_lag_selection_gc(GC, GC_est, GC_est_lag_all, save_to: str):
    # Verify lag selection
    for i in range(len(GC_est)):
        # Get true GC
        GC_lag = np.zeros((5, len(GC_est)))
        GC_lag[:3, GC[i].astype(bool)] = 1.0

        # Get estimated GC
        GC_est_lag = GC_est_lag_all[i].T[::-1]

        # Make figures
        fig, axarr = plt.subplots(1, 2, figsize=(16, 5))
        axarr[0].imshow(GC_lag, cmap='Blues', extent=(0, len(GC_est), 5, 0))
        axarr[0].set_title('Series %d true GC' % (i + 1), fontsize=25)
        axarr[0].set_ylabel('Lag', fontsize=25)
        axarr[0].set_xlabel('Series', fontsize=25)
        axarr[0].set_xticks(np.arange(len(GC_est)) + 0.5)
        axarr[0].set_xticklabels(range(len(GC_est)), fontsize=25)
        axarr[0].set_yticks(np.arange(5) + 0.5)
        axarr[0].set_yticklabels(range(1, 5 + 1), fontsize=25)
        axarr[0].tick_params(axis='both', length=0)

        axarr[1].imshow(GC_est_lag, cmap='Blues', extent=(0, len(GC_est), 5, 0))
        axarr[1].set_title('Series %d estimated GC' % (i + 1), fontsize=25)
        axarr[1].set_ylabel('Lag', fontsize=25)
        axarr[1].set_xlabel('Series', fontsize=25)
        axarr[1].set_xticks(np.arange(len(GC_est)) + 0.5)
        axarr[1].set_xticklabels(range(len(GC_est)), fontsize=25)
        axarr[1].set_yticks(np.arange(5) + 0.5)
        axarr[1].set_yticklabels(range(1, 5 + 1), fontsize=25)
        axarr[1].tick_params(axis='both', length=0)

        # Mark nonzeros
        for k in range(len(GC_est)):
            for j in range(5):
                if GC_est_lag[j, k] > 0.0:
                    rect = plt.Rectangle((k, j), 1, 1, facecolor='none', edgecolor='green', linewidth=1.0)
                    axarr[1].add_patch(rect)

        plt.savefig(f"{save_to}_{i}.pdf")
        
def show_estimated_gc_with_feature_names(GC_est, feature_names: List[str], save_to: str):
    fig, ax = plt.subplots(figsize=(12, 12))
    cax = ax.imshow(GC_est, cmap="Blues", vmin=0, vmax=1)

    # Set axis labels
    ax.set_xticks(np.arange(len(feature_names)))
    ax.set_yticks(np.arange(len(feature_names)))
    ax.set_xticklabels(feature_names, fontsize=20, rotation=45, ha="right")
    ax.set_yticklabels(feature_names, fontsize=20)

    # Add gridlines
    ax.grid(False)

    # Save the figure
    plt.savefig(save_to, dpi=200, bbox_inches="tight")
    plt.close(fig)