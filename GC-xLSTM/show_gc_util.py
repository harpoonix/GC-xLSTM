import matplotlib.pyplot as plt

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

def show_gc_only_estimated(GC, GC_est, save_to: str):
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

