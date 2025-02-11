import time
import os
import logging
import numpy as np
import torch
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from omegaconf import OmegaConf, DictConfig
from dacite import from_dict
from models.clstm import train_model_ista, componentXLSTM
from xlstm import (
    xLSTMBlockStackConfig,
)
from prepare_data import return_data
from datasets.mocap.all_asfamc.AMCParser.plot_motion_gc import plot_graph_from_GC
from show_gc_util import show_gc, show_gc_only_estimated

# torch.manual_seed(500)

parser = ArgumentParser()
parser.add_argument("--config", required=True, type=str)
parser.add_argument("--no-log", action="store_true", help="Do not log to file")
parser.add_argument("--lam", type=float, help="Lambda value for training")

args = parser.parse_args()

with open(args.config, "r", encoding="utf8") as fp:
    config_yaml = fp.read()
cfg: DictConfig = OmegaConf.create(config_yaml)
OmegaConf.resolve(cfg)

if args.lam is not None:
    cfg.training.lam = args.lam
    cfg.training.lam_alpha = args.lam


EXPERIMENT_IDENTIFIER = f"{time.strftime('%y%m%d-%H%M')}-{cfg.results.gc}"
exp_directory = f"exp/{cfg.dataset.name}/{EXPERIMENT_IDENTIFIER}"
# Create a new directory inside results with the name EXPERIMENT_IDENTIFIER
os.makedirs(f"exp/{cfg.dataset.name}/{EXPERIMENT_IDENTIFIER}/images", exist_ok=True)
# Configure logging
if not args.no_log:
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        filename=f"{exp_directory}/train.log",
        filemode="w",
    )
else:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)
logger.info("Configuration loaded:\n%s", OmegaConf.to_yaml(cfg))

device = torch.device("cuda")

# Get data
data = return_data(cfg.dataset)
X_np = data[0]
GC = data[1]
if "molene" in cfg.dataset.name:
    stations = data[2]
elif "mocap" in cfg.dataset.name:
    joint_info = data[2]

X = torch.tensor(X_np[np.newaxis], dtype=torch.float32, device=device)

config = from_dict(
    data_class=xLSTMBlockStackConfig, data=OmegaConf.to_container(cfg.model)
)

xlstm: componentXLSTM = componentXLSTM(
    X.shape[-1], hidden=cfg.model.embedding_dim, config=config
).cuda(device=device)

logger.info("Model initialized:\n%s", xlstm)

# Train with ISTA
(
    train_loss_list,
    pred_loss_list,
    alpha_loss_list,
    var_usage_list,
    accuracy_list,
    best_accuracy_gc,
    best_accuracy_model,
    lam_list,
) = train_model_ista(
    xlstm,
    X,
    **cfg.training,
    true_GC=GC,
)

# Save the model
model_path = f"{exp_directory}/minima.pt"
torch.save(xlstm.state_dict(), model_path)
logger.info("Best Loss Model saved to %s", model_path)

# Save the model with best accuracy
model_path = f"{exp_directory}/accuracy.pt"
torch.save(best_accuracy_model.state_dict(), model_path)
logger.info("Best Accuracy Model saved to %s", model_path)

# best epoch
best_epoch = np.argmin(train_loss_list)
# best accuracy
best_accuracy = np.argmax(accuracy_list)

# Loss function plot
fig, ax1 = plt.subplots(figsize=(8, 8))

color = "tab:blue"
ax1.set_xlabel("Training steps")
ax1.set_ylabel("Loss", color=color)
ax1.plot(
    50 * np.arange(len(train_loss_list)),
    train_loss_list,
    color=color,
    label="Train Loss",
)
ax1.plot(
    50 * np.arange(len(pred_loss_list)),
    pred_loss_list,
    color="tab:orange",
    label="Pred Loss",
)
ax1.plot(
    50 * np.arange(len(alpha_loss_list)),
    alpha_loss_list,
    color="tab:green",
    label="Reduction Loss",
)
ax1.tick_params(axis="y", labelcolor=color)
# a red dot marks the best epoch
ax1.plot(50 * best_epoch, train_loss_list[best_epoch], "ro", label="Best Model")
# a red star marks the best accuracy
# ax1.plot(
#     50 * best_accuracy, train_loss_list[best_accuracy], "r*", label="Best Accuracy"
# )

fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.title("")
ax1.legend()
plt.savefig(f"{exp_directory}/images/training.pdf", dpi=200)

# Variable Usage Plot
plt.figure(figsize=(8, 5))
plt.plot(50 * np.arange(len(var_usage_list)), var_usage_list)
plt.plot(50 * best_epoch, var_usage_list[best_epoch], "ro")
# plt.plot(50 * best_accuracy, var_usage_list[best_accuracy], "r*")
plt.title("variable usage")
plt.ylabel("Usage")
plt.xlabel("Training steps")
plt.tight_layout()
plt.savefig(f"{exp_directory}/images/usage.pdf", dpi=200)

# Accuracy Plot
plt.figure(figsize=(8, 5))
plt.plot(50 * np.arange(len(accuracy_list)), accuracy_list)
plt.plot(50 * best_epoch, accuracy_list[best_epoch], "ro")
plt.plot(50 * best_accuracy, accuracy_list[best_accuracy], "r*")
plt.title("accuracy")
plt.ylabel("Accuracy")
plt.xlabel("Training steps")
plt.tight_layout()
plt.savefig(f"{exp_directory}/images/accuracy.pdf", dpi=200)

# Check learned Granger causality
GC_est = xlstm.GC().cpu().data.numpy()

logger.info("True variable usage = %.2f%%", 100 * np.mean(GC) if GC is not None else 0)
logger.info("Estimated variable usage = %.2f%%", 100 * np.mean(GC_est))
logger.info("Accuracy = %.2f%%", 100 * np.mean(GC == GC_est) if GC is not None else 0)

# pretty print learned Gc, which is a matrix of size (p, p)
logger.info("Actual GC:\n%s", GC)
logger.info(
    "Learned GC (best loss):\n%s",
    np.array2string(
        xlstm.GC(threshold=False).softmax(-1).cpu().data.numpy(),
        formatter={"float_kind": lambda x: "%.3f" % x},
    ),
)
logger.info(
    "Learned GC (best accuracy):\n%s",
    np.array2string(
        best_accuracy_gc.softmax(-1).cpu().data.numpy(),
        formatter={"float_kind": lambda x: "%.3f" % x},
    ),
)

logger.info(
    "For debugging, actual values of these. Learned GC (best loss) : \n%s",
    xlstm.GC(threshold=False).cpu().data.numpy(),
)
logger.info(
    "For debugging, actual values of these. Learned GC (best accuracy) : \n%s",
    best_accuracy_gc.cpu().data.numpy(),
)
best_accuracy_gc = (best_accuracy_gc > 0).int().cpu().data.numpy()

# show_gc(GC, GC_est, best_accuracy_gc, f"{exp_directory}/images/GC.pdf")
show_gc_only_estimated(GC, GC_est, f"{exp_directory}/images/GC_estimated.pdf")


GC_results = {"pred" : GC_est}
if GC is not None:
    GC_results.update({"true" : GC})
    # calculate TPR and FPR for AUC-ROC

    true_positives = np.sum(GC * GC_est)
    false_positives = np.sum((1 - GC) * GC_est)
    true_negatives = np.sum((1 - GC) * (1 - GC_est))
    false_negatives = np.sum(GC * (1 - GC_est))

    tpr = true_positives / (true_positives + false_negatives)
    fpr = false_positives / (false_positives + true_negatives)

    logger.info(f"True positives: {true_positives}, False negatives: {false_negatives}")
    logger.info(f"False positives: {false_positives}, True negatives: {true_negatives}")
    logger.info("TPR = %.2f, FPR = %.2f", tpr, fpr)

    # save tpr and fpr to a file GC-xLSTMexp/lorenz-aucroc/lorenz-tpr-fpr.csv
    with open(f"GC-xLSTMexp/{cfg.dataset.name}/tpr-fpr.csv", "a") as f:
        f.write(f"{cfg.training.lam},{tpr},{fpr}\n")

np.savez_compressed(f"{exp_directory}/GC_results.npz", **GC_results)
logger.info("Saved GC results to %s", f"{exp_directory}/GC_results.npz")

# for molene dataset plot
if "molene" in cfg.dataset.name:
    from datasets.molene.plot_results import create_weather_map
    
    print(f"stations: {stations}")
    print(f"number of stations: {len(stations)}")
    print(f"GC_est: {GC_est.shape}")

    create_weather_map(
        station_ids=stations,
        adj_matrix=GC_est,
        output_file=f"{exp_directory}/images/weather_map.html",
    )
    
    create_weather_map(
        station_ids=stations,
        adj_matrix=GC_est,
        output_file=f"{exp_directory}/images/weather_map_no_empty.html",
        display_empty_stations=False
    )

# for mocap dataset plot
if "mocap" in cfg.dataset.name:
    np.savez_compressed(f"{exp_directory}/mocap_data.npz", GC_est=GC_est, joint_info=joint_info)
    logger.info("Saved mocap data to %s", f"{exp_directory}/mocap_data.npz")
    
    plot_graph_from_GC(gc=GC_est, joint_info=joint_info, filename=f"{exp_directory}/images/mocap_plot.pdf")
    logger.info("Saved mocap plot to %s", f"{exp_directory}/images/mocap_plot.pdf")                