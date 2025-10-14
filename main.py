import torch, argparse
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from models import GCN, GAT
try:
    from torch_geometric.datasets import Planetoid
    import torch_geometric.transforms as T
except Exception as e:
    print("PyTorch Geometric not available in the current kernel. "
          "Run the uv installation commands and then restart the kernel using the 'test/.venv' interpreter.")
    raise


# add near your other imports
import matplotlib.pyplot as plt

def init_history():
    return {"epoch": [], "val_roc": [], "test_roc": []}

def update_history(hist, epoch, val_roc, test_roc):
    hist["epoch"].append(epoch)
    hist["val_roc"].append(val_roc)
    hist["test_roc"].append(test_roc)

def plot_roc_auc(hist, save_path=None, show=True):
    """
    Plot ROC-AUC over epochs for validation and test sets.

    Args:
        hist: dict with keys 'epoch', 'val_roc', 'test_roc'
        save_path: optional file path to save the figure (e.g., 'roc_auc.png')
        show: whether to display the plot (set False in headless runs)
    """
    if not hist["epoch"]:
        print("Nothing to plot: history is empty.")
        return

    plt.figure(figsize=(7, 4.5))
    plt.plot(hist["epoch"], hist["val_roc"], marker="o", label="Val ROC-AUC")
    plt.plot(hist["epoch"], hist["test_roc"], marker="s", label="Test ROC-AUC")
    plt.xlabel("Epoch")
    plt.ylabel("ROC-AUC")
    plt.title("ROC-AUC over Epochs")
    plt.ylim(0.5, 1.0)  # typical baseline for binary tasks
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Saved ROC-AUC plot to: {save_path}")
    if show:
        plt.show()
    plt.close()




def run_split(split_data, model, optimizer, criterion, device, train=False):
    x = split_data.x.to(device)
    edge_index = split_data.edge_index.to(device)
    edge_label_index = split_data.edge_label_index.to(device)
    edge_label = split_data.edge_label.to(device).float()

    if train:
        model.train()
        optimizer.zero_grad()
        z = model(x, edge_index)
        logits = model.link_logits(z, edge_label_index)
        loss = criterion(logits, edge_label)
        loss.backward()
        optimizer.step()
        return float(loss.item())
    else:
        model.eval()
        with torch.no_grad():
            z = model(x, edge_index)
            logits = model.link_logits(z, edge_label_index)
            probs = torch.sigmoid(logits).detach().cpu().numpy()
            preds = (probs >= 0.5).astype("int32")
            labels = edge_label.detach().cpu().numpy()
            acc = accuracy_score(labels, preds)
            f1 = f1_score(labels, preds)
            roc = roc_auc_score(labels, probs)
            return acc, f1, roc



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, choices= ["GCN", "GAT"], default="GCN")
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--hidden_dim', type=int, default=200, help='Number of training epochs')
    parser.add_argument('--out_dim', type=int, default=1, help='Batch size for training')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    dataset = Planetoid(root="data/Planetoid", name="Cora")
    data = dataset[0]


    transform = T.RandomLinkSplit(
        num_val=0.05, num_test=0.10, 
        is_undirected=True, 
        add_negative_train_samples=True
    )

    train_data, val_data, test_data = transform(data)
    print(train_data, val_data, test_data, flush=True)

    ## MODEL parameters
    in_dim = dataset.num_features
    model_ = eval(args.model)
    model = model_(in_dim, args.hidden_dim, args.out_dim).to(device)

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=5e-4)

    # Loss function (Binary Cross Entropy - values between 0 and 1 - suitable for link classification)
    criterion = nn.BCEWithLogitsLoss()

    best_valid = -1.0
    best = {"valid": (0,0,0), "test": (0,0,0)}

    history = init_history()

    for epoch in range(1, args.epochs+1):
        loss = run_split(train_data, model, optimizer, criterion, device, train=True)
        val_acc, val_f1, val_roc = run_split(val_data, model, optimizer, criterion, device, train=False)
        test_acc, test_f1, test_roc = run_split(test_data, model, optimizer, criterion, device,train=False)

        update_history(history, epoch, val_roc, test_roc)
        if val_f1 > best_valid:
            best_valid = val_f1
            best["valid"] = (val_acc, val_f1, val_roc)
            best["test"] = (test_acc, test_f1, test_roc)

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:03d} | loss={loss:.4f} | val: acc={val_acc:.4f} f1={val_f1:.4f} roc={val_roc:.4f}"
                f" | test: acc={test_acc:.4f} f1={test_f1:.4f} roc={test_roc:.4f}")

    plot_roc_auc(history, save_path=args.model + "_roc_auc_over_epochs.png", show=True)

if __name__ == "__main__":
    main()