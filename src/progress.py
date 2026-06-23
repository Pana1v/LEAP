"""Live training progress bar. Run: python progress.py"""
import csv, time, os
from pathlib import Path

LOG = Path("logs/gnn_dataset_200_objects_20260405_110305/training_log.csv")
TOTAL_EPOCHS = 12  # 3 (stage5) + 3 (stage10) + 6 (stage20)
BAR_WIDTH = 40

def read_last_row(path):
    rows = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows[-1] if rows else None

def bar(done, total, width=BAR_WIDTH):
    filled = int(width * done / total)
    return f"[{'█' * filled}{'░' * (width - filled)}] {done}/{total}"

print(f"Watching: {LOG}")
while True:
    if LOG.exists():
        row = read_last_row(LOG)
        if row:
            epoch = int(row["epoch"])
            stage = row["stage"]
            gap   = float(row["val_mean_gap_vs_greedy"])
            cost  = float(row["val_mean_cost"])
            wr    = float(row["val_win_rate"])
            loss  = float(row["train_loss"])
            done  = epoch >= TOTAL_EPOCHS

            os.system("clear")
            print("─" * (BAR_WIDTH + 20))
            print(f"  GNN Training — 200 objects")
            print("─" * (BAR_WIDTH + 20))
            print(f"  {bar(epoch, TOTAL_EPOCHS)}")
            print(f"  Stage       : {stage}")
            print(f"  Train loss  : {loss:.4f}")
            print(f"  Val cost    : {cost:.1f}")
            print(f"  Gap/greedy  : {gap:+.2f}%")
            print(f"  Win rate    : {wr:.1f}%")
            print("─" * (BAR_WIDTH + 20))
            if done:
                print("  ✓ Training complete!")
                break
            print(f"  (refreshes every 30s — Ctrl+C to exit)")
    time.sleep(30)
