import csv

rows = []
with open("all_checkpoints_validation.csv") as f:
    reader = csv.DictReader(f)
    for row in reader:
        rows.append(row)

print(f"{'Stage':<32} | {'AbsRel':>8} | {'RMSE (m)':>9} | {'min err (m)':>11} | {'max err (m)':>11} | {'mean err (m)':>12}")
print("-" * 100)
for r in rows:
    abs_rel = float(r["abs_rel"])  # unaffected by scale
    rmse_m = float(r["rmse"]) * 0.256
    min_m = float(r["min_error_m"]) * 0.256
    max_m = float(r["max_error_m"]) * 0.256
    mean_m = float(r["mean_error_m"]) * 0.256
    print(f"{r['label']:<32} | {abs_rel:>8.4f} | {rmse_m:>9.4f} | {min_m:>11.4f} | {max_m:>11.4f} | {mean_m:>12.4f}")

with open("all_checkpoints_validation_corrected.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["label", "abs_rel", "rmse_m", "min_error_m", "max_error_m", "mean_error_m"])
    for r in rows:
        writer.writerow([
            r["label"], r["abs_rel"],
            float(r["rmse"]) * 0.256,
            float(r["min_error_m"]) * 0.256,
            float(r["max_error_m"]) * 0.256,
            float(r["mean_error_m"]) * 0.256,
        ])
print("\nWrote all_checkpoints_validation_corrected.csv")
