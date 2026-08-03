import csv

rows = []
with open("all_checkpoints_validation.csv") as f:
    lines = f.readlines()

header = lines[0].strip().split(",")
for line in lines[1:]:
    parts = line.strip().split(",")
    # last 5 fields are always the numeric ones; everything else is the label
    numeric_parts = parts[-5:]
    label = ",".join(parts[:-5])
    rows.append({
        "label": label,
        "abs_rel": numeric_parts[0],
        "rmse": numeric_parts[1],
        "min_error_m": numeric_parts[2],
        "max_error_m": numeric_parts[3],
        "mean_error_m": numeric_parts[4],
    })

print(f"{'Stage':<32} | {'AbsRel':>8} | {'RMSE (m)':>9} | {'min err (m)':>11} | {'max err (m)':>11} | {'mean err (m)':>12}")
print("-" * 100)
corrected = []
for r in rows:
    abs_rel = float(r["abs_rel"])  # unaffected by scale
    rmse_m = float(r["rmse"]) * 0.256
    min_m = float(r["min_error_m"]) * 0.256
    max_m = float(r["max_error_m"]) * 0.256
    mean_m = float(r["mean_error_m"]) * 0.256
    print(f"{r['label']:<32} | {abs_rel:>8.4f} | {rmse_m:>9.4f} | {min_m:>11.4f} | {max_m:>11.4f} | {mean_m:>12.4f}")
    corrected.append([r["label"], abs_rel, rmse_m, min_m, max_m, mean_m])

with open("all_checkpoints_validation_corrected.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["label", "abs_rel", "rmse_m", "min_error_m", "max_error_m", "mean_error_m"])
    writer.writerows(corrected)
print("\nWrote all_checkpoints_validation_corrected.csv")
