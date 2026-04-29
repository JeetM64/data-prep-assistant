"""Quick validation script for /execute endpoint."""
import pandas as pd
import numpy as np
import io, json, time, sys

# Create a synthetic test dataset (generic, not Titanic)
np.random.seed(42)
n = 500
df = pd.DataFrame({
    "id": range(1, n+1),
    "age": np.random.randint(18, 80, n).astype(float),
    "income": np.random.lognormal(10, 1, n),
    "gender": np.random.choice(["Male", "Female"], n),
    "city": np.random.choice(["NYC", "LA", "SF", "Chicago", "Houston", "Miami", "Dallas", "Seattle"], n),
    "score": np.random.normal(50, 15, n),
    "notes": ["This is a long text note " + str(i) for i in range(n)],
    "constant_col": 1,
    "target": np.random.choice([0, 1], n)
})
# Inject some NaNs
df.loc[df.sample(50, random_state=1).index, "age"] = np.nan
df.loc[df.sample(350, random_state=2).index, "notes"] = np.nan  # >70% missing

# === Test prepare.py directly ===
sys.path.insert(0, r"c:\Users\jeetm\OneDrive\Desktop\data-prep-assistant\backend")

from app.prepare import autofix_dataset

print("=" * 60)
print("VALIDATION: autofix_dataset")
print("=" * 60)

start = time.time()
cleaned, log_messages = autofix_dataset(df, "target")
elapsed = time.time() - start

print(f"\n--- Shape: {cleaned.shape}")
print(f"--- Columns: {list(cleaned.columns)}")
print(f"--- Time: {elapsed:.3f}s")
print(f"--- Dtypes:\n{cleaned.dtypes.value_counts()}")

# Check 1: No object columns
obj_cols = cleaned.select_dtypes(exclude=[np.number]).columns.tolist()
print(f"\n[CHECK 1] Object columns remaining: {obj_cols}")
assert len(obj_cols) == 0, f"FAIL: Object columns found: {obj_cols}"
print("  PASS - 100% numeric")

# Check 2: No NaNs
nan_count = cleaned.isnull().sum().sum()
print(f"\n[CHECK 2] NaN count: {nan_count}")
assert nan_count == 0, f"FAIL: NaNs found: {nan_count}"
print("  PASS - 0 NaNs")

# Check 3: Target preserved
print(f"\n[CHECK 3] Target in output: {'target' in cleaned.columns}")
assert "target" in cleaned.columns, "FAIL: Target column missing"
print("  PASS - Target preserved")

# Check 4: Log messages exist
print(f"\n[CHECK 4] Log messages count: {len(log_messages)}")
assert len(log_messages) > 0, "FAIL: No log messages"
print("  PASS - Logging works")

# Check 5: Log is valid JSON
log_json = json.dumps(log_messages)
json.loads(log_json)  # Will throw if invalid
print(f"\n[CHECK 5] Log JSON valid: True (length: {len(log_json)})")
print("  PASS - Valid JSON")

# Check 6: Reasonable number of features
n_features = cleaned.shape[1] - 1  # minus target
print(f"\n[CHECK 6] Features remaining: {n_features}")
if n_features < 3:
    print(f"  WARNING: Only {n_features} features remaining - too many dropped!")
else:
    print("  PASS - Reasonable feature count")

print("\n" + "=" * 60)
print("ALL LOG MESSAGES:")
print("=" * 60)
for i, msg in enumerate(log_messages, 1):
    print(f"  {i}. {msg}")

print(f"\n--- Performance: {elapsed:.3f}s")
print("=" * 60)
