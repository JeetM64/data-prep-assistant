import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis


def analyze_features(df: pd.DataFrame) -> dict:
    """"
    For each column produces:
    - Basic stats (dtype, missing, unique)
    - For numeric: mean, std, min, max, skewness, kurtosis,
                   IQR, range, coefficient of variation,
                   distribution shape, scaling suggestion
    - For categorical: dominance, entropy, cardinality ratio,
                       encoding suggestion, bias warning
    - Column role hint: is this likely a feature, target, or ID?
    """

    report = {}
    n_rows = len(df)

    for col in df.columns:
        series = df[col]
        data = {}

        # ── BASIC INFO ───────────────────────────────────────────────────
        data["dtype"] = str(series.dtype)
        data["missing_count"] = int(series.isnull().sum())
        data["missing_percent"] = round(series.isnull().mean() * 100, 2)
        data["unique_values"] = int(series.nunique())
        data["unique_ratio"] = round(series.nunique() / max(n_rows, 1), 4)

        # ── COLUMN ROLE HINT ─────────────────────────────────────────────
        if series.nunique() == n_rows:
            data["likely_role"] = "identifier"
            data["role_warning"] = "Near-unique values — likely an ID column, not a feature"
        elif col == df.columns[-1]:
            data["likely_role"] = "target_candidate"
        else:
            data["likely_role"] = "feature"

        # ── NUMERIC ANALYSIS ─────────────────────────────────────────────
        if pd.api.types.is_numeric_dtype(series):
            clean = series.dropna()

            if len(clean) == 0:
                data["warning"] = "All values missing"
                report[col] = data
                continue

            mean_val = float(clean.mean())
            std_val = float(clean.std())
            min_val = float(clean.min())
            max_val = float(clean.max())
            q1 = float(clean.quantile(0.25))
            q3 = float(clean.quantile(0.75))
            iqr = q3 - q1
            skewness = float(clean.skew())
            kurt = float(clean.kurtosis())

            data["mean"] = round(mean_val, 4)
            data["median"] = round(float(clean.median()), 4)
            data["std"] = round(std_val, 4)
            data["min"] = round(min_val, 4)
            data["max"] = round(max_val, 4)
            data["range"] = round(max_val - min_val, 4)
            data["q1"] = round(q1, 4)
            data["q3"] = round(q3, 4)
            data["iqr"] = round(iqr, 4)
            data["skewness"] = round(skewness, 4)
            data["kurtosis"] = round(kurt, 4)

            # Coefficient of variation — spread relative to mean
            data["coefficient_of_variation"] = round(
                std_val / (abs(mean_val) + 1e-9), 4
            )

            # Distribution shape
            if abs(skewness) < 0.5:
                data["distribution_shape"] = "approximately_normal"
            elif skewness > 1.5:
                data["distribution_shape"] = "heavy_right_skew"
            elif skewness > 0.5:
                data["distribution_shape"] = "mild_right_skew"
            elif skewness < -1.5:
                data["distribution_shape"] = "heavy_left_skew"
            else:
                data["distribution_shape"] = "mild_left_skew"

            # Outlier count (IQR method)
            iqr_outliers = int(
                ((clean < q1 - 1.5 * iqr) | (clean > q3 + 1.5 * iqr)).sum()
            )
            data["iqr_outlier_count"] = iqr_outliers
            data["iqr_outlier_percent"] = round(iqr_outliers / max(n_rows, 1) * 100, 2)

            # Binary numeric detection
            if set(clean.unique()).issubset({0, 1}):
                data["is_binary"] = True
                data["scaling_suggestion"] = "none — binary column"
            elif std_val > 50 or (max_val - min_val) > 100:
                data["is_binary"] = False
                data["scaling_suggestion"] = (
                    "RobustScaler" if iqr_outliers > n_rows * 0.05
                    else "StandardScaler"
                )
            else:
                data["is_binary"] = False
                data["scaling_suggestion"] = "optional — low variance"

            # Transformation suggestion
            if abs(skewness) > 2 and min_val >= 0:
                data["transformation_suggestion"] = "log1p — high skew, non-negative"
            elif abs(skewness) > 1 and min_val >= 0:
                data["transformation_suggestion"] = "sqrt — moderate skew"
            else:
                data["transformation_suggestion"] = "none needed"

        # ── CATEGORICAL ANALYSIS ─────────────────────────────────────────
        else:
            value_counts = series.value_counts(normalize=True)
            top_ratio = float(value_counts.iloc[0]) if len(value_counts) > 0 else 0
            n_unique = series.nunique()

            data["top_category"] = str(value_counts.index[0]) if len(value_counts) > 0 else None
            data["top_category_percent"] = round(top_ratio * 100, 2)
            data["cardinality_ratio"] = round(n_unique / max(n_rows, 1), 4)

            # Shannon entropy — diversity of categories
            probs = value_counts.values
            entropy = float(-np.sum(probs * np.log2(probs + 1e-9)))
            data["category_entropy"] = round(entropy, 4)

            # Bias warning
            if top_ratio > 0.9:
                data["bias_warning"] = (
                    f"SEVERE: '{value_counts.index[0]}' dominates {top_ratio:.0%} "
                    "of values — near-zero variance, likely uninformative"
                )
            elif top_ratio > 0.8:
                data["bias_warning"] = (
                    f"MODERATE: '{value_counts.index[0]}' appears in {top_ratio:.0%} "
                    "of rows — possible class imbalance"
                )

            # Encoding suggestion
            if n_unique == 2:
                data["encoding_suggestion"] = "LabelEncoding — binary column"
            elif n_unique <= 10:
                data["encoding_suggestion"] = "OneHotEncoding — low cardinality"
            elif n_unique <= 50:
                data["encoding_suggestion"] = "TargetEncoding — medium cardinality"
            else:
                data["encoding_suggestion"] = (
                    "HashEncoding or drop — very high cardinality "
                    f"({n_unique} unique values)"
                )

            # Top 5 categories
            data["top_5_categories"] = {
                str(k): round(float(v), 4)
                for k, v in value_counts.head(5).items()
            }

        report[col] = data

    return report