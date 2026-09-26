# ================================================================
# AMAZON ML CHALLENGE 2026 — COMPLETE EDA
# Business Entity Resolution
#
# Run from either:
#   D:\Amazon-ML-Challenge-2026\
# or:
#   D:\Amazon-ML-Challenge-2026\dataset\
#
# Expected structure:
# dataset/
#   train/
#       train_source1.tsv
#       train_source2.tsv
#       train_source3.tsv
#       train_ground_truth.tsv
#   test/
#       test_source1.tsv
#       test_source2.tsv
#       test_source3.tsv
#
# Output:
#   analysis/EDA/
#       source_summary.csv
#       source_summary.json
#       ground_truth_summary.json
#       positive_pair_analysis.json
#       blocking_recall.json
#       detailed_eda_report.txt
#       plots/*.png
#       examples/*.csv
#
# Designed for LARGE TSV files:
# - Reads source files in chunks
# - Does not load all 26M records into pandas at once
# - Uses exact/hash-based EDA where possible
# ================================================================

import os
import re
import json
import math
import time
import unicodedata
from pathlib import Path
from collections import Counter, defaultdict

import pandas as pd
import numpy as np

# Plotting
import matplotlib.pyplot as plt

# ------------------------------------------------
# 0. CONFIGURATION
# ------------------------------------------------

HERE = Path.cwd()

# Detect project/dataset location
if (HERE / "train").exists() and (HERE / "test").exists():
    DATASET = HERE
elif (HERE / "dataset").exists():
    DATASET = HERE / "dataset"
else:
    # Change this manually if required
    DATASET = Path(r"D:\Amazon-ML-Challenge-2026\dataset")

TRAIN = DATASET / "train"
TEST = DATASET / "test"

PROJECT_ROOT = DATASET.parent
OUT = PROJECT_ROOT / "analysis" / "EDA"
PLOTS = OUT / "plots"
EXAMPLES = OUT / "examples"

OUT.mkdir(parents=True, exist_ok=True)
PLOTS.mkdir(parents=True, exist_ok=True)
EXAMPLES.mkdir(parents=True, exist_ok=True)

CHUNK_SIZE = 250_000
TOP_K = 30
MAX_EXAMPLES = 100

FILES = {
    "train_source1": TRAIN / "train_source1.tsv",
    "train_source2": TRAIN / "train_source2.tsv",
    "train_source3": TRAIN / "train_source3.tsv",
    "test_source1": TEST / "test_source1.tsv",
    "test_source2": TEST / "test_source2.tsv",
    "test_source3": TEST / "test_source3.tsv",
    "train_ground_truth": TRAIN / "train_ground_truth.tsv",
}

print("=" * 80)
print("AMAZON ML CHALLENGE 2026 — COMPLETE EDA")
print("=" * 80)
print("Dataset :", DATASET)
print("Output  :", OUT)
print("Chunk   :", f"{CHUNK_SIZE:,}")
print("=" * 80)


# ------------------------------------------------
# 1. BASIC HELPERS
# ------------------------------------------------

def safe_str(x):
    if x is None:
        return ""
    if isinstance(x, float) and math.isnan(x):
        return ""
    return str(x)


def normalize_text(x):
    """
    Conservative normalization.
    Keeps Unicode letters/digits but removes punctuation.
    """
    x = unicodedata.normalize("NFKC", safe_str(x)).casefold()

    # Remove zero-width characters
    x = re.sub(r"[\u200b-\u200d\ufeff]", "", x)

    # Treat & as 'and'
    x = x.replace("&", " and ")

    # Replace punctuation with spaces
    x = re.sub(r"[^\w\s]", " ", x, flags=re.UNICODE)

    # Collapse whitespace
    return re.sub(r"\s+", " ", x).strip()


def tokenize(x):
    return normalize_text(x).split()


def char_ngrams(x, n=3):
    x = re.sub(r"\W+", "", normalize_text(x), flags=re.UNICODE)
    if not x:
        return set()
    if len(x) < n:
        return {x}
    return {x[i:i+n] for i in range(len(x) - n + 1)}


def jaccard(a, b):
    a = set(a)
    b = set(b)

    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0

    return len(a & b) / len(a | b)


def token_jaccard(a, b):
    return jaccard(tokenize(a), tokenize(b))


def char3_jaccard(a, b):
    return jaccard(char_ngrams(a, 3), char_ngrams(b, 3))


def levenshtein_ratio(a, b):
    """
    Pure Python similarity.
    Used only on sampled positive pairs to avoid expensive full-dataset cost.
    """
    a = normalize_text(a)
    b = normalize_text(b)

    if a == b:
        return 1.0
    if not a or not b:
        return 0.0

    if len(a) < len(b):
        a, b = b, a

    previous = list(range(len(b) + 1))

    for i, ca in enumerate(a, 1):
        current = [i]

        for j, cb in enumerate(b, 1):
            insert = current[j - 1] + 1
            delete = previous[j] + 1
            replace = previous[j - 1] + (ca != cb)

            current.append(min(insert, delete, replace))

        previous = current

    distance = previous[-1]
    return 1 - distance / max(len(a), len(b))


def detect_script(x):
    counts = Counter()

    for ch in safe_str(x):
        if (
            ch.isspace()
            or ch.isdigit()
            or unicodedata.category(ch).startswith("P")
        ):
            continue

        name = unicodedata.name(ch, "")

        if "LATIN" in name:
            counts["Latin"] += 1
        elif "DEVANAGARI" in name:
            counts["Devanagari"] += 1
        elif "ARABIC" in name:
            counts["Arabic"] += 1
        elif "CYRILLIC" in name:
            counts["Cyrillic"] += 1
        elif "GREEK" in name:
            counts["Greek"] += 1
        else:
            counts["Other"] += 1

    return counts.most_common(1)[0][0] if counts else "NoLetters"


def percentile_stats(values):
    if not values:
        return {}

    x = np.asarray(values, dtype=float)

    return {
        "min": int(np.min(x)),
        "p01": float(np.percentile(x, 1)),
        "p05": float(np.percentile(x, 5)),
        "p25": float(np.percentile(x, 25)),
        "median": float(np.median(x)),
        "p75": float(np.percentile(x, 75)),
        "p95": float(np.percentile(x, 95)),
        "p99": float(np.percentile(x, 99)),
        "max": int(np.max(x)),
        "mean": float(np.mean(x)),
    }


def plot_bar(counter_or_dict, title, xlabel, filename, top=20):
    if isinstance(counter_or_dict, Counter):
        data = counter_or_dict.most_common(top)
    else:
        data = sorted(
            counter_or_dict.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top]

    if not data:
        return

    labels = [str(x[0]) for x in data][::-1]
    values = [x[1] for x in data][::-1]

    plt.figure(figsize=(10, 6))
    plt.barh(labels, values)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.tight_layout()
    plt.savefig(PLOTS / filename, dpi=150)
    plt.close()


# ------------------------------------------------
# 2. COLUMN DETECTION
# ------------------------------------------------

def find_column(columns, possible_names):
    lower = {str(c).lower(): c for c in columns}

    for name in possible_names:
        if name.lower() in lower:
            return lower[name.lower()]

    for c in columns:
        for name in possible_names:
            if name.lower() in str(c).lower():
                return c

    return None


def detect_columns(columns):
    return {
        "id": find_column(
            columns,
            ["entity_id", "id", "business_id", "source_id"]
        ),
        "name": find_column(
            columns,
            ["business_name", "name", "business"]
        ),
        "address": find_column(
            columns,
            ["business_address", "address", "location"]
        ),
        "country": find_column(
            columns,
            ["country", "country_code", "country_name"]
        ),
    }


# ------------------------------------------------
# 3. SOURCE EDA
# ------------------------------------------------

def analyze_source(name, path):
    print("\n" + "-" * 80)
    print("Analyzing:", name)
    print("-" * 80)

    if not path.exists():
        print("FILE NOT FOUND:", path)
        return {"source": name, "error": "FILE_NOT_FOUND"}

    start = time.time()

    rows = 0

    ids = set()
    raw_names = set()
    raw_addresses = set()

    name_freq = Counter()
    address_freq = Counter()
    country_freq = Counter()

    norm_name_freq = Counter()
    norm_address_freq = Counter()

    norm_name_variants = defaultdict(set)
    norm_address_variants = defaultdict(set)

    missing = Counter()

    name_lengths = []
    address_lengths = []

    name_scripts = Counter()
    address_scripts = Counter()

    columns = None
    inferred = None

    for chunk in pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        na_filter=False,
        chunksize=CHUNK_SIZE,
        on_bad_lines="warn",
    ):

        rows += len(chunk)

        if columns is None:
            columns = list(chunk.columns)
            inferred = detect_columns(columns)

            print("Columns:", columns)
            print("Detected:", inferred)

        # ------------------------------------------
        # IDs
        # ------------------------------------------

        if inferred["id"]:
            for value in chunk[inferred["id"]]:
                if value:
                    ids.add(value)

        # ------------------------------------------
        # BUSINESS NAME
        # ------------------------------------------

        if inferred["name"]:

            for value in chunk[inferred["name"]]:

                value = safe_str(value)

                if not value:
                    missing["business_name"] += 1
                    continue

                raw_names.add(value)
                name_freq[value] += 1

                normalized = normalize_text(value)

                norm_name_freq[normalized] += 1
                norm_name_variants[normalized].add(value)

                name_lengths.append(len(value))
                name_scripts[detect_script(value)] += 1

        # ------------------------------------------
        # BUSINESS ADDRESS
        # ------------------------------------------

        if inferred["address"]:

            for value in chunk[inferred["address"]]:

                value = safe_str(value)

                if not value:
                    missing["business_address"] += 1
                    continue

                raw_addresses.add(value)
                address_freq[value] += 1

                normalized = normalize_text(value)

                norm_address_freq[normalized] += 1
                norm_address_variants[normalized].add(value)

                address_lengths.append(len(value))
                address_scripts[detect_script(value)] += 1

        # ------------------------------------------
        # COUNTRY
        # ------------------------------------------

        if inferred["country"]:

            for value in chunk[inferred["country"]]:

                value = safe_str(value)

                if not value:
                    missing["country"] += 1
                else:
                    country_freq[value] += 1

    result = {
        "source": name,
        "path": str(path),
        "rows": rows,
        "columns": columns,
        "detected_columns": inferred,

        "unique_ids": len(ids),

        "unique_raw_names": len(raw_names),
        "unique_raw_addresses": len(raw_addresses),

        "unique_normalized_names": len(norm_name_freq),
        "unique_normalized_addresses": len(norm_address_freq),

        "missing": dict(missing),

        "country_distribution": dict(country_freq),

        "name_length_stats": percentile_stats(name_lengths),
        "address_length_stats": percentile_stats(address_lengths),

        "name_scripts": dict(name_scripts),
        "address_scripts": dict(address_scripts),

        "raw_name_repeated_values": sum(
            v > 1 for v in name_freq.values()
        ),

        "raw_address_repeated_values": sum(
            v > 1 for v in address_freq.values()
        ),

        "normalized_name_repeated_values": sum(
            v > 1 for v in norm_name_freq.values()
        ),

        "normalized_address_repeated_values": sum(
            v > 1 for v in norm_address_freq.values()
        ),

        "name_normalization_collision_values": sum(
            len(v) > 1 for v in norm_name_variants.values()
        ),

        "address_normalization_collision_values": sum(
            len(v) > 1 for v in norm_address_variants.values()
        ),

        "generic_name_frequency": {
            f">={k}": sum(v >= k for v in name_freq.values())
            for k in [2, 5, 10, 50, 100]
        },

        "top_raw_names": name_freq.most_common(TOP_K),
        "top_raw_addresses": address_freq.most_common(TOP_K),
        "top_normalized_names": norm_name_freq.most_common(TOP_K),
        "top_normalized_addresses": norm_address_freq.most_common(TOP_K),

        "top_name_normalization_collisions": [
            {
                "normalized": normalized,
                "raw_variants": sorted(list(variants))[:20],
                "variant_count": len(variants),
                "frequency": norm_name_freq[normalized],
            }
            for normalized, variants
            in sorted(
                norm_name_variants.items(),
                key=lambda x: norm_name_freq[x[0]],
                reverse=True
            )[:TOP_K]
        ],

        "top_address_normalization_collisions": [
            {
                "normalized": normalized,
                "raw_variants": sorted(list(variants))[:20],
                "variant_count": len(variants),
                "frequency": norm_address_freq[normalized],
            }
            for normalized, variants
            in sorted(
                norm_address_variants.items(),
                key=lambda x: norm_address_freq[x[0]],
                reverse=True
            )[:TOP_K]
        ],

        "elapsed_seconds": round(time.time() - start, 2),
    }

    print("Rows:", f"{rows:,}")
    print("Unique IDs:", f"{len(ids):,}")
    print("Unique raw names:", f"{len(raw_names):,}")
    print("Unique normalized names:", f"{len(norm_name_freq):,}")
    print("Unique raw addresses:", f"{len(raw_addresses):,}")
    print("Unique normalized addresses:", f"{len(norm_address_freq):,}")
    print("Countries:", dict(country_freq))
    print("Missing:", dict(missing))
    print("Time:", result["elapsed_seconds"], "sec")

    return result


# ------------------------------------------------
# 4. LOAD TRAIN/TEST SOURCE EDA
# ------------------------------------------------

source_results = {}

for source_name in [
    "train_source1",
    "train_source2",
    "train_source3",
    "test_source1",
    "test_source2",
    "test_source3",
]:

    source_results[source_name] = analyze_source(
        source_name,
        FILES[source_name]
    )


# ------------------------------------------------
# 5. SOURCE SUMMARY TABLE
# ------------------------------------------------

summary_rows = []

for source, r in source_results.items():

    rows = r.get("rows", 0)

    missing_name = r.get("missing", {}).get(
        "business_name", 0
    )

    missing_address = r.get("missing", {}).get(
        "business_address", 0
    )

    missing_country = r.get("missing", {}).get(
        "country", 0
    )

    summary_rows.append({
        "source": source,
        "rows": rows,
        "unique_ids": r.get("unique_ids", 0),
        "unique_raw_names": r.get("unique_raw_names", 0),
        "unique_normalized_names":
            r.get("unique_normalized_names", 0),
        "unique_raw_addresses":
            r.get("unique_raw_addresses", 0),
        "unique_normalized_addresses":
            r.get("unique_normalized_addresses", 0),
        "missing_name": missing_name,
        "missing_address": missing_address,
        "missing_country": missing_country,
        "name_repeat_values":
            r.get("raw_name_repeated_values", 0),
        "address_repeat_values":
            r.get("raw_address_repeated_values", 0),
        "name_normalization_collisions":
            r.get("name_normalization_collision_values", 0),
        "address_normalization_collisions":
            r.get("address_normalization_collision_values", 0),
    })

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(OUT / "source_summary.csv", index=False)

print("\nSOURCE SUMMARY")
print(summary_df.to_string(index=False))


# ------------------------------------------------
# 6. COUNTRY PLOTS
# ------------------------------------------------

for source, r in source_results.items():

    countries = Counter(
        r.get("country_distribution", {})
    )

    plot_bar(
        countries,
        f"{source} — Country Distribution",
        "Records",
        f"{source}_country.png",
        top=10,
    )


# ------------------------------------------------
# 7. GENERATE EDA PERCENTAGE REPORT
# ------------------------------------------------

country_percentages = {}

for source, r in source_results.items():

    total = max(r.get("rows", 1), 1)

    country_percentages[source] = {
        country: count / total
        for country, count
        in r.get("country_distribution", {}).items()
    }


# ------------------------------------------------
# 8. GROUND TRUTH ANALYSIS
# ------------------------------------------------

def load_ground_truth():

    path = FILES["train_ground_truth"]

    if not path.exists():
        return {
            "error": "GROUND_TRUTH_NOT_FOUND"
        }

    print("\n" + "=" * 80)
    print("GROUND TRUTH ANALYSIS")
    print("=" * 80)

    df = pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
    )

    columns = list(df.columns)

    print("Columns:", columns)
    print("Rows:", f"{len(df):,}")

    # Expected:
    # source1_entity_id
    # matched_entity_ids

    source1_col = find_column(
        columns,
        [
            "source1_entity_id",
            "source_1_entity_id",
            "entity_id",
            "id",
        ],
    )

    if source1_col is None:
        source1_col = columns[0]

    match_cols = [
        c for c in columns
        if any(
            term in str(c).lower()
            for term in [
                "matched",
                "match",
                "source2",
                "source_2",
                "source3",
                "source_3",
            ]
        )
    ]

    if not match_cols:
        match_cols = columns[1:]

    print("Source1 column:", source1_col)
    print("Match columns:", match_cols)

    multiplicity = Counter()
    total_links = 0

    source2_links = 0
    source3_links = 0

    mapping = {}

    for _, row in df.iterrows():

        s1_id = safe_str(row[source1_col]).strip()

        matched = []

        for col in match_cols:

            value = safe_str(row[col]).strip()

            if not value:
                continue

            # Support comma / semicolon / pipe separated IDs
            parts = [
                x.strip()
                for x in re.split(r"[|;,]", value)
                if x.strip()
            ]

            if not parts:
                parts = [value]

            for target_id in parts:

                matched.append(target_id)
                total_links += 1

                target_lower = target_id.lower()

                if target_lower.startswith("s2-"):
                    source2_links += 1

                elif target_lower.startswith("s3-"):
                    source3_links += 1

                else:
                    # If IDs do not contain prefix, don't guess.
                    pass

        mapping[s1_id] = matched
        multiplicity[len(matched)] += 1

    result = {
        "rows": len(df),
        "unique_source1_entities": len(mapping),
        "total_match_links": total_links,

        "source2_links": source2_links,
        "source3_links": source3_links,

        "zero_match_entities": multiplicity[0],
        "one_match_entities": multiplicity[1],

        "multi_match_entities":
            sum(
                count
                for k, count in multiplicity.items()
                if k >= 2
            ),

        "multiplicity_distribution":
            dict(sorted(multiplicity.items())),
    }

    print("\nGround Truth Summary")
    print(json.dumps(result, indent=2))

    with open(
        OUT / "ground_truth_summary.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(result, f, indent=2)

    # Save mapping separately
    with open(
        OUT / "ground_truth_mapping.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(mapping, f)

    # Multiplicity plot
    plot_bar(
        multiplicity,
        "Ground Truth — Matches per Source1 Entity",
        "Number of Source1 entities",
        "ground_truth_multiplicity.png",
        top=20,
    )

    return result, mapping


gt_result, gt_mapping = load_ground_truth()


# ------------------------------------------------
# 9. LOAD RECORD LOOKUPS FOR POSITIVE PAIRS
# ------------------------------------------------
#
# IMPORTANT:
# This loads only the ID/name/address/country lookup dictionaries.
# It still contains millions of IDs, but avoids storing unnecessary
# columns. It is required for true-pair diagnostics.
# ------------------------------------------------

def load_lookup(source_name):

    path = FILES[source_name]

    result = {}

    if not path.exists():
        return result

    print("Loading lookup:", source_name)

    for chunk in pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        na_filter=False,
        chunksize=CHUNK_SIZE,
        on_bad_lines="warn",
    ):

        c = detect_columns(chunk.columns)

        if not c["id"]:
            continue

        for _, row in chunk.iterrows():

            entity_id = safe_str(
                row[c["id"]]
            )

            result[entity_id] = {
                "name":
                    safe_str(row[c["name"]])
                    if c["name"] else "",

                "address":
                    safe_str(row[c["address"]])
                    if c["address"] else "",

                "country":
                    safe_str(row[c["country"]])
                    if c["country"] else "",
            }

    print(
        source_name,
        "lookup records:",
        f"{len(result):,}"
    )

    return result


# ------------------------------------------------
# 10. TRUE POSITIVE PAIR DIAGNOSTICS
# ------------------------------------------------

def pair_diagnostics(a, b):

    an = normalize_text(a["name"])
    bn = normalize_text(b["name"])

    aa = normalize_text(a["address"])
    ba = normalize_text(b["address"])

    return {
        "name_exact_raw":
            bool(a["name"] and a["name"] == b["name"]),

        "name_exact_normalized":
            bool(an and an == bn),

        "address_exact_raw":
            bool(a["address"] and a["address"] == b["address"]),

        "address_exact_normalized":
            bool(aa and aa == ba),

        "country_exact":
            bool(
                a["country"]
                and b["country"]
                and normalize_text(a["country"])
                    == normalize_text(b["country"])
            ),

        "name_token_jaccard":
            token_jaccard(an, bn),

        "address_token_jaccard":
            token_jaccard(aa, ba),

        "name_char3_jaccard":
            char3_jaccard(an, bn),

        "address_char3_jaccard":
            char3_jaccard(aa, ba),

        "name_length_difference":
            abs(len(an) - len(bn)),

        "address_length_difference":
            abs(len(aa) - len(ba)),
    }


def positive_pair_analysis(gt_mapping):

    print("\n" + "=" * 80)
    print("POSITIVE / TRUE MATCH ANALYSIS")
    print("=" * 80)

    s1 = load_lookup("train_source1")
    s2 = load_lookup("train_source2")
    s3 = load_lookup("train_source3")

    counts = Counter()
    sums = Counter()

    examples = []

    missing_s1 = 0
    missing_target = 0

    for s1_id, targets in gt_mapping.items():

        if s1_id not in s1:
            missing_s1 += 1
            continue

        a = s1[s1_id]

        for target_id in targets:

            target = None
            target_source = None

            # Prefix is safest
            if target_id.startswith("S2-"):
                target = s2.get(target_id)
                target_source = "source2"

            elif target_id.startswith("S3-"):
                target = s3.get(target_id)
                target_source = "source3"

            else:
                # Do not guess if duplicate IDs existed.
                if target_id in s2:
                    target = s2[target_id]
                    target_source = "source2"
                elif target_id in s3:
                    target = s3[target_id]
                    target_source = "source3"

            if target is None:
                missing_target += 1
                continue

            d = pair_diagnostics(a, target)

            counts["total"] += 1

            for key in [
                "name_exact_raw",
                "name_exact_normalized",
                "address_exact_raw",
                "address_exact_normalized",
                "country_exact",
            ]:
                counts[key] += int(d[key])

            for key in [
                "name_token_jaccard",
                "address_token_jaccard",
                "name_char3_jaccard",
                "address_char3_jaccard",
                "name_length_difference",
                "address_length_difference",
            ]:
                sums[key] += d[key]

            if len(examples) < MAX_EXAMPLES:

                examples.append({
                    "source1_id": s1_id,
                    "target_id": target_id,
                    "target_source": target_source,
                    "s1_name": a["name"],
                    "target_name": target["name"],
                    "s1_address": a["address"],
                    "target_address": target["address"],
                    "s1_country": a["country"],
                    "target_country": target["country"],
                    **d,
                })

    total = counts["total"]

    result = {
        "total_evaluable_true_links": total,

        "missing_source1_ids": missing_s1,
        "missing_target_ids": missing_target,

        "rates": {
            key:
                counts[key] / total
                if total else 0
            for key in [
                "name_exact_raw",
                "name_exact_normalized",
                "address_exact_raw",
                "address_exact_normalized",
                "country_exact",
            ]
        },

        "mean_similarities": {
            key:
                sums[key] / total
                if total else 0
            for key in sums
        },

        "examples": examples,
    }

    print(
        json.dumps(
            {
                k: v
                for k, v in result.items()
                if k != "examples"
            },
            indent=2,
        )
    )

    with open(
        OUT / "positive_pair_analysis.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(result, f, indent=2)

    if examples:
        pd.DataFrame(examples).to_csv(
            EXAMPLES / "positive_pair_examples.csv",
            index=False,
        )

    return result


positive_result = positive_pair_analysis(gt_mapping)


# ------------------------------------------------
# 11. BLOCKING RECALL ANALYSIS
# ------------------------------------------------
#
# This is one of the MOST IMPORTANT parts of the EDA.
#
# We test whether true matches would be included by:
#   1. same country
#   2. exact raw name
#   3. exact normalized name
#   4. exact raw address
#   5. exact normalized address
#   6. normalized name OR address
#   7. normalized name AND country
#   8. normalized address AND country
#   9. normalized name AND normalized address
#
# This tells us which candidate-generation rules are safe.
# ------------------------------------------------

def blocking_recall(gt_mapping):

    print("\n" + "=" * 80)
    print("BLOCKING RECALL ANALYSIS")
    print("=" * 80)

    s1 = load_lookup("train_source1")
    s2 = load_lookup("train_source2")
    s3 = load_lookup("train_source3")

    rules = [
        "same_country",
        "exact_raw_name",
        "exact_normalized_name",
        "exact_raw_address",
        "exact_normalized_address",
        "name_or_address_normalized",
        "name_and_country",
        "address_and_country",
        "name_and_address_normalized",
    ]

    hits = Counter()
    source_hits = defaultdict(Counter)
    totals = Counter()

    for s1_id, targets in gt_mapping.items():

        a = s1.get(s1_id)

        if not a:
            continue

        for target_id in targets:

            if target_id.startswith("S2-"):
                b = s2.get(target_id)
                source = "source2"

            elif target_id.startswith("S3-"):
                b = s3.get(target_id)
                source = "source3"

            else:
                if target_id in s2:
                    b = s2[target_id]
                    source = "source2"
                elif target_id in s3:
                    b = s3[target_id]
                    source = "source3"
                else:
                    b = None
                    source = "unknown"

            if not b:
                continue

            totals["all"] += 1
            totals[source] += 1

            an = normalize_text(a["name"])
            bn = normalize_text(b["name"])

            aa = normalize_text(a["address"])
            ba = normalize_text(b["address"])

            same_country = bool(
                a["country"]
                and b["country"]
                and normalize_text(a["country"])
                    == normalize_text(b["country"])
            )

            raw_name = bool(
                a["name"]
                and a["name"] == b["name"]
            )

            normalized_name = bool(
                an and an == bn
            )

            raw_address = bool(
                a["address"]
                and a["address"] == b["address"]
            )

            normalized_address = bool(
                aa and aa == ba
            )

            rule_values = {
                "same_country":
                    same_country,

                "exact_raw_name":
                    raw_name,

                "exact_normalized_name":
                    normalized_name,

                "exact_raw_address":
                    raw_address,

                "exact_normalized_address":
                    normalized_address,

                "name_or_address_normalized":
                    normalized_name or normalized_address,

                "name_and_country":
                    normalized_name and same_country,

                "address_and_country":
                    normalized_address and same_country,

                "name_and_address_normalized":
                    normalized_name and normalized_address,
            }

            for rule, hit in rule_values.items():

                if hit:
                    hits[rule] += 1
                    source_hits[source][rule] += 1

    overall_recall = {
        rule:
            hits[rule] / totals["all"]
            if totals["all"] else 0
        for rule in rules
    }

    source2_recall = {
        rule:
            source_hits["source2"][rule]
            / totals["source2"]
            if totals["source2"] else 0
        for rule in rules
    }

    source3_recall = {
        rule:
            source_hits["source3"][rule]
            / totals["source3"]
            if totals["source3"] else 0
        for rule in rules
    }

    result = {
        "total_true_links": totals["all"],
        "source2_true_links": totals["source2"],
        "source3_true_links": totals["source3"],

        "overall_recall": overall_recall,
        "source2_recall": source2_recall,
        "source3_recall": source3_recall,

        "hit_counts": dict(hits),
    }

    print("\nOverall Blocking Recall")

    for rule, value in overall_recall.items():
        print(
            f"{rule:<35}",
            f"{value:.4%}"
        )

    with open(
        OUT / "blocking_recall.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(result, f, indent=2)

    # Plot
    plot_bar(
        overall_recall,
        "Blocking Rule Recall on True Matches",
        "Recall",
        "blocking_recall.png",
        top=20,
    )

    return result


blocking_result = blocking_recall(gt_mapping)


# ------------------------------------------------
# 12. WRITE DETAILED TEXT REPORT
# ------------------------------------------------

def pct(x):
    return f"{100*x:.2f}%"


report = []

report.append("=" * 90)
report.append("AMAZON ML CHALLENGE 2026 — COMPLETE EDA REPORT")
report.append("=" * 90)

report.append("")
report.append("1. DATASET OVERVIEW")
report.append("-" * 90)

for source in [
    "train_source1",
    "train_source2",
    "train_source3",
    "test_source1",
    "test_source2",
    "test_source3",
]:

    r = source_results.get(source, {})

    report.append(
        f"{source}: "
        f"{r.get('rows', 0):,} rows | "
        f"{r.get('unique_ids', 0):,} unique IDs | "
        f"{r.get('unique_raw_names', 0):,} raw names | "
        f"{r.get('unique_raw_addresses', 0):,} raw addresses"
    )


report.append("")
report.append("2. MISSING VALUES")
report.append("-" * 90)

for source, r in source_results.items():

    total = max(r.get("rows", 1), 1)

    missing = r.get("missing", {})

    if not missing:
        report.append(
            f"{source}: no missing values detected."
        )
    else:
        for col, count in missing.items():
            report.append(
                f"{source} | {col}: "
                f"{count:,} "
                f"({count/total:.2%})"
            )


report.append("")
report.append("3. COUNTRY DISTRIBUTION")
report.append("-" * 90)

for source, r in source_results.items():

    total = max(r.get("rows", 1), 1)

    report.append(f"\n{source}")

    for country, count in sorted(
        r.get("country_distribution", {}).items(),
        key=lambda x: x[1],
        reverse=True,
    ):

        report.append(
            f"  {country}: "
            f"{count:,} "
            f"({count/total:.2%})"
        )


report.append("")
report.append("4. NAME ANALYSIS")
report.append("-" * 90)

for source, r in source_results.items():

    report.append(f"\n{source}")

    report.append(
        f"  Raw repeated name values: "
        f"{r.get('raw_name_repeated_values', 0):,}"
    )

    report.append(
        f"  Normalized repeated name values: "
        f"{r.get('normalized_name_repeated_values', 0):,}"
    )

    report.append(
        f"  Normalization collision values: "
        f"{r.get('name_normalization_collision_values', 0):,}"
    )

    report.append(
        f"  Generic names >=2: "
        f"{r.get('generic_name_frequency', {}).get('>=2', 0):,}"
    )

    report.append(
        f"  Generic names >=5: "
        f"{r.get('generic_name_frequency', {}).get('>=5', 0):,}"
    )

    report.append(
        f"  Generic names >=10: "
        f"{r.get('generic_name_frequency', {}).get('>=10', 0):,}"
    )


report.append("")
report.append("5. ADDRESS ANALYSIS")
report.append("-" * 90)

for source, r in source_results.items():

    report.append(f"\n{source}")

    report.append(
        f"  Raw repeated address values: "
        f"{r.get('raw_address_repeated_values', 0):,}"
    )

    report.append(
        f"  Normalized repeated address values: "
        f"{r.get('normalized_address_repeated_values', 0):,}"
    )

    report.append(
        f"  Address normalization collisions: "
        f"{r.get('address_normalization_collision_values', 0):,}"
    )


report.append("")
report.append("6. GROUND TRUTH")
report.append("-" * 90)

for key, value in gt_result.items():
    report.append(
        f"{key}: {value}"
    )


report.append("")
report.append("7. TRUE MATCH DIAGNOSTICS")
report.append("-" * 90)

report.append(
    f"Evaluable true links: "
    f"{positive_result.get('total_evaluable_true_links', 0):,}"
)

for key, value in positive_result.get("rates", {}).items():
    report.append(
        f"{key}: {pct(value)}"
    )

report.append("")

for key, value in positive_result.get(
    "mean_similarities", {}
).items():

    report.append(
        f"mean {key}: {value:.4f}"
    )


report.append("")
report.append("8. BLOCKING RECALL")
report.append("-" * 90)

for rule, value in blocking_result.get(
    "overall_recall", {}
).items():

    report.append(
        f"{rule:<35} {pct(value)}"
    )


report.append("")
report.append("9. MODELING IMPLICATIONS")
report.append("-" * 90)

report.append(
    "• Do not assume one-to-one matching."
)

report.append(
    "• Do not use business name alone as the final decision."
)

report.append(
    "• Keep raw and normalized fields."
)

report.append(
    "• Missing addresses mean address cannot be mandatory."
)

report.append(
    "• Normalization collisions mean aggressive normalization "
    "can create false candidates."
)

report.append(
    "• Candidate generation must prioritize recall."
)

report.append(
    "• Final matching should use multiple name/address/country features."
)

report.append(
    "• Threshold must be tuned for F0.5 rather than accuracy."
)

report.append(
    "• Final inference must allow zero, one, or multiple matches."
)

report.append("")
report.append("=" * 90)
report.append("END OF EDA")
report.append("=" * 90)

report_path = OUT / "detailed_eda_report.txt"

report_path.write_text(
    "\n".join(report),
    encoding="utf-8",
)

print("\nEDA report written to:")
print(report_path)


# ------------------------------------------------
# 13. SAVE COMPLETE JSON
# ------------------------------------------------

complete = {
    "configuration": {
        "dataset": str(DATASET),
        "chunk_size": CHUNK_SIZE,
    },

    "sources": source_results,

    "ground_truth": gt_result,

    "positive_pair_analysis": positive_result,

    "blocking_recall": blocking_result,
}

with open(
    OUT / "COMPLETE_EDA.json",
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        complete,
        f,
        ensure_ascii=False,
        indent=2,
    )


# ------------------------------------------------
# 14. FINAL SCREEN SUMMARY
# ------------------------------------------------

print("\n")
print("=" * 90)
print("EDA COMPLETE")
print("=" * 90)

print("\nImportant files:")

for p in [
    OUT / "source_summary.csv",
    OUT / "ground_truth_summary.json",
    OUT / "positive_pair_analysis.json",
    OUT / "blocking_recall.json",
    OUT / "detailed_eda_report.txt",
    OUT / "COMPLETE_EDA.json",
]:

    print(" ", p)

print("\nPlots:")
print(" ", PLOTS)

print("\nNext step:")
print("Use blocking_recall.json to design candidate generation.")
print("Use positive_pair_analysis.json to design matching features.")
print("=" * 90)
