#!/usr/bin/env bash
# =============================================================================
# Fetch the IBM AML HI-Small dataset.
# =============================================================================
# The dataset is published on Kaggle by IBM Research:
#   https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml
#
# This script uses the Kaggle CLI to download the HI-Small variant
# (~5 million transactions, ~1.2 GB unzipped). It verifies the resulting
# file matches the expected schema before declaring success so a corrupt
# or partial download fails immediately rather than at training time.
#
# Prerequisites:
#   1. pip install kaggle  (already pinned in requirements.txt)
#   2. Kaggle credentials at ~/.kaggle/kaggle.json with permissions 600.
#      Create at https://www.kaggle.com/settings/account → "Create New API Token".

set -euo pipefail

# Resolve the project root so the script works regardless of where it is invoked from.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
RAW_DIR="${PROJECT_ROOT}/data/raw"
TARGET_FILE="${RAW_DIR}/HI-Small_Trans.csv"

# Bail out early if the file is already present. We do not silently
# overwrite because regenerating the file would invalidate any model
# previously trained against the original snapshot.
if [[ -f "${TARGET_FILE}" ]]; then
    echo "Dataset already present at ${TARGET_FILE}. Skipping download."
    echo "Delete the file manually to force a fresh download."
    exit 0
fi

# Verify the Kaggle CLI is installed and configured before attempting
# the download. A clear error here saves the operator from a confusing
# 'command not found' failure mid-script.
if ! command -v kaggle >/dev/null 2>&1; then
    echo "ERROR: kaggle CLI not found." >&2
    echo "Install with: pip install kaggle" >&2
    echo "Then provision credentials per the instructions in this script's header." >&2
    exit 1
fi

# Kaggle's CLI supports two authentication file formats:
#   * Legacy ``~/.kaggle/kaggle.json`` (deprecated by Kaggle, kept for
#     backward compatibility with operators on older setups).
#   * Modern ``~/.kaggle/access_token`` (the named-API-token system
#     introduced in Kaggle CLI 1.8.0).
# We accept either. The CLI itself picks whichever is present, so no
# further action is needed once one of the two exists.
if [[ ! -f "${HOME}/.kaggle/kaggle.json" && ! -f "${HOME}/.kaggle/access_token" ]]; then
    echo "ERROR: Kaggle credentials not found." >&2
    echo "Expected either ~/.kaggle/kaggle.json (legacy) or" >&2
    echo "~/.kaggle/access_token (recommended)." >&2
    echo "Create a token at https://www.kaggle.com/settings/account and" >&2
    echo "save with chmod 600. The 'API Tokens (Recommended)' section is" >&2
    echo "the future-proof choice; legacy is being deprecated." >&2
    exit 1
fi

mkdir -p "${RAW_DIR}"

echo "Downloading IBM AML HI-Small to ${RAW_DIR}..."
echo "This is ~1.2 GB and will take a few minutes on a typical connection."

# The -f flag scopes the download to the HI-Small files only; the
# Kaggle dataset also contains HI-Medium and HI-Large which are large
# enough to be undesirable as an accidental download.
kaggle datasets download \
    -d ealtman2019/ibm-transactions-for-anti-money-laundering-aml \
    -f HI-Small_Trans.csv \
    -p "${RAW_DIR}" \
    --unzip

# Sanity check: the file exists and the header row matches the schema
# the loader expects. If a Kaggle release upstream renames a column the
# rest of the pipeline will fail in non-obvious ways; catching the
# rename here surfaces the issue at the right layer.
if [[ ! -f "${TARGET_FILE}" ]]; then
    echo "ERROR: Download completed but ${TARGET_FILE} is missing." >&2
    exit 1
fi

# The raw CSV intentionally has two columns both literally named
# ``Account`` — one for the source bank's account, one for the
# destination's. pandas disambiguates these to ``Account`` and
# ``Account.1`` at read time, which is what the rest of the codebase
# expects. This shell-level validator runs *before* pandas touches the
# file (via ``head -n 1``), so it compares against the raw,
# pre-disambiguation header.
EXPECTED_HEADER="Timestamp,From Bank,Account,To Bank,Account,Amount Received,Receiving Currency,Amount Paid,Payment Currency,Payment Format,Is Laundering"
OBSERVED_HEADER="$(head -n 1 "${TARGET_FILE}")"

if [[ "${OBSERVED_HEADER}" != "${EXPECTED_HEADER}" ]]; then
    echo "ERROR: Downloaded file has an unexpected header." >&2
    echo "  Expected: ${EXPECTED_HEADER}" >&2
    echo "  Observed: ${OBSERVED_HEADER}" >&2
    echo "The upstream dataset schema may have changed. Update both this" >&2
    echo "script's EXPECTED_HEADER and src/data/loader.py RAW_COLUMNS to match." >&2
    exit 1
fi

ROW_COUNT="$(wc -l < "${TARGET_FILE}")"
echo "Download complete. ${TARGET_FILE}"
echo "  Row count: ${ROW_COUNT} (includes header)"
echo "Schema validated. Ready for src.data.DataLoader."
