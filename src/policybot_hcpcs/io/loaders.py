"""Load and normalize CSV data into Pydantic models."""
import logging
from typing import Optional

import pandas as pd

from src.policybot_hcpcs.models.schemas import HCPCSCode, PolicyInput

logger = logging.getLogger(__name__)


def load_policies(csv_path: str, limit: Optional[int] = None) -> list[PolicyInput]:
    """
    Load policies from policies_cleaned.csv.

    Expected columns: policy_id, policy_name, plan_name, plan_state,
    policy_type, source_url, cleaned_policy_text, original_text_length.
    """
    df = pd.read_csv(csv_path, nrows=limit)
    logger.info(f"Loaded {len(df)} rows from {csv_path}")

    if "cleaned_policy_text" not in df.columns:
        raise ValueError(
            f"Input CSV '{csv_path}' is missing required column 'cleaned_policy_text'. "
            f"Found columns: {', '.join(df.columns)}. "
            f"Use policies_cleaned.csv as --input (not the labels file)."
        )

    policies = []
    for _, row in df.iterrows():
        raw_text = str(row.get("cleaned_policy_text", ""))
        if not raw_text.strip():
            logger.warning(f"Skipping policy {row.get('policy_id')} — empty text")
            continue

        normalized = " ".join(raw_text.split())

        policies.append(PolicyInput(
            policy_id=str(row["policy_id"]),
            policy_text=normalized,
        ))

    logger.info(f"Parsed {len(policies)} valid policies")
    return policies


def load_hcpcs_catalog(csv_path: str) -> list[HCPCSCode]:
    """
    Load HCPCS reference data.

    Expected columns: code, description.
    """
    df = pd.read_csv(csv_path)
    logger.info(f"Loaded {len(df)} HCPCS codes from {csv_path}")

    return [
        HCPCSCode(
            code=str(row["code"]).strip(),
            description=str(row["description"]).strip(),
        )
        for _, row in df.iterrows()
        if pd.notna(row.get("code")) and pd.notna(row.get("description"))
    ]


def load_labels(csv_path: str) -> dict[str, list[str]]:
    """
    Load ground truth labels from policies_cleaned_labels.csv.

    Expected columns: policy_id, hcpcs_codes, icd_10_codes.
    Returns a dict mapping policy_id -> list of HCPCS code strings.
    """
    df = pd.read_csv(csv_path)
    logger.info(f"Loaded {len(df)} label rows from {csv_path}")

    labels: dict[str, list[str]] = {}
    for _, row in df.iterrows():
        policy_id = str(row["policy_id"])
        hcpcs_raw = str(row.get("hcpcs_codes", ""))
        if pd.isna(row.get("hcpcs_codes")) or not hcpcs_raw.strip():
            labels[policy_id] = []
        else:
            # Codes may be comma or pipe delimited
            codes = [c.strip() for c in hcpcs_raw.replace("|", ",").split(",") if c.strip()]
            labels[policy_id] = codes

    return labels
