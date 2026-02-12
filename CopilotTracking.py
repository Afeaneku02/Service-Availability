# -*- coding: utf-8 -*-
"""
Service Sub Evaluator (Single-Sheet Output) with MCC-based backfill, strict ALL-OF SS rule,
and near-zero -> 1 rounding for Calc Forecast / Cal ISO.

Created on Sun Jan  4 21:05:58 2026
@author: F6KN5K9
"""# -*- coding: utf-8 -*-
"""
Service Sub Evaluator (Single-Sheet Output), with robust column normalization first:
- Normalize/alias columns for both Primary and SPM before any computation.
- Vectorized SPM aggregation and SS evaluation (performance).
- MCC-based backfill with near-zero->1 rounding.
- Strict ALL-OF SS rule preserved, strict 'No FYF' toggle preserved.
- ON_ORDER awareness; status = 'ON_ORDER' when neither On-Hand nor In-Transit cover but ON_ORDER > 0.
- Notes show amounts checked: [req=R, onhand=H, intransit=I, on_order=O].
"""

import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
import math
from datetime import datetime

# =========================
# Configs
# =========================
CLASS_FACTORS = {
    "A": 0.90,
    "B": 0.80,
    "C": 0.75,
    "D": 0.60,
    "E": 0.50,
    "I": 1.00,
}
DEFAULT_FACTOR = 0.70
STRICT_NO_FYF_LT_ZERO = True  # toggle in UI

# =========================
# Utilities
# =========================
def todays_date() -> str:
    now = datetime.now()
    return f"{now.month}/{now.day}"

def _round_whole(x: float, method: str = "round", *, raw: float | None = None) -> int:
    if pd.isna(x):
        return 0
    if raw is None:
        raw = x
    if method == "ceil":
        out = int(math.ceil(x))
    elif method == "floor":
        out = int(math.floor(x))
    else:
        out = int(round(x))
    # near-zero -> 1
    if (raw is not None) and (raw > 0) and (out == 0):
        return 1
    return out

def normalize_inv_class(cls_val) -> str | None:
    if pd.isna(cls_val):
        return None
    s = str(cls_val).strip().upper()
    return s if s else None

def alias_depot_series(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.strip()
    return s.replace({
        "3Q01": "2003",
        # Add more alias rules if needed:
        "02303": "2303",
    })

def alias_depot(depot: str) -> str:
    return alias_depot_series(pd.Series([depot])).iat[0]

def split_service_sub_to_list(val) -> list[str]:
    if pd.isna(val):
        return []
    s = str(val).strip()
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        s = s[1:-1]
    if not s:
        return []
    parts = [p.strip() for p in s.split(",") if p.strip()]
    seen = set()
    out = []
    for p in parts:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out

# =========================
# Column normalization (critical)
# =========================
def _norm(s: str) -> str:
    """Normalize header token: upper, trim, replace hyphens with underscores, collapse whitespace."""
    return (
        str(s)
        .replace("\r", " ")
        .replace("\n", " ")
        .strip()
        .upper()
        .replace("-", "_")
    )

def normalize_columns(df: pd.DataFrame, is_primary: bool) -> pd.DataFrame:
    """
    Normalize incoming headers to canonical names used by the evaluator.
    is_primary=True -> primary sheet mappings
    is_primary=False -> SPM mappings
    """
    if df is None or df.empty:
        return df

    # Build map normalized->actual
    norm_to_actual = { _norm(c): c for c in df.columns }

    def pick(variants: list[str], new_name: str | None = None, create_zero: bool = False):
        """Find first matching normalized column; optionally create zero column if requested."""
        for v in variants:
            key = _norm(v)
            if key in norm_to_actual:
                actual = norm_to_actual[key]
                if (new_name is not None) and (new_name != actual):
                    df.rename(columns={actual: new_name}, inplace=True)
                elif new_name is None:
                    # keep original if new_name not provided
                    pass
                return new_name or actual
        if create_zero and new_name is not None:
            df[new_name] = 0
            return new_name
        return None

    if is_primary:
        # Canonical names we will use downstream
        pick(["PART"], "Part")
        pick(["SERVICE SUB", "SERVICESUB"], "Service Sub")
        pick(["DIST CTR CODE", "DIST_CTR_CODE", "DIST CTR", "DIST_CTR", "DIST C", "DIST CENTER"], "Dist Ctr Code")
        pick(["INVTY CLASS", "INV CLASS", "INV CLS"], "INVTY CLASS")
        pick(["SBOM QTY", "SBOM", "SBOM_QTY"], "SBOM Qty")
        pick(["ENG VOL BY DIST", "ENG VOL DIST", "ENG VOL BY DISTR", "ENG VOL BY DC"], "Eng Vol by Dist")
        pick(["CALC FORECAST", "CALC_FORECAST"], "Calc Forecast")
        pick(["CAL ISO", "CALC ISO", "CAL_ISO"], "Cal ISO")
        pick(["MCC", "MATERIAL CRITICAL CODE"], "MCC")
        pick(["NOT TO BE STOCKED", "NOT_TO_BE_STOCKED"], "Not to be Stocked")
        pick(["STATUS"], "Status")
    else:
        # SPM canonical names
        pick(["PRT NUM", "PRT_NUM", "PART NUMBER", "PARTNUMBER"], "PRT NUM")
        pick(["DIST_CTR", "DIST CTR", "DIST_C", "DIST CENTER", "DIST_CTR CODE"], "DIST_CTR")
        pick(["INV CLS", "INV CLASS", "INVTY CLASS"], "INV CLS")
        pick(["AVAIL", "AVAILABLE_QTY", "AVAILABLE"], "AVAIL")
        pick(["ON_HAND", "ON HAND", "OH_QTY", "OH"], "ON_HAND")  # supplemental; not used for coverage formula, but retained
        pick(["INHOUSE", "IN_HOUSE"], "INHOUSE")
        pick(["WIP"], "WIP")
        pick(["INTRANSIT", "IN TRANSIT", "IN_TRANSIT"], "INTRANSIT")
        pick(["ON_ORDER", "ON ORDER", "ON-ORDER", "ONORDER", "ON ORDER QTY"], "ON_ORDER", create_zero=True)
        pick(["NETBO", "NET_BO"], "NETBO")
        pick(["FCST_1_YR DOS", "FCST 1 YR DOS", "FCST_1YR_DOS", "FCST1YR DOS"], "FCST_1_YR DOS")

    # Finally, trim all string values for key columns we know we will join on
    if is_primary:
        if "Part" in df.columns:
            df["Part"] = df["Part"].astype(str).str.strip()
        if "Dist Ctr Code" in df.columns:
            df["Dist Ctr Code"] = alias_depot_series(df["Dist Ctr Code"])
    else:
        if "PRT NUM" in df.columns:
            df["PRT NUM"] = df["PRT NUM"].astype(str).str.strip()
        if "DIST_CTR" in df.columns:
            df["DIST_CTR"] = alias_depot_series(df["DIST_CTR"])

    return df

# =========================
# MCC backfill
# =========================
def backfill_calc_fields(primary_df: pd.DataFrame, mcc_map: dict[int, float], rounding_method: str = "round") -> pd.DataFrame:
    df = primary_df.copy()

    # Ensure canonical columns exist after normalization
    has_mcc = "MCC" in df.columns
    if not has_mcc:
        # Nothing to do
        return df

    if "Calc Forecast" not in df.columns:
        df["Calc Forecast"] = pd.NA
    if "Cal ISO" not in df.columns:
        df["Cal ISO"] = pd.NA
    if "Calc Fields Note" not in df.columns:
        df["Calc Fields Note"] = ""

    sbom = pd.to_numeric(df.get("SBOM Qty"), errors="coerce")
    engv = pd.to_numeric(df.get("Eng Vol by Dist"), errors="coerce")
    mcc = pd.to_numeric(df.get("MCC"), errors="coerce")

    need_cf = df["Calc Forecast"].isna()
    need_ci = df["Cal ISO"].isna()

    mcc_pct = mcc.map(lambda x: mcc_map.get(int(x), np.nan) if not pd.isna(x) else np.nan)
    can_calc = (need_cf | need_ci) & sbom.notna() & engv.notna() & mcc_pct.notna()

    cf_raw = (sbom * engv * mcc_pct).where(can_calc)

    # Fill CF
    cf_fill = []
    for raw, cur, need in zip(cf_raw, df["Calc Forecast"], need_cf):
        if need and not pd.isna(raw):
            cf_fill.append(_round_whole(raw, rounding_method, raw=raw))
        else:
            cf_fill.append(cur)
    df["Calc Forecast"] = pd.to_numeric(cf_fill, errors="coerce").fillna(0).round(0).astype("Int64")

    # Cal ISO from CF
    final_cf = df["Calc Forecast"].astype(float)
    ci_raw = (final_cf * 0.25).where(need_ci & final_cf.notna())

    ci_fill = []
    for raw, cur, need in zip(ci_raw, df["Cal ISO"], need_ci):
        if need and not pd.isna(raw):
            ci_fill.append(_round_whole(raw, rounding_method, raw=raw))
        else:
            ci_fill.append(cur)
    df["Cal ISO"] = pd.to_numeric(ci_fill, errors="coerce").fillna(0).round(0).astype("Int64")

    # Notes for rows changed
    notes = []
    for i in range(len(df)):
        bits = []
        if need_cf.iat[i] and not pd.isna(cf_raw.iat[i]):
            bits.append(f"Calc Forecast={df['Calc Forecast'].iat[i]}")
        if need_ci.iat[i] and not pd.isna(ci_raw.iat[i]):
            bits.append(f"Cal ISO={df['Cal ISO'].iat[i]}")
        if bits:
            notes.append("Backfilled via MCC rule: " + ", ".join(bits))
        else:
            notes.append(df["Calc Fields Note"].iat[i] if "Calc Fields Note" in df.columns else "")
    df["Calc Fields Note"] = notes

    return df

# =========================
# SPM aggregation (vectorized)
# =========================
@st.cache_data(show_spinner=False)
def build_spm_agg(spm_raw: pd.DataFrame) -> pd.DataFrame:
    # Normalize headers FIRST (critical)
    df = normalize_columns(spm_raw.copy(), is_primary=False)

    # Validate required canonical columns
    required = ["PRT NUM", "DIST_CTR", "INV CLS", "AVAIL", "INHOUSE", "WIP", "INTRANSIT", "NETBO", "FCST_1_YR DOS", "ON_ORDER"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"SPM missing required columns after normalization: {missing}")

    # Coerce numerics
    for col in ["AVAIL", "WIP", "INHOUSE", "INTRANSIT", "NETBO", "FCST_1_YR DOS", "ON_ORDER", "ON_HAND"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # Aggregate (sum all additive, max forecast, first non-null INV CLS)
    def first_non_null_upper(s: pd.Series):
        s2 = s.dropna().astype(str).str.upper().str.strip()
        return s2.iloc[0] if not s2.empty else pd.NA

    agg_dict = {
        "AVAIL": "sum",
        "WIP": "sum",
        "INHOUSE": "sum",
        "INTRANSIT": "sum",
        "NETBO": "sum",
        "FCST_1_YR DOS": "max",
        "INV CLS": first_non_null_upper,
        "ON_ORDER": "sum",
    }
    if "ON_HAND" in df.columns:
        agg_dict["ON_HAND"] = "sum"

    agg = (
        df.groupby(["PRT NUM", "DIST_CTR"], as_index=False)
          .agg(agg_dict)
    )

    return agg

# =========================
# Vectorized evaluation
# =========================
def compute_material_metrics(df_mat: pd.DataFrame, spm_agg: pd.DataFrame) -> pd.DataFrame:
    """
    df_mat required columns: _group_id, material, depot, inv_cls_primary, cal_iso
    Returns columns: required, onhand, intransit_net, on_order, on_hand_raw(optional), forecast_val, forecasted, no_fyf, stocked_by, status, note
    """
    left = df_mat.copy()
    left["material"] = left["material"].astype(str).str.strip()
    left["depot"] = alias_depot_series(left["depot"])

    # Join
    spm = spm_agg.rename(columns={"PRT NUM": "material", "DIST_CTR": "depot"})
    merged = left.merge(spm, on=["material", "depot"], how="left", suffixes=("", "_spm"))

    # Requirement from INV class (primary preferred; fallback SPM INV)
    inv_cls_used = []
    factors = []
    for p_cls, s_cls in zip(merged.get("inv_cls_primary"), merged.get("INV CLS")):
        used = normalize_inv_class(p_cls) or normalize_inv_class(s_cls)
        inv_cls_used.append(used)
        factors.append(CLASS_FACTORS.get(used, DEFAULT_FACTOR) if used else DEFAULT_FACTOR)

    merged["inv_cls_used"] = inv_cls_used
    merged["rec_factor"] = factors
    cal_iso_num = pd.to_numeric(merged["cal_iso"], errors="coerce").fillna(0.0)
    merged["required"] = (cal_iso_num * merged["rec_factor"]).clip(lower=0.0)

    # Numeric fields (may be NaN if no SPM row)
    avail = pd.to_numeric(merged.get("AVAIL"), errors="coerce")
    wip = pd.to_numeric(merged.get("WIP"), errors="coerce")
    inhouse = pd.to_numeric(merged.get("INHOUSE"), errors="coerce")
    intransit = pd.to_numeric(merged.get("INTRANSIT"), errors="coerce")
    netbo = pd.to_numeric(merged.get("NETBO"), errors="coerce")
    fcst = pd.to_numeric(merged.get("FCST_1_YR DOS"), errors="coerce")
    on_order = pd.to_numeric(merged.get("ON_ORDER"), errors="coerce")
    on_hand_raw = pd.to_numeric(merged.get("ON_HAND"), errors="coerce")  # optional, for reporting only

    have_spm = avail.notna() & wip.notna() & inhouse.notna() & intransit.notna() & netbo.notna()

    # Coverage model (same as original): On-Hand coverage from AVAIL+WIP+INHOUSE-NETBO
    onhand = ((avail.fillna(0) + wip.fillna(0) + inhouse.fillna(0)) - netbo.fillna(0)).clip(lower=0.0)
    intransit_net = (intransit.fillna(0) - netbo.fillna(0)).clip(lower=0.0)

    merged["onhand"] = onhand.where(have_spm, np.nan)
    merged["intransit_net"] = intransit_net.where(have_spm, np.nan)
    merged["on_order"] = on_order.fillna(0) if on_order is not None else 0
    merged["on_hand_raw"] = on_hand_raw.fillna(0) if on_hand_raw is not None else 0
    merged["forecast_val"] = fcst.where(have_spm, np.nan)

    # Flags
    f_flag = (merged["forecast_val"] > 0)
    no_fyf = (merged["forecast_val"] < 0) if STRICT_NO_FYF_LT_ZERO else (merged["forecast_val"] <= 0)
    merged["forecasted"] = f_flag.fillna(False)
    merged["no_fyf"] = no_fyf.fillna(False)

    # Coverage decisions (priority: On-Hand -> In-Transit -> ON_ORDER -> Not stocked)
    covers_onhand = (merged["onhand"] >= merged["required"]) & have_spm
    covers_transit = (merged["intransit_net"] >= merged["required"]) & have_spm
    has_on_order = (merged["on_order"] > 0) & have_spm

    merged["stocked_by"] = np.where(covers_onhand, "On-Hand",
                             np.where(covers_transit, "In-Transit",
                             np.where(~have_spm, None, None)))

    merged["status"] = np.where(~have_spm, "Review SPM (missing row)",
                         np.where(covers_onhand,
                                  np.where(merged["forecasted"], "Stocked&Forecasted", "Stocked No FYF"),
                                  np.where(covers_transit, "In-Transit",
                                           np.where(has_on_order, "ON_ORDER", "Not stocked"))))

    # Notes
    date_str = todays_date()
    def mk_note(row):
        if not have_spm.loc[row.name]:
            return f"{date_str}: {row['depot']}: {row['material']} SPM missing row"
        req = float(row["required"]) if pd.notna(row["required"]) else 0.0
        oh = float(row["onhand"]) if pd.notna(row["onhand"]) else 0.0
        it = float(row["intransit_net"]) if pd.notna(row["intransit_net"]) else 0.0
        oo = float(row["on_order"]) if pd.notna(row["on_order"]) else 0.0

        if row["status"] in ("Stocked&Forecasted", "Stocked No FYF"):
            return f"{date_str}: {row['depot']}: {row['material']} Covered by On-Hand [req={math.ceil(req)}, onhand={math.ceil(oh)}, intransit={math.ceil(it)}, on_order={math.ceil(oo)}]"
        if row["status"] == "In-Transit":
            return f"{date_str}: {row['depot']}: {row['material']} Covered by In-Transit [req={math.ceil(req)}, onhand={math.ceil(oh)}, intransit={math.ceil(it)}, on_order={math.ceil(oo)}]"
        if row["status"] == "ON_ORDER":
            return f"{date_str}: {row['depot']}: {row['material']} Not covered yet; ON_ORDER in place [req={math.ceil(req)}, onhand={math.ceil(oh)}, intransit={math.ceil(it)}, on_order={math.ceil(oo)}]"
        best_supply = max(oh, it)
        deficit = max(0, math.ceil(req - best_supply))
        return f"{date_str}: {row['depot']}: {row['material']} need ~{deficit} pcs [req={math.ceil(req)}, onhand={math.ceil(oh)}, intransit={math.ceil(it)}, on_order={math.ceil(oo)}]"

    merged["note"] = merged.apply(mk_note, axis=1)
    return merged

def evaluate_rows_vectorized(primary_df: pd.DataFrame, spm_agg: pd.DataFrame) -> pd.DataFrame:
    base = primary_df.copy()
    base = normalize_columns(base, is_primary=True)  # ensure canonical columns
    base["_group_id"] = np.arange(len(base))

    # Normalize essential fields
    base["Part"] = base["Part"].astype(str).str.strip()
    base["Dist Ctr Code"] = alias_depot_series(base["Dist Ctr Code"])
    base["Cal ISO"] = pd.to_numeric(base.get("Cal ISO"), errors="coerce").fillna(0)
    base["INVTY CLASS"] = base.get("INVTY CLASS").apply(normalize_inv_class)

    # Explode SS list
    ss_lists = base.get("Service Sub", pd.Series([""] * len(base))).apply(split_service_sub_to_list)
    has_ss = ss_lists.map(len) > 0

    df_ss = base.loc[has_ss, ["_group_id", "Dist Ctr Code", "INVTY CLASS", "Cal ISO"]].copy()
    df_ss = df_ss.join(pd.DataFrame({"ss_list": ss_lists[has_ss].values}), how="left")
    df_ss = df_ss.explode("ss_list").rename(columns={
        "Dist Ctr Code": "depot",
        "INVTY CLASS": "inv_cls_primary",
        "Cal ISO": "cal_iso",
        "ss_list": "material"
    })

    # Compute metrics for SS
    if not df_ss.empty:
        mat_ss = compute_material_metrics(df_ss, spm_agg)
    else:
        mat_ss = pd.DataFrame(columns=[
            "_group_id","material","depot","inv_cls_primary","cal_iso","required","onhand","intransit_net",
            "on_order","on_hand_raw","forecast_val","forecasted","no_fyf","stocked_by","status","note","inv_cls_used"
        ])

    # Helper: summarize SS group
    def summarize_ss(group: pd.DataFrame) -> dict:
        any_missing = (group["status"] == "Review SPM (missing row)").any()
        eval_materials = ", ".join(group["material"].astype(str))
        req_summary = "; ".join(f"{m}={math.ceil(r) if pd.notna(r) else 'N/A'}"
                                for m, r in zip(group["material"], group["required"]))
        onhand_summary = "; ".join(f"{m}={math.ceil(v) if pd.notna(v) else 'N/A'}"
                                   for m, v in zip(group["material"], group["onhand"]))
        intransit_summary = "; ".join(f"{m}={math.ceil(v) if pd.notna(v) else 'N/A'}"
                                      for m, v in zip(group["material"], group["intransit_net"]))
        onorder_summary = "; ".join(f"{m}={math.ceil(v) if pd.notna(v) else 'N/A'}"
                                    for m, v in zip(group["material"], group["on_order"]))
        notes = "; ".join(group["note"].astype(str))
        any_forecasted = "Yes" if group["forecasted"].any() else "No"

        if any_missing:
            return {
                "Status": "Review SPM (missing row)",
                "Evaluated Materials": eval_materials,
                "Required (pcs)": req_summary,
                "On-Hand (AVAIL+WIP+INHOUSE-NETBO)": onhand_summary,
                "In-Transit net (INTRANSIT-NETBO)": intransit_summary,
                "On-Order": onorder_summary,
                "Forecasted": any_forecasted,
                "Notes": notes
            }

        covers = group["stocked_by"].isin(["On-Hand", "In-Transit"])
        if covers.all():
            any_onhand = (group["stocked_by"] == "On-Hand").any()
            any_intransit = (group["stocked_by"] == "In-Transit").any()
            transit_only = any_intransit and not any_onhand
            any_f = group["forecasted"].any()
            status = "In-Transit" if transit_only else ("Stocked&Forecasted" if any_f else "Stocked No FYF")
            return {
                "Status": status,
                "Evaluated Materials": eval_materials,
                "Required (pcs)": req_summary,
                "On-Hand (AVAIL+WIP+INHOUSE-NETBO)": onhand_summary,
                "In-Transit net (INTRANSIT-NETBO)": intransit_summary,
                "On-Order": onorder_summary,
                "Forecasted": "Yes" if any_f else "No",
                "Notes": notes + "; Service Sub group sufficient (all cover)"
            }

        # Some SS do NOT cover -> need part fallback
        def approx_deficit_row(row):
            req = row["required"] if pd.notna(row["required"]) else 0
            best_supply = max(row["onhand"] if pd.notna(row["onhand"]) else 0,
                              row["intransit_net"] if pd.notna(row["intransit_net"]) else 0)
            return max(0, math.ceil(req - best_supply))
        not_cover = group[~covers]
        ss_deficits = ", ".join(f"{m} short ~{approx_deficit_row(r)}"
                                for m, r in zip(not_cover["material"], not_cover.to_dict("records")))
        return {"_need_part_fallback": True, "_ss_note": f"Service Sub not enough: {ss_deficits}",
                "_eval_materials": eval_materials, "_req_summary": req_summary,
                "_onhand_summary": onhand_summary, "_intransit_summary": intransit_summary,
                "_onorder_summary": onorder_summary, "_notes": notes, "_any_forecasted": any_forecasted}

    ss_summary = {}
    if not mat_ss.empty:
        for gid, grp in mat_ss.groupby("_group_id"):
            ss_summary[gid] = summarize_ss(grp)

    enriched = base.copy()
    for col in ["Evaluated Materials", "Required (pcs)", "On-Hand (AVAIL+WIP+INHOUSE-NETBO)",
                "In-Transit net (INTRANSIT-NETBO)", "On-Order", "Forecasted", "Status", "Notes"]:
        enriched[col] = ""

    # Handle SS rows
    ss_gids = set(ss_summary.keys())
    for i, row in enriched.iterrows():
        gid = row["_group_id"]
        if gid in ss_gids:
            info = ss_summary[gid]
            if "Status" in info:  # final
                enriched.at[i, "Evaluated Materials"] = info["Evaluated Materials"]
                enriched.at[i, "Required (pcs)"] = info["Required (pcs)"]
                enriched.at[i, "On-Hand (AVAIL+WIP+INHOUSE-NETBO)"] = info["On-Hand (AVAIL+WIP+INHOUSE-NETBO)"]
                enriched.at[i, "In-Transit net (INTRANSIT-NETBO)"] = info["In-Transit net (INTRANSIT-NETBO)"]
                enriched.at[i, "On-Order"] = info["On-Order"]
                enriched.at[i, "Forecasted"] = info["Forecasted"]
                enriched.at[i, "Status"] = info["Status"]
                enriched.at[i, "Notes"] = info["Notes"]
            else:
                # Fallback evaluate Part
                df_part = pd.DataFrame({
                    "_group_id": [gid],
                    "material": [row["Part"]],
                    "depot": [row["Dist Ctr Code"]],
                    "inv_cls_primary": [row["INVTY CLASS"]],
                    "cal_iso": [row["Cal ISO"]],
                })
                pe = compute_material_metrics(df_part, spm_agg).iloc[0].to_dict()

                # Merge summaries (SS + Part)
                grp_ss = mat_ss[mat_ss["_group_id"] == gid]
                eval_materials = info["_eval_materials"] + f", {row['Part']}"
                req_summary = info["_req_summary"] + f"; {row['Part']}={math.ceil(pe['required']) if pd.notna(pe['required']) else 'N/A'}"
                onhand_summary = info["_onhand_summary"] + f"; {row['Part']}={math.ceil(pe['onhand']) if pd.notna(pe['onhand']) else 'N/A'}"
                intransit_summary = info["_intransit_summary"] + f"; {row['Part']}={math.ceil(pe['intransit_net']) if pd.notna(pe['intransit_net']) else 'N/A'}"
                onorder_summary = info["_onorder_summary"] + f"; {row['Part']}={math.ceil(pe['on_order']) if pd.notna(pe['on_order']) else 'N/A'}"
                any_forecasted = "Yes" if (grp_ss["forecasted"].any() or bool(pe["forecasted"])) else "No"

                def covers(pe_row: dict) -> bool:
                    return pe_row.get("stocked_by") in ("On-Hand", "In-Transit")

                if str(pe["status"]).startswith("Review SPM"):
                    status = "Review SPM (missing row)"
                    notes = info["_notes"] + f"; {info['_ss_note']}; " + pe["note"]
                elif covers(pe):
                    status = "In-Transit" if pe["stocked_by"] == "In-Transit" else ("Stocked&Forecasted" if pe["forecasted"] else "Stocked No FYF")
                    notes = info["_notes"] + f"; {info['_ss_note']}; fallback=Part via {pe['stocked_by']}"
                else:
                    status = "ON_ORDER" if pe["status"] == "ON_ORDER" else "Not stocked"
                    notes = info["_notes"] + f"; {info['_ss_note']}"

                enriched.at[i, "Evaluated Materials"] = eval_materials
                enriched.at[i, "Required (pcs)"] = req_summary
                enriched.at[i, "On-Hand (AVAIL+WIP+INHOUSE-NETBO)"] = onhand_summary
                enriched.at[i, "In-Transit net (INTRANSIT-NETBO)"] = intransit_summary
                enriched.at[i, "On-Order"] = onorder_summary
                enriched.at[i, "Forecasted"] = any_forecasted
                enriched.at[i, "Status"] = status
                enriched.at[i, "Notes"] = notes

    # No-SS rows → Part only
    no_ss_mask = ~has_ss
    if no_ss_mask.any():
        df_part_only = enriched.loc[no_ss_mask, ["_group_id", "Part", "Dist Ctr Code", "INVTY CLASS", "Cal ISO"]].rename(
            columns={"Part": "material", "Dist Ctr Code": "depot", "INVTY CLASS": "inv_cls_primary", "Cal ISO": "cal_iso"}
        )
        part_eval = compute_material_metrics(df_part_only, spm_agg).set_index("_group_id")
        for idx in enriched.index[no_ss_mask]:
            gid = enriched.at[idx, "_group_id"]
            if gid not in part_eval.index:
                continue
            pe = part_eval.loc[gid].to_dict()

            part_mat = enriched.at[idx, "Part"]
            req_summary = f"{part_mat}={math.ceil(pe['required']) if pd.notna(pe['required']) else 'N/A'}"
            oh_summary = f"{part_mat}={math.ceil(pe['onhand']) if pd.notna(pe['onhand']) else 'N/A'}"
            it_summary = f"{part_mat}={math.ceil(pe['intransit_net']) if pd.notna(pe['intransit_net']) else 'N/A'}"
            oo_summary = f"{part_mat}={math.ceil(pe['on_order']) if pd.notna(pe['on_order']) else 'N/A'}"
            status = pe["status"]
            if status == "In Transit":
                status = "In-Transit"
            enriched.at[idx, "Evaluated Materials"] = part_mat
            enriched.at[idx, "Required (pcs)"] = req_summary
            enriched.at[idx, "On-Hand (AVAIL+WIP+INHOUSE-NETBO)"] = oh_summary
            enriched.at[idx, "In-Transit net (INTRANSIT-NETBO)"] = it_summary
            enriched.at[idx, "On-Order"] = oo_summary
            enriched.at[idx, "Forecasted"] = "Yes" if pe["forecasted"] else "No"
            enriched.at[idx, "Status"] = status
            enriched.at[idx, "Notes"] = pe["note"]

    return enriched.drop(columns=["_group_id"])

# =========================
# Streamlit UI
# =========================
st.set_page_config(page_title="Service Sub Evaluator", page_icon="📦", layout="wide")

def main():
    st.title("Service Sub Evaluator — normalized columns first")

    st.markdown("""
**Upload two files:**
- **Primary sheet** (required canonical fields after normalization): `Part`, `Service Sub`, `Dist Ctr Code`, `INVTY CLASS`, `SBOM Qty`, `Eng Vol by Dist`, `Calc Forecast` (opt), `Cal ISO` (opt), `Not to be Stocked` (opt), `Status` (opt), `MCC` or `Material Critical Code`.
- **SPM data** (required canonical fields after normalization): `PRT NUM`, `DIST_CTR`, `INV CLS`, `AVAIL`, `INHOUSE`, `WIP`, `INTRANSIT`, `NETBO`, `FCST_1_YR DOS`, and (optional but recommended) `ON_ORDER`; if present, `ON_HAND` is preserved for reporting.
    """)

    with st.expander("Settings"):
        strict = st.checkbox("Strict 'No FYF' (FCST_1_YR DOS < 0)", value=True)
        globals()["STRICT_NO_FYF_LT_ZERO"] = strict

        rounding_method = st.selectbox(
            "Whole-number rounding for Calc Forecast / Cal ISO",
            options=["round", "ceil", "floor"],
            index=0,
            help="Near-zero positive values are forced to 1."
        )

        st.caption("Edit MCC% if needed:")
        mcc_df = pd.DataFrame({
            "MCC": [11,12,13,14,15,20,21,50,51,52,54,55],
            "Percent": [0.10,0.10,0.10,0.10,0.06,0.06,0.06,0.03,0.03,0.03,0.03,0.03]
        })
        mcc_df = st.data_editor(mcc_df, num_rows="dynamic", use_container_width=True, height=250)
        mcc_map = {}
        for _, r in mcc_df.iterrows():
            try:
                k = int(r["MCC"]); v = float(r["Percent"])
                if v >= 0: mcc_map[k] = v
            except Exception:
                continue

    col1, col2 = st.columns(2)
    with col1:
        primary_file = st.file_uploader("Primary sheet (.xlsx, .csv)", type=["xlsx", "csv"])
    with col2:
        spm_file = st.file_uploader("SPM data (.xlsx, .csv)", type=["xlsx", "csv"])

    run = st.button("Run Evaluation")

    if run:
        if not primary_file or not spm_file:
            st.error("Please upload both files.")
            return

        try:
            with st.spinner("Reading files…"):
                # Read + normalize primary
                if primary_file.name.lower().endswith(".csv"):
                    primary_df_raw = pd.read_csv(primary_file)
                else:
                    primary_df_raw = pd.read_excel(primary_file, engine="openpyxl")
                primary_df = normalize_columns(primary_df_raw.copy(), is_primary=True)

                # Backfill via MCC (after normalization)
                primary_df = backfill_calc_fields(primary_df, mcc_map, rounding_method)

                # Read SPM
                if spm_file.name.lower().endswith(".csv"):
                    spm_df_raw = pd.read_csv(spm_file)
                else:
                    spm_df_raw = pd.read_excel(spm_file, engine="openpyxl")

            with st.spinner("Aggregating SPM and evaluating…"):
                spm_agg = build_spm_agg(spm_df_raw)  # build_spm_agg internally normalizes SPM first
                enriched_df = evaluate_rows_vectorized(primary_df, spm_agg)

            st.success("Done ✅")
            st.dataframe(enriched_df, use_container_width=True, height=600)

            # Download Excel
            buffer = BytesIO()
            with pd.ExcelWriter(buffer, engine="xlsxwriter") as w:
                enriched_df.to_excel(w, sheet_name="Evaluations", index=False)
                ws = w.sheets["Evaluations"]
                nrows, ncols = enriched_df.shape
                ws.autofilter(0, 0, nrows, ncols - 1)
                ws.freeze_panes(1, 1)
            buffer.seek(0)
            st.download_button(
                label="Download evaluations.xlsx",
                data=buffer.getvalue(),
                file_name="evaluations.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

        except Exception as e:
            st.error(f"Error: {e}")

if __name__ == "__main__":

    main()
