#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GAM Analysis - Northern Pike (Esox lucius) - Main script (Tasks 1-6b)
====================================================================
Repository script accompanying:
"Beyond the Fast-Slow Continuum: Environmental Organization of Growth
and Lifespan in Northern Pike Esox lucius across Lentic Systems"

Data        : 52 lentic populations (Esox reichertii excluded); expected file
              name 'Pike_data.xlsx' (any 'Pike*.xlsx' is auto-detected).
Environment : Google Colab or any Python >= 3.9
Dependencies: pandas, numpy, scipy, openpyxl, pygam

Install in Colab (first cell):
    !pip install pygam openpyxl

Structure
---------
Task 1  : response ~ GDD + Area + AvgDepth (+ k as smooth control) - individual p
Task 2  : response ~ single driver (GDD, Area, AvgDepth)
Task 3  : response ~ LS  (fast-slow continuum test)
Task 4  : joint effect of GDD + Area + AvgDepth (no k control), LRT
Task 5  : joint effect of GDD + Area + AvgDepth WITH k control, LRT
Task 6a : quantile-bin dispersion - Spearman trend + bootstrap CI + bin-count sensitivity
Task 6b : heteroscedasticity - |residuals| ~ s(GDD)+s(Area)+s(AvgDepth), dAIC evidence

Output  : GAM_Pike_Results_Main.xlsx (Tasks 1-5)
          GAM_Pike_Results_Full.xlsx (all sheets: Tasks1-5, Task6a_trends,
          Task6a_bins, Task6a_sensitivity, Task6b)

Methodological notes
--------------------
1. Water bodies reported by two independent sources (Erie, Waskesiu, Lough Rea,
   Lough Glore, Coeur d'Alene) were averaged into single population estimates
   in the data file (arithmetic means of numeric traits; the Lmax/Llim ratio
   recomputed from the averaged L_max and L_lim). The analysed file therefore
   contains n = 52 unique water bodies.
2. Lmax_Llim_ratio is always recomputed as L_max / L_lim * 100
   (three literature values were found to be misprinted in the source).
3. Non-breaking spaces / en-dashes / decimal commas in the data file
   are sanitized before numeric conversion.
"""

import glob
import pandas as pd
import numpy as np
from scipy.stats import f, spearmanr
import warnings
warnings.filterwarnings('ignore')

from pygam import LinearGAM, s

# ============================================================
# CONFIGURATION
# ============================================================
N_BOOT = 1000
SEED = 42
np.random.seed(SEED)

# data file: 'Pike_data.xlsx' preferred, otherwise any Pike*.xlsx in the directory
candidates = ['Pike_data.xlsx'] + sorted(glob.glob('Pike*.xlsx'))
FILE_PATH = next((c for c in candidates if not c.startswith('~$')), None)
assert FILE_PATH is not None, 'No data file found: put Pike_data.xlsx (or any Pike*.xlsx) here.'

# ============================================================
# 1. LOAD & CLEAN
# ============================================================
df_raw = pd.read_excel(FILE_PATH, sheet_name=0)
df_raw.columns = df_raw.columns.str.strip()
df_raw = df_raw.rename(columns={
    'T degrees-days': 'GDD',
    'Average depth': 'AvgDepth'
})
# sanitize: decimal commas -> dots, en-dashes -> minus, non-breaking spaces -> space
df_raw = df_raw.replace({'\xa0': ' ', ',': '.', '\u2013': '-'}, regex=True)

num_cols = ['GDD', 'Area', 'AvgDepth', 'Lmax_Llim_ratio', 'L_lim', 'k', 'L_max', 'LS']
for col in num_cols:
    df_raw[col] = pd.to_numeric(df_raw[col], errors='coerce')

required_cols = ['GDD', 'Area', 'AvgDepth', 'L_lim', 'L_max', 'k', 'LS', 'Lmax_Llim_ratio']
df_clean = df_raw.dropna(subset=required_cols).copy()

# recompute ratio from L_max and L_lim (do not trust the column: three misprints
# in Flinders & Bonar 2008 rows were corrected this way)
df_clean['Lmax_Llim_ratio'] = df_clean['L_max'] / df_clean['L_lim'] * 100

# Duplicates were already averaged into single estimates in the data file (n = 52)
df_avg = df_clean.copy()
n_total = len(df_avg)

print(f"Data file: {FILE_PATH}")
print(f"Loaded populations: {len(df_clean)} (unique water bodies, duplicates averaged)")
print(f"Sample size analysed: {n_total}")

# ============================================================
# 2. HELPERS
# ============================================================
def get_scalar(x):
    if isinstance(x, dict):
        return list(x.values())[0] if x else np.nan
    elif isinstance(x, (list, tuple)):
        return x[0]
    else:
        return x

def bootstrap_sd_ci(values, n_boot=N_BOOT):
    if len(values) < 3:
        return np.nan, np.nan
    boot_sds = []
    for _ in range(n_boot):
        sample = np.random.choice(values, size=len(values), replace=True)
        boot_sds.append(np.std(sample, ddof=1))
    return np.percentile(boot_sds, [2.5, 97.5])

# ============================================================
# 3. MODEL SETUP
# ============================================================
responses_all = ['k', 'Lmax_Llim_ratio', 'L_lim', 'L_max', 'LS']
drivers_3 = ['GDD', 'Area', 'AvgDepth']
results = []

# ============================================================
# TASK 1 — with k as smooth control (individual p-values)
# ============================================================
print("\n" + "="*60)
print("TASK 1: response ~ GDD + Area + AvgDepth + (k if response != k)")
print("="*60)

for response in responses_all:
    print(f"\nProcessing: {response}")
    if response == 'k':
        X = df_avg[drivers_3].values
        gam = LinearGAM(s(0) + s(1) + s(2)).fit(X, df_avg[response].values)
        pred_names = drivers_3
    else:
        X = df_avg[drivers_3 + ['k']].values
        gam = LinearGAM(s(0) + s(1) + s(2) + s(3)).fit(X, df_avg[response].values)
        pred_names = drivers_3 + ['k']

    aic = gam.statistics_['AIC']
    p_vals = gam.statistics_['p_values']
    edof = gam.statistics_['edof']
    r2 = gam.statistics_['pseudo_r2']
    r2_scalar = get_scalar(r2)

    for i, pred in enumerate(pred_names):
        p_val = get_scalar(p_vals[i]) if isinstance(p_vals, (list, tuple)) else get_scalar(p_vals)
        edf = get_scalar(edof[i]) if isinstance(edof, (list, tuple)) else get_scalar(edof)
        results.append({
            'Task': 'Task1_with_k_individual',
            'Response': response,
            'Predictor': pred,
            'p_value': p_val,
            'EDF': edf,
            'R_squared': r2_scalar,
            'AIC': aic,
            'n': len(df_avg)
        })
        print(f"  {pred}: p = {p_val:.6f}, EDF = {edf:.3f}")

# ============================================================
# TASK 2 — single driver models (no k)
# ============================================================
print("\n" + "="*60)
print("TASK 2: response ~ one driver (GDD, Area, AvgDepth)")
print("="*60)

for driver in drivers_3:
    for response in responses_all:
        print(f"\nProcessing: {response} ~ {driver}")
        X = df_avg[[driver]].values
        y = df_avg[response].values
        gam = LinearGAM(s(0)).fit(X, y)

        aic = gam.statistics_['AIC']
        p_vals = gam.statistics_['p_values']
        edof = gam.statistics_['edof']
        r2 = gam.statistics_['pseudo_r2']

        p_val = get_scalar(p_vals)
        edf = get_scalar(edof)
        r2_scalar = get_scalar(r2)

        results.append({
            'Task': 'Task2_single_driver',
            'Response': response,
            'Predictor': driver,
            'p_value': p_val,
            'EDF': edf,
            'R_squared': r2_scalar,
            'AIC': aic,
            'n': len(df_avg)
        })
        print(f"  p = {p_val:.6f}, EDF = {edf:.3f}, R2 = {r2_scalar:.3f}, AIC = {aic:.2f}")

# ============================================================
# TASK 3 — LS as driver
# ============================================================
print("\n" + "="*60)
print("TASK 3: response ~ LS (responses without LS)")
print("="*60)

responses_no_ls = ['Lmax_Llim_ratio', 'L_lim', 'k', 'L_max']

for response in responses_no_ls:
    print(f"\nProcessing: {response} ~ LS")
    X = df_avg[['LS']].values
    y = df_avg[response].values
    gam = LinearGAM(s(0)).fit(X, y)

    aic = gam.statistics_['AIC']
    p_vals = gam.statistics_['p_values']
    edof = gam.statistics_['edof']
    r2 = gam.statistics_['pseudo_r2']

    p_val = get_scalar(p_vals)
    edf = get_scalar(edof)
    r2_scalar = get_scalar(r2)

    results.append({
        'Task': 'Task3_LS',
        'Response': response,
        'Predictor': 'LS',
        'p_value': p_val,
        'EDF': edf,
        'R_squared': r2_scalar,
        'AIC': aic,
        'n': len(df_avg)
    })
    print(f"  p = {p_val:.6f}, EDF = {edf:.3f}, R2 = {r2_scalar:.3f}, AIC = {aic:.2f}")

# ============================================================
# TASK 4 — joint effect of three drivers (NO k control)
# ============================================================
print("\n" + "="*60)
print("TASK 4: joint effect of GDD + Area + AvgDepth (NO k control)")
print("="*60)

for response in responses_all:
    print(f"\nProcessing joint effect for: {response}")
    X = df_avg[drivers_3].values
    gam_full = LinearGAM(s(0) + s(1) + s(2)).fit(X, df_avg[response].values)
    gam_null = LinearGAM().fit(np.ones((len(df_avg), 1)), df_avg[response].values)

    aic = gam_full.statistics_['AIC']
    edof = gam_full.statistics_['edof']
    r2 = gam_full.statistics_['pseudo_r2']
    r2_scalar = get_scalar(r2)
    edf = get_scalar(edof) if isinstance(edof, (list, tuple)) else get_scalar(edof)

    try:
        ll_full = gam_full.statistics_['loglikelihood']
        ll_null = gam_null.statistics_['loglikelihood']
        lr_stat = -2 * (ll_null - ll_full)
        edf_null = get_scalar(gam_null.statistics_['edof'])
        df_diff = edf - edf_null if not np.isnan(edf) and not np.isnan(edf_null) else 1
        p_joint = 1 - f.cdf(lr_stat, df_diff, len(df_avg) - df_diff) if df_diff > 0 else np.nan
    except Exception:
        p_joint = np.nan

    results.append({
        'Task': 'Task4_joint_no_k',
        'Response': response,
        'Predictor': 'GDD + Area + AvgDepth (joint)',
        'p_value': p_joint,
        'EDF': edf,
        'R_squared': r2_scalar,
        'AIC': aic,
        'n': len(df_avg)
    })
    print(f"  Joint p = {p_joint:.6f}, EDF = {edf:.3f}, R2 = {r2_scalar:.3f}, AIC = {aic:.2f}")

# ============================================================
# TASK 5 — joint effect of three drivers WITH k control
# ============================================================
print("\n" + "="*60)
print("TASK 5: joint effect of GDD + Area + AvgDepth WITH k control")
print("="*60)

for response in responses_all:
    if response == 'k':
        continue
    else:
        print(f"\nProcessing joint effect for: {response}")
        X_full = df_avg[drivers_3 + ['k']].values
        gam_full = LinearGAM(s(0) + s(1) + s(2) + s(3)).fit(X_full, df_avg[response].values)

        X_k = df_avg[['k']].values
        gam_k = LinearGAM(s(0)).fit(X_k, df_avg[response].values)

        aic = gam_full.statistics_['AIC']
        edof = gam_full.statistics_['edof']
        r2 = gam_full.statistics_['pseudo_r2']
        r2_scalar = get_scalar(r2)
        edf = get_scalar(edof) if isinstance(edof, (list, tuple)) else get_scalar(edof)

        try:
            ll_full = gam_full.statistics_['loglikelihood']
            ll_k = gam_k.statistics_['loglikelihood']
            lr_stat = -2 * (ll_k - ll_full)
            edf_k = get_scalar(gam_k.statistics_['edof'])
            df_diff = edf - edf_k if not np.isnan(edf) and not np.isnan(edf_k) else 1
            p_joint = 1 - f.cdf(lr_stat, df_diff, len(df_avg) - df_diff) if df_diff > 0 else np.nan
        except Exception:
            p_joint = np.nan

        results.append({
            'Task': 'Task5_joint_with_k',
            'Response': response,
            'Predictor': 'GDD + Area + AvgDepth (joint, +k control)',
            'p_value': p_joint,
            'EDF': edf,
            'R_squared': r2_scalar,
            'AIC': aic,
            'n': len(df_avg)
        })
        print(f"  Joint p = {p_joint:.6f}, EDF = {edf:.3f}, R2 = {r2_scalar:.3f}, AIC = {aic:.2f}")

# ============================================================
# TASK 6a — QUANTILE BIN DISPERSION (Spearman trend)
# ============================================================
bin_data = []
trend_results = []
sensitivity_results = []

default_n_bins = max(3, min(6, n_total // 20))  # for n = 52 -> 3

print("\n" + "="*60)
print(f"TASK 6a: Quantile-bin dispersion (default n_bins={default_n_bins})")
print("="*60)

for driver in drivers_3:
    for response in responses_all:
        df_sorted = df_avg.sort_values(by=driver).reset_index(drop=True)

        # Default binning
        for n_bins in range(default_n_bins, 2, -1):
            try:
                df_sorted['bin'] = pd.qcut(df_sorted[driver], q=n_bins, labels=False, duplicates='drop')
                if df_sorted['bin'].nunique() >= 3:
                    break
            except ValueError:
                continue
        else:
            print(f"  {response} ~ {driver}: cannot create bins")
            continue

        bin_stats = []
        for b in sorted(df_sorted['bin'].unique()):
            subset = df_sorted[df_sorted['bin'] == b]
            vals = subset[response].values
            if len(vals) < 3:
                continue
            sd_val = np.std(vals, ddof=1)
            mean_val = np.mean(vals)
            ci_low, ci_high = bootstrap_sd_ci(vals)
            bin_stats.append({
                'Driver': driver, 'Response': response, 'bin_id': b,
                'driver_mid': subset[driver].mean(),
                'driver_min': subset[driver].min(), 'driver_max': subset[driver].max(),
                'sd': sd_val, 'log_sd': np.log(sd_val) if sd_val > 0 else np.nan,
                'cv': sd_val / abs(mean_val) if abs(mean_val) > 1e-9 else np.nan,
                'iqr': np.percentile(vals, 75) - np.percentile(vals, 25),
                'mean_resp': mean_val, 'n_bin': len(vals),
                'sd_ci_low': ci_low, 'sd_ci_high': ci_high
            })

        df_bin = pd.DataFrame(bin_stats).dropna()
        if len(df_bin) < 3:
            print(f"  {response} ~ {driver}: only {len(df_bin)} bins")
            continue

        # Spearman trend (primary)
        rho_sd, p_sd = spearmanr(df_bin['driver_mid'], df_bin['sd'])
        rho_iqr, p_iqr = spearmanr(df_bin['driver_mid'], df_bin['iqr'])
        rho_cv, p_cv = spearmanr(df_bin['driver_mid'], df_bin['cv'])

        trend_results.append({
            'Task': 'Task6a_Bin_Dispersion',
            'Response': response, 'Driver': driver,
            'n_bins': len(df_bin),
            'rho_sd': rho_sd, 'p_sd': p_sd,
            'rho_iqr': rho_iqr, 'p_iqr': p_iqr,
            'rho_cv': rho_cv, 'p_cv': p_cv,
            'note': 'Marginal; drivers correlated, see T6b'
        })
        print(f"  {response:12s} ~ {driver:12s} | bins={len(df_bin)} | "
              f"SD rho={rho_sd:+.2f} p={p_sd:.3f} | IQR rho={rho_iqr:+.2f} p={p_iqr:.3f}")

        bin_data.extend(bin_stats)

        # Sensitivity 3/4/5/6
        for n_test in [3, 4, 5, 6]:
            if n_test > n_total // 5:
                continue
            try:
                df_sorted['bin_test'] = pd.qcut(df_sorted[driver], q=n_test, labels=False, duplicates='drop')
                if df_sorted['bin_test'].nunique() < 3:
                    continue
                bs = []
                for b in sorted(df_sorted['bin_test'].unique()):
                    ss = df_sorted[df_sorted['bin_test'] == b]
                    vv = ss[response].values
                    if len(vv) < 3:
                        continue
                    bs.append({'mid': ss[driver].mean(), 'sd': np.std(vv, ddof=1)})
                df_bs = pd.DataFrame(bs).dropna()
                if len(df_bs) >= 3:
                    r, p = spearmanr(df_bs['mid'], df_bs['sd'])
                else:
                    r, p = np.nan, np.nan
            except Exception:
                r, p = np.nan, np.nan

            sensitivity_results.append({
                'Response': response, 'Driver': driver,
                'n_bins_test': n_test, 'rho_sd': r, 'p_sd': p
            })

df_bin_data = pd.DataFrame(bin_data)
df_trend = pd.DataFrame(trend_results)
df_sensitivity = pd.DataFrame(sensitivity_results)

# ============================================================
# TASK 6b — Heteroscedasticity (dAIC, no in-sample LRT)
# ============================================================
resid_results = []

print("\n" + "="*60)
print("TASK 6b: Heteroscedasticity — |residuals| ~ s(GDD)+s(Area)+s(AvgDepth)")
print("="*60)

for response in responses_all:
    print(f"\nProcessing: {response}")

    # Base model (same as Task 1)
    if response == 'k':
        preds_base = drivers_3
        terms_base = s(0) + s(1) + s(2)
    else:
        preds_base = drivers_3 + ['k']
        terms_base = s(0) + s(1) + s(2) + s(3)

    sub = df_avg[[response] + preds_base].dropna()
    X_base = sub[preds_base].values
    y_base = sub[response].values
    gam_base = LinearGAM(terms_base).fit(X_base, y_base)

    y_pred = gam_base.predict(X_base)
    abs_resid = np.abs(y_base - y_pred)

    # Full variance model
    X_env = sub[drivers_3].values
    gam_resid = LinearGAM(s(0) + s(1) + s(2)).fit(X_env, abs_resid)
    gam_resid_null = LinearGAM().fit(np.ones((len(sub), 1)), abs_resid)

    aic_full = get_scalar(gam_resid.statistics_.get('AIC', np.nan))
    aic_null = get_scalar(gam_resid_null.statistics_.get('AIC', np.nan))
    delta_aic = aic_null - aic_full if not np.isnan(aic_null) else np.nan

    p_vals = gam_resid.statistics_.get('p_values', None)
    p_ind = {}
    for i, d in enumerate(drivers_3):
        if isinstance(p_vals, (list, tuple)) and len(p_vals) > i:
            p_ind[d] = get_scalar(p_vals[i])
        else:
            p_ind[d] = get_scalar(p_vals)

    r2 = get_scalar(gam_resid.statistics_.get('pseudo_r2', np.nan))
    edf = get_scalar(gam_resid.statistics_.get('edof', np.nan))
    if isinstance(edf, (list, tuple)):
        edf = float(np.nansum(edf))

    if np.isnan(delta_aic):
        evidence = 'NA'
    elif delta_aic > 10:
        evidence = 'very strong'
    elif delta_aic > 6:
        evidence = 'strong'
    elif delta_aic > 2:
        evidence = 'moderate'
    else:
        evidence = 'weak/none'

    resid_results.append({
        'Task': 'Task6b_Heteroscedasticity',
        'Response': response,
        'p_GDD': p_ind.get('GDD', np.nan),
        'p_Area': p_ind.get('Area', np.nan),
        'p_AvgDepth': p_ind.get('AvgDepth', np.nan),
        'AIC_full': aic_full, 'AIC_null': aic_null, 'delta_AIC': delta_aic,
        'evidence': evidence,
        'EDF': edf, 'R_squared': r2, 'n': len(sub)
    })
    print(f"  GDD p={p_ind.get('GDD', np.nan):.3f}, Area p={p_ind.get('Area', np.nan):.3f}, "
          f"AvgDepth p={p_ind.get('AvgDepth', np.nan):.3f} | dAIC={delta_aic:+.1f} ({evidence})")

df_resid = pd.DataFrame(resid_results)

# ============================================================
# 4. SAVE RESULTS
# ============================================================
df_results = pd.DataFrame(results)
df_results['significant'] = (df_results['p_value'] < 0.05).map({True: 'TRUE', False: 'FALSE'})
column_order = ['Task', 'Response', 'Predictor', 'p_value', 'EDF', 'R_squared', 'AIC', 'n', 'significant']
df_results = df_results[column_order]
df_results.to_excel('GAM_Pike_Results_Main.xlsx', index=False)

with pd.ExcelWriter('GAM_Pike_Results_Full.xlsx') as writer:
    df_results.to_excel(writer, sheet_name='Tasks1-5', index=False)
    df_trend.to_excel(writer, sheet_name='Task6a_trends', index=False)
    df_bin_data.to_excel(writer, sheet_name='Task6a_bins', index=False)
    df_sensitivity.to_excel(writer, sheet_name='Task6a_sensitivity', index=False)
    df_resid.to_excel(writer, sheet_name='Task6b', index=False)

print("\n" + "="*60)
print("DONE. Saved: GAM_Pike_Results_Main.xlsx, GAM_Pike_Results_Full.xlsx")
print("="*60)