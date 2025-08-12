import pandas as pd
import numpy as np
import scipy.stats as stats
import statsmodels.stats.multitest as smm
import pulp
import gzip
import re
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path
from bioservices import KEGG
from mygene import MyGeneInfo
import gseapy as gp
import os
import matplotlib.pyplot as plt
import seaborn as sns

import osqp
from scipy import sparse

CANCER_LIST = ['BLCA', 'BRCA', 'CESC', 'COAD', 'ESCA', 'HNSC', 'KICH', 'KIRC', 'KIRP', 'LIHC', 'LUAD', 'LUSC', 'OV',
               'PAAD', 'READ', 'SKCM', 'STAD', 'THCA', 'UCEC']
THRESHOLD = 1.0
PENALTY = 10
SINK_LIST = [
    "v_g6p_to_nuc", "v_udp_to_sialic", "v_udp_to_glyco",
    "v_g3p_to_triacyl", "v_cit_to_lipid", "v_cer_to_sph",
    "v_cer_to_gang", "v_pyr_to_lac",
    "v_cit_to_tca", "v_serine_to_oneC", "v_ser_to_remain"
]
STAGE_LIST = ['StageI', 'StageII', 'StageIII', 'StageIV']
INTER_REACTIONS = ["v_glc_to_g6p", "v_g6p_to_f6p", "v_f6p_to_udp", "v_f6p_to_g3p", "v_g3p_to_3p", "v_3p_to_ser",
                   "v_ser_to_cer", "v_3p_to_pyr", "v_pyr_to_accoa", "v_accoa_to_cit"]
ALL_REACTIONS = SINK_LIST + INTER_REACTIONS + ["v_glc_in"]

KGML_DIR = Path("./data/kgml_pathways")
CACHE_DIR = Path("./cache")
CACHE_DIR.mkdir(exist_ok=True)
kegg = KEGG()

glucose_3downstream_pathway_genes_path = './biological-process-genes/glucose-downstream-pathway.csv'
glucose_3downstream_pathway_genes = pd.read_csv(glucose_3downstream_pathway_genes_path).drop_duplicates()

lactate_transporter_df = pd.read_csv(
    './Lactation-Data/lactate_transporter.csv',
    header=None)
lactate_efflux_genes = lactate_transporter_df[lactate_transporter_df[0] == 'efflux'].iloc[0, 1:].dropna().tolist()

lactylation_raw_df = pd.read_csv(
    './Lactation-Data/lactylation.csv')
lactylation_genes = []
for enzymes in lactylation_raw_df['enzyme']:
    lactylation_genes.extend(str(enzymes).split(', '))
lactylation_genes = list(set(g for g in lactylation_genes if g != 'nan'))


def _cache_write(txt: str, fp: Path) -> str:
    with gzip.open(fp, "wt", encoding='utf-8') as f: f.write(txt)
    return txt


def kegg_get(entry: str) -> str:
    fp = CACHE_DIR / f"{entry.replace(':', '_').replace('/', '_')}.txt.gz"
    if fp.exists():
        with gzip.open(fp, "rt", encoding='utf-8') as f: return f.read()
    txt = kegg.get(entry)
    return _cache_write(txt, fp) if isinstance(txt, str) else ""


def inverse_transform_log2(df):
    return (2 ** df) - 0.001


def calculate_sig_up(filtered_tumor_data, filtered_normal_data):
    common_genes = filtered_tumor_data.index.intersection(filtered_normal_data.index)
    if common_genes.empty: return pd.DataFrame(columns=['Gene']), pd.DataFrame()
    filtered_tumor_data = filtered_tumor_data.loc[common_genes]
    filtered_normal_data = filtered_normal_data.loc[common_genes]
    p_values, fc_ = [], []
    for gene in common_genes:
        tumor_values = filtered_tumor_data.loc[gene].dropna().values
        normal_values = filtered_normal_data.loc[gene].dropna().values
        if len(normal_values) > 1 and np.std(normal_values) > 1e-6:
            _, p_val = stats.ttest_ind(tumor_values, normal_values, equal_var=False, nan_policy='omit')
            fc_value = tumor_values.mean() / normal_values.mean() if normal_values.mean() > 1e-6 else 9999
        else:
            p_val, fc_value = 1.0, 1.0
        p_values.append(p_val);
        fc_.append(fc_value)
    if not p_values: return pd.DataFrame(columns=['Gene']), pd.DataFrame()
    _, p_values_corrected, _, _ = smm.multipletests(p_values, alpha=0.05, method='fdr_bh')
    diff_results = pd.DataFrame({'Gene': common_genes, 'Adjusted P-value': p_values_corrected, 'Fold Change': fc_})
    sig_genes = diff_results[diff_results['Adjusted P-value'] < 0.05]
    return sig_genes[sig_genes['Fold Change'] >= THRESHOLD], diff_results


def calculate_h_plus_factors() -> dict:
    print("Calculating H+ production factors for all pathways...")
    H_PLUS_NON_IGNORED_MET = re.compile(r"^(H2O|PI|PPI|CO2)$", re.I)
    SEP = re.compile(r"\s*(?:<=>|=>|->|=)\s*")
    reactions, r2hsa = {}, defaultdict(set)
    for xml_f in KGML_DIR.glob("hsa*.xml"):
        hsa_code = re.search(r"hsa(\d+)", xml_f.name).group(1)
        root = ET.parse(xml_f).getroot()
        for entry in root.findall("entry"):
            if entry.get("type") == "gene":
                for rxn in entry.get("reaction", "").split():
                    r2hsa[rxn.replace("rn:", "")].add(hsa_code)
    for rid in r2hsa.keys():
        txt = kegg_get(f"rn:{rid}")
        m = re.search(r"^EQUATION\s+(.+)", txt, flags=re.M)
        if not m: continue
        lhs, rhs = SEP.split(m.group(1), 1)
        sto = defaultdict(float)

        def add_side(seg, sgn):
            for term in seg.split(" + "):
                sp = term.strip().split(" ", 1)
                coef, met = (float(sp[0]) if sp[0].replace('.', '', 1).isdigit() else 1.0, sp[1]) if len(sp) == 2 else (
                1.0, sp[0])
                if not H_PLUS_NON_IGNORED_MET.match(met):
                    sto[met] += sgn * coef

        add_side(lhs, -1);
        add_side(rhs, 1)
        reactions[rid] = dict(equation=dict(sto))
    hsa_map = {
        "v_g6p_to_nuc": ["00230", "00240"], "v_udp_to_sialic": ["00520"], "v_udp_to_glyco": ["00510", "00512"],
        "v_g3p_to_triacyl": ["00561"], "v_serine_to_oneC": ["00670"], "v_cer_to_sph": ["00600"],
        "v_cer_to_gang": ["00604"], "v_cit_to_lipid": ["01040"], "v_pyr_to_lac": ["00620"],
        "v_cit_to_tca": ["00020"], "v_ser_to_remain": ["00260"],
    }
    pathway_h_factors = {}
    for pathway_name, hsa_codes in hsa_map.items():
        pathway_rids = {rid for rid, codes in r2hsa.items() if not set(codes).isdisjoint(hsa_codes)}
        h_coeffs = [reactions[rid]['equation'].get('C00080', 0.0) for rid in pathway_rids if rid in reactions]
        pathway_h_factors[pathway_name] = np.mean([h for h in h_coeffs if h != 0]) if any(
            h != 0 for h in h_coeffs) else 0.0
    pathway_h_factors['v_pyr_to_lac'] = 1.0
    print("H+ production factors calculated.")
    return pathway_h_factors


def get_ssgsea_score(pathway_genes: list, expression_df: pd.DataFrame) -> float:
    if 'Gene' in expression_df.columns:
        expression_df = expression_df.set_index('Gene')
    valid_genes = [g for g in pathway_genes if g in expression_df.index]
    if not valid_genes: return 1e-9
    gene_sets = {"pathway": valid_genes}
    try:
        ss = gp.ssgsea(data=expression_df, gene_sets=gene_sets, sample_norm_method='rank', outdir=None, verbose=False,
                       min_size=1)
        raw_score = ss.res2d["NES"].mean()
        return np.exp(raw_score)
    except Exception as e:
        print(f"Warning: ssGSEA failed. Error: {e}")
        return 1e-9


def run_fba_with_qp(ub_dict, w_dict, lambda_penalty=0.1):
    rxn_map = {name: i for i, name in enumerate(ALL_REACTIONS)}
    n_vars = len(ALL_REACTIONS)
    P_diag = np.zeros(n_vars)
    for sink in SINK_LIST: P_diag[rxn_map[sink]] = 2 * lambda_penalty
    P = sparse.diags(P_diag, format='csc')
    q = np.zeros(n_vars)
    for sink in SINK_LIST: q[rxn_map[sink]] = -w_dict.get(sink, 0)
    s_matrix = np.zeros((11, n_vars))
    met_map = {"G6P": 0, "F6P": 1, "UDPGlcNAc": 2, "G3P": 3, "3P": 4, "Ser": 5, "Ceramide": 6, "Pyruvate": 7,
               "AcCoA": 8, "Citrate": 9}
    stoich = {"MB_G6P": {"v_glc_to_g6p": 1, "v_g6p_to_nuc": -1, "v_g6p_to_f6p": -1},
              "MB_F6P": {"v_g6p_to_f6p": 1, "v_f6p_to_udp": -1, "v_f6p_to_g3p": -1},
              "MB_UDPGlcNAc": {"v_f6p_to_udp": 1, "v_udp_to_sialic": -1, "v_udp_to_glyco": -1},
              "MB_G3P": {"v_f6p_to_g3p": 1, "v_g3p_to_triacyl": -1, "v_g3p_to_3p": -1},
              "MB_3P": {"v_g3p_to_3p": 1, "v_3p_to_ser": -1, "v_3p_to_pyr": -1},
              "MB_Ser": {"v_3p_to_ser": 1, "v_ser_to_remain": -1, "v_ser_to_cer": -1, "v_serine_to_oneC": -1},
              "MB_Ceramide": {"v_ser_to_cer": 1, "v_cer_to_sph": -1, "v_cer_to_gang": -1},
              "MB_Pyruvate": {"v_3p_to_pyr": 1, "v_pyr_to_lac": -1, "v_pyr_to_accoa": -1},
              "MB_AcCoA": {"v_pyr_to_accoa": 1, "v_accoa_to_cit": -1},
              "MB_Citrate": {"v_accoa_to_cit": 1, "v_cit_to_tca": -1, "v_cit_to_lipid": -1}}
    for met, row_idx in met_map.items():
        for rxn, val in stoich[f"MB_{met}"].items():
            if rxn in rxn_map: s_matrix[row_idx, rxn_map[rxn]] = val
    s_matrix[10, rxn_map['v_glc_in']] = 1;
    s_matrix[10, rxn_map['v_glc_to_g6p']] = -1
    A = sparse.csc_matrix(s_matrix)
    l_constraints = np.zeros(A.shape[0]);
    u_constraints = np.zeros(A.shape[0])
    l_bounds = np.zeros(n_vars);
    u_bounds = np.full(n_vars, 1000.0)
    for rxn_name, i in rxn_map.items():
        u_bounds[i] = ub_dict.get(rxn_name, 1000.0)
        l_bounds[i] = 0
    A_full = sparse.vstack([A, sparse.identity(n_vars, format='csc')]).tocsc()
    l_full = np.concatenate([l_constraints, l_bounds]);
    u_full = np.concatenate([u_constraints, u_bounds])
    prob = osqp.OSQP();
    prob.setup(P, q, A_full, l_full, u_full, verbose=False, polish=True)
    results = prob.solve()
    if results.info.status == 'solved':
        tmp = {name: np.maximum(results.x[i], 0) for name, i in rxn_map.items()}
        tmp['v_ser_to_remain'] = tmp['v_3p_to_ser']
        return tmp
    else:
        print(f"Warning: QP Solver failed with status: {results.info.status}");
        return {name: 0 for name in ALL_REACTIONS}


def run_lactate_fba_qp_optimized(total_lactate_flux, weights, lambda_penalty=0.1):
    """
    Partitions lactate flux by solving a true quadratic optimization problem.
    This approach finds a balanced distribution based on weights, rather than
    enforcing a strict algebraic proportionality.
    """

    P = sparse.diags([2 * lambda_penalty, 2 * lambda_penalty], format='csc')

    w_export = weights.get("v_lac_to_transport", 1e-9)
    w_lactylation = weights.get("v_lac_to_lactylation", 1e-9)
    q = np.array([-w_export, -w_lactylation])

    A_constraints = sparse.csc_matrix([[1, 1]])
    l_constraints = np.array([total_lactate_flux])
    u_constraints = np.array([total_lactate_flux])

    A_bounds = sparse.identity(2, format='csc')
    l_bounds = np.zeros(2)
    u_bounds = np.full(2, np.inf)

    A_full = sparse.vstack([A_constraints, A_bounds], format='csc')
    l_full = np.concatenate([l_constraints, l_bounds])
    u_full = np.concatenate([u_constraints, u_bounds])

    prob = osqp.OSQP()
    prob.setup(P, q, A_full, l_full, u_full, verbose=False, polish=True)
    results = prob.solve()

    if results.info.status != 'solved':
        print(f"Warning: Lactate FBA QP Solver failed with status: {results.info.status}. Defaulting to 50/50 split.")
        return {"transport_prop": 50.0, "lactylation_prop": 50.0}

    v_export, v_lactylation = np.maximum(results.x, 0)
    total_out_flux = v_export + v_lactylation

    if total_out_flux < 1e-9:
        return {"transport_prop": 0, "lactylation_prop": 0}

    return {
        "transport_prop": (v_export / total_out_flux) * 100,
        "lactylation_prop": (v_lactylation / total_out_flux) * 100
    }


def get_de_based_ssgsea_score(all_pathway_genes, tumor_expr, normal_expr):
    """
    Calculates ssGSEA score, using only significantly upregulated genes if available,
    otherwise falls back to all genes in the pathway.
    """
    filtered_tumor = tumor_expr.loc[tumor_expr.index.isin(all_pathway_genes)]
    filtered_normal = normal_expr.loc[normal_expr.index.isin(all_pathway_genes)]

    if filtered_tumor.empty: return 1e-9

    up_sig_genes_df, _ = calculate_sig_up(filtered_tumor, filtered_normal)

    if not up_sig_genes_df.empty:
        ssgsea_gene_set = up_sig_genes_df['Gene'].tolist()
    else:
        ssgsea_gene_set = all_pathway_genes

    return get_ssgsea_score(ssgsea_gene_set, tumor_expr)


h_plus_factors = calculate_h_plus_factors()

for cancer_name in CANCER_LIST:
    print(f"\n--- Processing {cancer_name} ---")
    estimation_Kcat_path = f'./output/{cancer_name}_pathway_kcat_summary_1.5.csv'
    estimation_Kcat = pd.read_csv(estimation_Kcat_path)
    pathway_rename_map = {
        'Nucleotide synthesis': 'v_g6p_to_nuc', 'Sialic acid': 'v_udp_to_sialic', 'Glycosylation': 'v_udp_to_glyco',
        'Triglyceride': 'v_g3p_to_triacyl', 'Unsaturated fatty acid': 'v_cit_to_lipid', 'Sphingolipid': 'v_cer_to_sph',
        'Ganglioside': 'v_cer_to_gang', 'Lactate': 'v_pyr_to_lac', 'TCA': 'v_cit_to_tca',
        'One-carbon': 'v_serine_to_oneC', 'Serine remain': 'v_ser_to_remain'
    }
    estimation_Kcat['pathway'] = estimation_Kcat['pathway'].replace(pathway_rename_map)
    mean_kinetics_param = estimation_Kcat.set_index('pathway')['kcat_coeff'].to_dict()

    cancer_tpm = pd.read_csv(f'../../data/tpm-clinic/{cancer_name}.Primary.TCGA-GTEX.GENCODE-v23.log2.tpm.csv')
    cancer_clinical = pd.read_csv(f'../../data/tpm-clinic/{cancer_name}.Primary.TCGA-GTEX.GENCODE-v23.Clinical.csv')
    normal_samples = cancer_clinical[cancer_clinical['tissue'].str.lower() == 'normal']['samples'].tolist()
    common_samples = list(set(cancer_tpm.columns).intersection(set(normal_samples)))
    normal_data_log = cancer_tpm[['Gene'] + common_samples]

    Flux_res, H_plus_res = defaultdict(list), defaultdict(list)
    Lactate_Fate_Res = defaultdict(list)
    stage3_df = None

    for per_stage in STAGE_LIST:
        tumor_stage_tpm_path = f'../../data/stage-data/TCGA-{cancer_name}/{per_stage}/TCGA-{cancer_name}-{per_stage}-TPM.csv'
        try:
            tumor_stage_tpm = pd.read_csv(tumor_stage_tpm_path)
        except FileNotFoundError:
            print(f"Warning: Stage file not found for {cancer_name} - {per_stage}. Skipping.")
            continue

        if per_stage == 'StageIII':
            stage3_df = tumor_stage_tpm.copy()
            continue

        current_stage_name = per_stage
        if per_stage == 'StageIV' and stage3_df is not None:
            tumor_stage_tpm = pd.merge(tumor_stage_tpm, stage3_df, on="Gene", how="outer").fillna(0)
            current_stage_name = "StageIII-IV"

        print(f"  Processing {current_stage_name}...")

        tumor_tpm_indexed = tumor_stage_tpm.set_index('Gene')
        up_bound_pathway = {}

        for group_name, group_df in glucose_3downstream_pathway_genes.groupby('gene_type'):
            cur_genes = list(set(group_df['gene'].str.strip().tolist()))
            pathway_activity_score = get_ssgsea_score(cur_genes, tumor_tpm_indexed)
            up_bound_pathway[group_name] = pathway_activity_score * mean_kinetics_param.get(group_name, 0)

        sum_ub = sum(up_bound_pathway.values())
        norm_ub = {p: (v / sum_ub) for p, v in up_bound_pathway.items()} if sum_ub > 0 else {}
        norm_ub['v_glc_in'] = sum(norm_ub.values())

        dampened_caps = {p: np.sqrt(v) for p, v in up_bound_pathway.items()}
        sum_dampened_caps = sum(dampened_caps.values())
        weights = {p: v / sum_dampened_caps for p, v in dampened_caps.items()} if sum_dampened_caps > 0 else {}

        solution = run_fba_with_qp(norm_ub, weights, lambda_penalty=PENALTY)
        total_flux = sum(solution.get(s, 0) for s in SINK_LIST)
        print(f"    Total Flux for {current_stage_name} = {total_flux:.4g}")

        Flux_res['Stage'].append(current_stage_name)
        H_plus_res['Stage'].append(current_stage_name)

        if total_flux > 1e-9:

            non_lipid_sinks = [s for s in SINK_LIST if s not in ['v_cit_to_lipid', 'v_cer_to_sph', 'v_g3p_to_triacyl']]
            for sink in SINK_LIST:
                pathway_name = sink.split('_')[-1]
                Flux_res[f'Proportion_{pathway_name}_(%)'].append((solution.get(sink, 0) / total_flux) * 100)

            h_prods = {p: solution.get(p, 0) * h_plus_factors.get(p, 0) for p in SINK_LIST}
            h_lipid = sum(h_prods.get(s, 0) for s in ['v_cit_to_lipid', 'v_cer_to_sph', 'v_g3p_to_triacyl'])
            h_total = sum(h_prods.values())
            H_plus_res['H_lipid'].append(h_lipid)
            H_plus_res['H_Total'].append(h_total)
            for sink in non_lipid_sinks:
                H_plus_res[f'H_{sink.split("_")[-1]}'].append(h_prods.get(sink, 0))

            lactate_production_flux = solution.get('v_pyr_to_lac', 0)
            if lactate_production_flux > 1e-9:
                normal_data_indexed = normal_data_log.set_index('Gene')

                transport_score = get_de_based_ssgsea_score(lactate_efflux_genes, tumor_tpm_indexed,
                                                            normal_data_indexed)
                lactylation_score = get_de_based_ssgsea_score(lactylation_genes, tumor_tpm_indexed, normal_data_indexed)

                total_score = transport_score + lactylation_score
                lactate_weights = {
                    "v_lac_to_transport": transport_score / total_score if total_score > 0 else 0.5,
                    "v_lac_to_lactylation": lactylation_score / total_score if total_score > 0 else 0.5
                }

                lactate_fates = run_lactate_fba_qp_optimized(lactate_production_flux, lactate_weights,
                                                             lambda_penalty=PENALTY)
                Lactate_Fate_Res['Stage'].append(current_stage_name)
                Lactate_Fate_Res['Transport (%)'].append(lactate_fates['transport_prop'])
                Lactate_Fate_Res['Lactylation (%)'].append(lactate_fates['lactylation_prop'])
            else:
                Lactate_Fate_Res['Stage'].append(current_stage_name)
                Lactate_Fate_Res['Transport (%)'].append(0)
                Lactate_Fate_Res['Lactylation (%)'].append(0)

        else:
            for key_list in [Flux_res, H_plus_res, Lactate_Fate_Res]:
                for key in key_list:
                    if key != 'Stage': key_list[key].append(np.nan)

    for res_dict in [Flux_res, H_plus_res, Lactate_Fate_Res]:
        max_len = len(res_dict.get('Stage', []))
        for key, val_list in res_dict.items():
            if len(val_list) != max_len:
                res_dict[key] = val_list + [np.nan] * (max_len - len(val_list))

    Flux_res_df = pd.DataFrame(dict(Flux_res))
    Flux_res_df.to_csv(f'./Res/FBA_res_{cancer_name}-stage_flux_proportions-QP-PENALTY-{PENALTY}.csv', index=False)

    H_plus_res_df = pd.DataFrame(dict(H_plus_res))
    H_plus_res_df.to_csv(f'./Res/H_plus_production_{cancer_name}-stage-QP-PENALTY-{PENALTY}.csv', index=False)

    if Lactate_Fate_Res['Stage']:
        lactate_df = pd.DataFrame(dict(Lactate_Fate_Res))
        output_dir = './lactate_fate_analysis'
        os.makedirs(output_dir, exist_ok=True)
        output_csv_path = os.path.join(output_dir, f'{cancer_name}_lactate_fates-PENALTY-{PENALTY}.csv')
        lactate_df.to_csv(output_csv_path, index=False)
        print(f"  Saved lactate fate results to {output_csv_path}")

        plot_df = lactate_df.set_index('Stage')
        if not plot_df.empty:
            sns.set_style("whitegrid")
            fig, ax = plt.subplots(figsize=(10, 7))
            plot_df.plot(kind='bar', stacked=True, ax=ax, width=0.7)
            ax.set_title(f'Predicted Metabolic Fates of Lactate in {cancer_name}', fontsize=16, weight='bold')
            ax.set_ylabel('Proportional Flux Distribution (%)', fontsize=12)
            ax.set_xlabel('Cancer Stage', fontsize=12)
            ax.tick_params(axis='x', rotation=0)
            ax.set_ylim(0, 100)
            ax.legend(title='Lactate Fate', bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            output_fig_path = os.path.join(output_dir, f'{cancer_name}_lactate_fates_plot-PENALTY-{PENALTY}.png')
            plt.savefig(output_fig_path, dpi=300)
            plt.show()
            plt.close(fig)
            print(f"  Saved lactate fate visualization to {output_fig_path}")

    print(f"  Saved all results for {cancer_name}")