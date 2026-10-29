#!/usr/bin/env python3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scanpy as sc
from sklearn.metrics import roc_curve, precision_recall_curve, auc
from scipy.stats import sem


def get_gene_order_from_adata(adata_file):
    """Efficiently get gene order from AnnData file."""
    print(f"Loading gene names from {adata_file}...")
    adata = sc.read_h5ad(adata_file, backed="r")  # Memory-efficient reading
    gene_order = adata.var.index.tolist()
    adata.file.close()  # Close the file
    return gene_order


def load_and_compute_curves(predicted_effects_file, rank_genes_file, gene_order, control_pert="non-targeting"):
    """Load data and compute ROC/PR curves with correct gene alignment."""

    print(f"Processing {predicted_effects_file}...")

    # Load predicted effects
    pred_effects = pd.read_csv(predicted_effects_file, index_col=0)

    # Load ground truth statistical results
    rank_data = np.load(rank_genes_file, allow_pickle=True)

    # Get perturbation names
    names_df = pd.DataFrame(rank_data["names"])
    perturbations = names_df.columns.tolist()

    print(f"Found {len(perturbations)} perturbations, {len(gene_order)} genes")

    roc_curves = []
    pr_curves = []
    roc_aucs = []
    pr_aucs = []
    processed_count = 0

    for pert in perturbations:
        if pert == control_pert or pert not in pred_effects.index:
            continue

        # Get predicted scores (in original gene order)
        pred_scores = pred_effects.loc[pert].values

        # Get ground truth results for this perturbation
        pert_col_idx = names_df.columns.get_loc(pert)
        pert_names = names_df.iloc[:, pert_col_idx].values  # Gene names ordered by significance
        pert_pvals = pd.DataFrame(rank_data["pvals_adj"]).iloc[:, pert_col_idx].values  # P-values in same order

        # Create a mapping from gene name to p-value
        gene_to_pval = dict(zip(pert_names, pert_pvals))

        # Create p-values in the same order as predicted scores (original gene order)
        aligned_pvals = []
        aligned_pred_scores = []

        for i, gene in enumerate(gene_order):
            if gene in gene_to_pval:
                aligned_pvals.append(gene_to_pval[gene])
                aligned_pred_scores.append(pred_scores[i])
            else:
                # If gene not in statistical test results, assign p-value of 1.0 (not significant)
                aligned_pvals.append(1.0)
                aligned_pred_scores.append(pred_scores[i])

        aligned_pvals = np.array(aligned_pvals)
        aligned_pred_scores = np.array(aligned_pred_scores)

        # Create binary labels
        true_labels = (aligned_pvals < 0.05).astype(int)

        # Skip if all labels are the same
        if len(np.unique(true_labels)) < 2:
            continue

        # Compute ROC curve
        fpr, tpr, _ = roc_curve(true_labels, aligned_pred_scores)
        roc_auc = auc(fpr, tpr)
        roc_curves.append((fpr, tpr))
        roc_aucs.append(roc_auc)

        # Compute PR curve
        precision, recall, _ = precision_recall_curve(true_labels, aligned_pred_scores)
        pr_auc = auc(recall, precision)
        pr_curves.append((precision, recall))
        pr_aucs.append(pr_auc)

        processed_count += 1

    print(f"Processed {processed_count} perturbations")

    if not roc_curves:
        return None

    # Average curves using interpolation
    # ROC curve averaging
    mean_fpr = np.linspace(0, 1, 100)
    tpr_curves = []
    for fpr, tpr in roc_curves:
        tpr_curves.append(np.interp(mean_fpr, fpr, tpr))
    tpr_curves = np.array(tpr_curves)
    mean_tpr = np.mean(tpr_curves, axis=0)
    stderr_tpr = sem(tpr_curves, axis=0)

    # PR curve averaging
    mean_recall = np.linspace(0, 1, 100)
    precision_curves = []
    for precision, recall in pr_curves:
        # Reverse for interpolation (recall should be increasing)
        precision_curves.append(np.interp(mean_recall, recall[::-1], precision[::-1]))
    precision_curves = np.array(precision_curves)
    mean_precision = np.mean(precision_curves, axis=0)
    stderr_precision = sem(precision_curves, axis=0)

    return (
        mean_fpr,
        mean_tpr,
        stderr_tpr,
        mean_recall,
        mean_precision,
        stderr_precision,
        np.mean(roc_aucs),
        sem(roc_aucs),
        np.mean(pr_aucs),
        sem(pr_aucs),
    )


def main():
    print("=== BASECOUNT vs CELLXGENE DE VALIDATION CURVES ===")

    # Get gene order from one of the reconstructed files (they should be identical)
    gene_order = get_gene_order_from_adata("basecount_step=75000_reconstructed.h5ad")

    # Load data for both models with corrected alignment
    print("\n1. Processing basecount model...")
    basecount_results = load_and_compute_curves(
        "basecount_step=75000_predicted_effects.csv", "basecount_step=75000_rank_genes_groups.npz", gene_order
    )

    print("\n2. Processing cellxgene model...")
    cellxgene_results = load_and_compute_curves(
        "cellxgene_step=75000_predicted_effects.csv", "cellxgene_step=75000_rank_genes_groups.npz", gene_order
    )

    if basecount_results is None or cellxgene_results is None:
        print("Error: Could not compute curves for one or both models")
        return

    # Unpack results
    (
        bc_fpr,
        bc_tpr,
        bc_tpr_err,
        bc_recall,
        bc_precision,
        bc_precision_err,
        bc_roc_auc,
        bc_roc_err,
        bc_pr_auc,
        bc_pr_err,
    ) = basecount_results

    (
        cxg_fpr,
        cxg_tpr,
        cxg_tpr_err,
        cxg_recall,
        cxg_precision,
        cxg_precision_err,
        cxg_roc_auc,
        cxg_roc_err,
        cxg_pr_auc,
        cxg_pr_err,
    ) = cellxgene_results

    # Create the plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # ROC Curve (left subplot)
    ax1.plot(bc_fpr, bc_tpr, "b-", linewidth=2.5, label=f"Basecount (AUC={bc_roc_auc:.3f}±{bc_roc_err:.3f})")
    ax1.fill_between(bc_fpr, bc_tpr - bc_tpr_err, bc_tpr + bc_tpr_err, alpha=0.2, color="blue")

    ax1.plot(cxg_fpr, cxg_tpr, "r-", linewidth=2.5, label=f"CellxGene (AUC={cxg_roc_auc:.3f}±{cxg_roc_err:.3f})")
    ax1.fill_between(cxg_fpr, cxg_tpr - cxg_tpr_err, cxg_tpr + cxg_tpr_err, alpha=0.2, color="red")

    ax1.plot([0, 1], [0, 1], "k--", alpha=0.5, linewidth=1, label="Random")
    ax1.set_xlabel("False Positive Rate", fontsize=12)
    ax1.set_ylabel("True Positive Rate", fontsize=12)
    ax1.set_title("ROC Curves", fontsize=14, fontweight="bold")
    ax1.legend(fontsize=10, loc="lower right")
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1])

    # PR Curve (right subplot)
    ax2.plot(bc_recall, bc_precision, "b-", linewidth=2.5, label=f"Basecount (AUC={bc_pr_auc:.3f}±{bc_pr_err:.3f})")
    ax2.fill_between(
        bc_recall, bc_precision - bc_precision_err, bc_precision + bc_precision_err, alpha=0.2, color="blue"
    )

    ax2.plot(cxg_recall, cxg_precision, "r-", linewidth=2.5, label=f"CellxGene (AUC={cxg_pr_auc:.3f}±{cxg_pr_err:.3f})")
    ax2.fill_between(
        cxg_recall, cxg_precision - cxg_precision_err, cxg_precision + cxg_precision_err, alpha=0.2, color="red"
    )

    ax2.set_xlabel("Recall", fontsize=12)
    ax2.set_ylabel("Precision", fontsize=12)
    ax2.set_title("Precision-Recall Curves", fontsize=14, fontweight="bold")
    ax2.legend(fontsize=10, loc="upper right")
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([0, 1])
    ax2.set_ylim([0, 1])

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig("basecount_vs_cellxgene_val_de.svg", dpi=300, bbox_inches="tight")
    print(f"\n✅ Plot saved as: basecount_vs_cellxgene_val_de.svg")

    # Print summary
    print(f"\n=== SUMMARY ===")
    print(f"Basecount - ROC AUC: {bc_roc_auc:.4f} ± {bc_roc_err:.4f}")
    print(f"Basecount - PR AUC:  {bc_pr_auc:.4f} ± {bc_pr_err:.4f}")
    print(f"CellxGene - ROC AUC: {cxg_roc_auc:.4f} ± {cxg_roc_err:.4f}")
    print(f"CellxGene - PR AUC:  {cxg_pr_auc:.4f} ± {cxg_pr_err:.4f}")

    # Show relative improvement
    roc_improvement = (bc_roc_auc - cxg_roc_auc) / cxg_roc_auc * 100
    pr_improvement = (bc_pr_auc - cxg_pr_auc) / cxg_pr_auc * 100

    print(f"\nBasecount improvement over CellxGene:")
    print(f"ROC AUC: +{roc_improvement:.1f}%")
    print(f"PR AUC:  +{pr_improvement:.1f}%")


if __name__ == "__main__":
    main()
