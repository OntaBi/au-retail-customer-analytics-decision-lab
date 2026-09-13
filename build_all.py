from __future__ import annotations

import argparse
import subprocess
import sys
import time
from dataclasses import dataclass


# ---------------------------------------------------------------------
# Pipeline configuration
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class PipelineStep:
    name: str
    module: str


BUILD_STEPS = [
    # Synthetic source layer
    PipelineStep(
        "Generate synthetic customer master",
        "src.data_generation.generate_customer_master",
    ),
    PipelineStep(
        "Generate synthetic customer identity records",
        "src.data_generation.generate_identity_records",
    ),
    PipelineStep(
        "Generate synthetic customer transactions",
        "src.data_generation.generate_transactions",
    ),

    # Identity and golden-customer foundation
    PipelineStep(
        "Resolve customer identities",
        "src.customer.identity_resolution",
    ),
    PipelineStep(
        "Build resolved golden customer layer",
        "src.customer.golden_customer_layer",
    ),
    PipelineStep(
        "Generate synthetic campaign exposures",
        "src.data_generation.generate_campaign_exposures",
    ),

    # Behaviour, value and segmentation
    PipelineStep(
        "Build customer cadence and lifecycle features",
        "src.customer.customer_features",
    ),
    PipelineStep(
        "Build customer momentum features",
        "src.customer.customer_momentum",
    ),
    PipelineStep(
        "Build longitudinal cadence trends",
        "src.customer.cadence_trend",
    ),
    PipelineStep(
        "Build customer value and RFM features",
        "src.customer.customer_value",
    ),
    PipelineStep(
        "Build clustering feature matrix",
        "src.customer.clustering_features",
    ),
    PipelineStep(
        "Fit behavioural customer clusters",
        "src.modelling.customer_clustering",
    ),
    PipelineStep(
        "Build customer priority decisions",
        "src.decision_engine.customer_priority",
    ),
    PipelineStep(
        "Build customer growth, cohort and LTV analytics",
        "src.customer.customer_growth_value",
    ),

    # Governed propensity layer
    PipelineStep(
        "Build re-engagement propensity training snapshots",
        "src.modelling.propensity_features",
    ),
    PipelineStep(
        "Train and govern re-engagement propensity model",
        "src.modelling.customer_propensity",
    ),
    PipelineStep(
        "Build cross-sell propensity training snapshots",
        "src.modelling.cross_sell_features",
    ),
    PipelineStep(
        "Train and govern cross-sell propensity model",
        "src.modelling.cross_sell_propensity",
    ),
    PipelineStep(
        "Build cross-sell category recommendations",
        "src.modelling.cross_sell_category_recommendation",
    ),
    PipelineStep(
        "Build promotion-response propensity training data",
        "src.modelling.promotion_response_training",
    ),
    PipelineStep(
        "Train and govern promotion-response propensity model",
        "src.modelling.promotion_response_propensity",
    ),
    PipelineStep(
        "Score current promotion-response opportunities",
        "src.modelling.promotion_response_operational",
    ),

    # Governed decisioning
    PipelineStep(
        "Build Next Best Action recommendations",
        "src.decision_engine.next_best_action",
    ),
]


QA_STEPS = [
    PipelineStep(
        "Validate identity resolution",
        "src.qa.validate_identity_resolution",
    ),
    PipelineStep(
        "Validate lapse model",
        "src.qa.validate_lapse_model",
    ),
    PipelineStep(
        "Validate momentum model",
        "src.qa.validate_momentum_model",
    ),
    PipelineStep(
        "Validate cadence trend model",
        "src.qa.validate_cadence_trend",
    ),
    PipelineStep(
        "Validate customer clusters",
        "src.qa.validate_customer_clusters",
    ),
]


# ---------------------------------------------------------------------
# Data-generation modules
# ---------------------------------------------------------------------

DATA_GENERATION_MODULES = {
    "src.data_generation.generate_customer_master",
    "src.data_generation.generate_transactions",
    "src.data_generation.generate_identity_records",
    "src.data_generation.generate_campaign_exposures",
}


# ---------------------------------------------------------------------
# Step runner
# ---------------------------------------------------------------------

def run_step(
    step: PipelineStep,
    step_number: int,
    total_steps: int,
) -> None:

    print("\n" + "=" * 78)
    print(
        f"[{step_number}/{total_steps}] "
        f"{step.name}"
    )
    print("=" * 78)

    print(
        f"Running: {sys.executable} -m {step.module}"
    )

    started = time.perf_counter()

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            step.module,
        ],
        check=False,
    )

    elapsed = (
        time.perf_counter()
        - started
    )

    if completed.returncode != 0:
        print("\nPIPELINE FAILED")
        print("-" * 78)

        print(
            f"Step: {step.name}"
        )

        print(
            f"Module: {step.module}"
        )

        print(
            f"Exit code: "
            f"{completed.returncode}"
        )

        raise SystemExit(
            completed.returncode
        )

    print(
        f"\nCompleted in "
        f"{elapsed:.1f} seconds."
    )


# ---------------------------------------------------------------------
# CLI arguments
# ---------------------------------------------------------------------

def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "Build the AU Retail Customer Analytics "
            "Decision Lab analytical pipeline."
        )
    )

    parser.add_argument(
        "--with-qa",
        action="store_true",
        help=(
            "Run validation modules after "
            "building analytical outputs."
        ),
    )

    parser.add_argument(
        "--skip-data-generation",
        action="store_true",
        help=(
            "Reuse existing generated customer, transaction, identity "
            "and campaign-exposure source data."
        ),
    )

    return parser.parse_args()


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:

    args = parse_args()

    steps = BUILD_STEPS.copy()

    if args.skip_data_generation:
        steps = [
            step
            for step in steps
            if step.module
            not in DATA_GENERATION_MODULES
        ]

    if args.with_qa:
        steps.extend(
            QA_STEPS
        )

    print(
        "\nAU RETAIL CUSTOMER ANALYTICS DECISION LAB"
    )
    print("=" * 78)

    print(
        f"Python executable: "
        f"{sys.executable}"
    )

    print(
        f"Pipeline steps: "
        f"{len(steps)}"
    )

    print(
        "Synthetic data generation: "
        + (
            "SKIPPED"
            if args.skip_data_generation
            else "INCLUDED"
        )
    )

    print(
        "QA validation: "
        + (
            "INCLUDED"
            if args.with_qa
            else "SKIPPED"
        )
    )

    pipeline_started = (
        time.perf_counter()
    )

    for index, step in enumerate(
        steps,
        start=1,
    ):
        run_step(
            step=step,
            step_number=index,
            total_steps=len(steps),
        )

    elapsed = (
        time.perf_counter()
        - pipeline_started
    )

    print("\n" + "=" * 78)
    print("PIPELINE COMPLETE")
    print("=" * 78)

    print(
        f"Completed {len(steps)} steps "
        f"in {elapsed:.1f} seconds."
    )

    print(
        "\nCore generated and runtime outputs:"
    )

    outputs = [
        # Core synthetic source data
        "data/generated/customer_master.parquet",
        "data/generated/transactions.parquet",
        "data/generated/transaction_ground_truth.parquet",

        # Identity Resolution
        "data/generated/customer_identity_records.parquet",
        "data/generated/identity_ground_truth.parquet",
        "data/runtime/customer_identity_resolution.parquet",
        "data/runtime/identity_resolution_summary.parquet",
        "data/runtime/identity_edge_audit.parquet",

        # Resolved golden customer layer
        "data/runtime/golden_customer_master.parquet",
        "data/runtime/golden_customer_transactions.parquet",
        "data/runtime/golden_customer_reconciliation.parquet",

        # Customer analytics
        "data/runtime/customer_features.parquet",
        "data/runtime/customer_momentum.parquet",
        "data/runtime/customer_cadence_trend.parquet",
        "data/runtime/customer_value.parquet",
        "data/runtime/clustering_features.parquet",
        "data/runtime/customer_clusters.parquet",
        "data/runtime/customer_priority.parquet",

        # Growth, cohort and LTV
        "data/runtime/customer_ltv.parquet",
        "data/runtime/cohort_retention.parquet",
        "data/runtime/cohort_summary.parquet",
        "data/runtime/customer_growth_monthly.parquet",

        # Propensity modelling - re-engagement
        "data/runtime/reengagement_training.parquet",
        "data/runtime/reengagement_model_scores.parquet",
        "data/runtime/reengagement_model_comparison.parquet",
        "data/runtime/reengagement_model_lift.parquet",
        "data/runtime/reengagement_logistic_coefficients.parquet",
        "data/runtime/reengagement_model_governance.parquet",
        "data/runtime/reengagement_model_metadata.json",

        # Propensity modelling - cross-sell
        "data/runtime/cross_sell_training.parquet",
        "data/runtime/cross_sell_model_scores.parquet",
        "data/runtime/cross_sell_model_comparison.parquet",
        "data/runtime/cross_sell_model_lift.parquet",
        "data/runtime/cross_sell_logistic_coefficients.parquet",
        "data/runtime/cross_sell_model_governance.parquet",
        "data/runtime/cross_sell_model_metadata.json",
        "data/runtime/cross_sell_category_recommendations.parquet",
        "data/runtime/cross_sell_category_validation.parquet",
        "data/runtime/cross_sell_category_candidate_audit.parquet",

        # Propensity modelling - promotion response
        "data/generated/campaign_exposures.parquet",
        "data/generated/campaign_exposure_ground_truth.parquet",
        "data/runtime/promotion_response_training.parquet",
        "data/runtime/promotion_response_model_scores.parquet",
        "data/runtime/promotion_response_model_comparison.parquet",
        "data/runtime/promotion_response_model_lift.parquet",
        "data/runtime/promotion_response_model_governance.parquet",
        "data/runtime/promotion_response_model_metadata.json",
        "data/runtime/promotion_response_operational_scores.parquet",

        # Governed decisioning
        "data/runtime/customer_next_best_action.parquet",
    ]

    for output in outputs:
        print(
            f"  - {output}"
        )

    print()
    print(
        "Note: identity_ground_truth.parquet, "
        "transaction_ground_truth.parquet and "
        "campaign_exposure_ground_truth.parquet are synthetic "
        "QA artefacts only. They are excluded from model features "
        "and operational decisioning. Downstream customer analytics "
        "operate on the resolved golden customer layer."
    )


if __name__ == "__main__":
    main()