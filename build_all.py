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
    PipelineStep(
        "Generate synthetic customer transactions",
        "src.data_generation.generate_transactions",
    ),
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
    PipelineStep(
        "Build Next Best Action recommendations",
        "src.decision_engine.next_best_action",
    ),
]


QA_STEPS = [
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
            "Reuse existing generated customer "
            "and transaction data."
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
            != "src.data_generation.generate_transactions"
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
        "\nCore runtime outputs:"
    )

    outputs = [
        "data/generated/customer_master.parquet",
        "data/generated/transactions.parquet",
        "data/runtime/customer_features.parquet",
        "data/runtime/customer_momentum.parquet",
        "data/runtime/customer_cadence_trend.parquet",
        "data/runtime/customer_value.parquet",
        "data/runtime/clustering_features.parquet",
        "data/runtime/customer_clusters.parquet",
        "data/runtime/customer_priority.parquet",
        "data/runtime/customer_ltv.parquet",
        "data/runtime/cohort_retention.parquet",
        "data/runtime/cohort_summary.parquet",
        "data/runtime/customer_growth_monthly.parquet",
        "data/runtime/customer_next_best_action.parquet",
    ]

    for output in outputs:
        print(
            f"  - {output}"
        )


if __name__ == "__main__":
    main()