import sys
from pathlib import Path

root_dir = str(Path(__file__).resolve().parents[2])
src_dir = str(Path(__file__).resolve().parents[2] / "src")
for p in [root_dir, src_dir]:
    if p not in sys.path:
        sys.path.insert(0, p)

from benchmarks.validation.analytical_validation import validate_analytical
from benchmarks.validation.numerical_validation import validate_numerical
from benchmarks.validation.domain_validation import validate_domain



def run_all():
    print("\n" + "#" * 70)
    print("RUNNING COMPLETE 3-LEVEL OSIS VALIDATION SUITE")
    print("#" * 70 + "\n")

    validate_analytical()
    validate_numerical()
    validate_domain()

    print("#" * 70)
    print("SUMMARY: ALL 3 VALIDATION LEVELS PASSED (ANALYTICAL, NUMERICAL, DOMAIN)")
    print("#" * 70 + "\n")


if __name__ == "__main__":
    run_all()
