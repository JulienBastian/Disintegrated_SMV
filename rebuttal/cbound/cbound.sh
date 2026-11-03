#!/usr/bin/env bash
# C-bound on the binary datasets, launched from the repository root: one task per seed.
read -r -a binary_datasets <<< "${SMV_BINARY_DATASETS:-MUSH HABER SVMGUIDE TTT CODRNA PHIS ADULT SPLICE AUSTRALIAN}"
MAX_JOBS="${MAX_JOBS:-1}"

export PYTHONPATH="rebuttal/cbound/.deps${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

run_tag="cbound-$(date -u +%Y%m%dT%H%M%SZ)-$$"
log_dir="rebuttal/cbound/logs/$run_tag"
mkdir -p "$log_dir"

failures=0
running=0

wait_one() {
    if ! wait -n; then
        failures=$((failures + 1))
    fi
    running=$((running - 1))
}

for trial in 0 1 2 3 4 5 6 7 8 9
do
    for d in "${binary_datasets[@]}"
    do
        echo "Launching $d/$trial -> $log_dir/$d-$trial.log"
        python3 -u rebuttal/cbound/training_cbound.py dataset="$d" training.first_trial="$trial" training.num_trials=1 \
            "hydra.run.dir=rebuttal/cbound/outputs/$run_tag/$d-$trial" > "$log_dir/$d-$trial.log" 2>&1 &
        running=$((running + 1))
        if (( running >= MAX_JOBS )); then
            wait_one
        fi
    done
done
while (( running > 0 )); do
    wait_one
done

# every seed is now cached: this only writes the per-dataset summaries
for d in "${binary_datasets[@]}"
do
    python3 -u rebuttal/cbound/training_cbound.py dataset="$d" "hydra.run.dir=rebuttal/cbound/outputs/$run_tag/$d-summary" >> "$log_dir/summary.log" 2>&1 || failures=$((failures + 1))
done

if (( failures > 0 )); then
    echo "$failures tasks failed. Inspect $log_dir." >&2
    exit 1
fi
echo "All tasks finished."
