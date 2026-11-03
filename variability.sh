#!/usr/bin/env bash
#SBATCH --time=4-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4

# This is intentionally separate from training.sh: every task trains exactly
# one model and then evaluates 100 final draws from the selected hyper-posterior.
read -r -a risks <<< "${SMV_RISKS:-des_rivasplata_dir des_dir}"
read -r -a binary_datasets <<< "${SMV_BINARY_DATASETS:-MUSH HABER SVMGUIDE TTT CODRNA PHIS ADULT SPLICE AUSTRALIAN}"
read -r -a multiclass_datasets <<< "${SMV_MULTICLASS_DATASETS:-SENSORLESS PROTEIN PENDIGITS SHUTTLE MNIST FASHION}"
read -r -a priors <<< "${SMV_PRIORS:-0.5}"
read -r -a orders <<< "${SMV_ORDERS:-1.5}"
read -r -a learnt_priors <<< "${SMV_LEARNT_PRIORS:-False}"

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    MAX_JOBS="${MAX_JOBS:-1}"
else
    MAX_JOBS="${MAX_JOBS:-1}"
fi

count=0
failures=0
pids=()
run_tag="${SLURM_JOB_ID:-variability-$(date -u +%Y%m%dT%H%M%SZ)-$$}"
log_dir="logs/$run_tag"
mkdir -p "$log_dir"

wait_batch() {
    local pid
    for pid in "${pids[@]}"; do
        if ! wait "$pid"; then
            failures=$((failures + 1))
        fi
    done
    pids=()
}

run_task() {
    count=$((count + 1))
    local task_id
    printf -v task_id '%04d' "$count"
    local hydra_dir="outputs/$run_tag/$task_id"
    local log_file="$log_dir/run_${task_id}.log"
    local command=(python3 -u training_variability.py "$@" "hydra.run.dir=$hydra_dir")

    if [[ -n "${SLURM_JOB_ID:-}" ]]; then
        command=(srun --nodes=1 --ntasks=1 --cpus-per-task="${SLURM_CPUS_PER_TASK:-4}" "${command[@]}")
    fi

    echo "Launching variability task $task_id -> $log_file"
    "${command[@]}" > "$log_file" 2>&1 &
    pids+=("$!")
    if (( ${#pids[@]} >= MAX_JOBS )); then
        wait_batch
    fi
}

for r in "${risks[@]}"; do
    for d in "${binary_datasets[@]}"; do
        if [ "$r" = "des_dir" ]; then
            for order in "${orders[@]}"; do
                for prior in "${priors[@]}"; do
                    run_task dataset="$d" model.M=10 training.risk="$r" training.MC_draws=1 \
                        training.batch_size=128 training.risk_type=draw model.pred=stumps-uniform \
                        training.order="$order" model.prior="$prior"
                done
            done
        elif [ "$r" = "des_rivasplata_dir" ]; then
            for prior in "${priors[@]}"; do
                run_task dataset="$d" model.M=10 training.risk="$r" training.MC_draws=1 \
                    training.batch_size=128 training.risk_type=draw model.pred=stumps-uniform \
                    model.prior="$prior"
            done
        elif [ "$r" = "MC" ]; then
            for prior in "${priors[@]}"; do
                run_task dataset="$d" model.M=10 training.risk="$r" training.MC_draws=10 \
                    training.batch_size=128 training.risk_type=stochastic model.pred=stumps-uniform \
                    model.prior="$prior"
            done
        elif [ "$r" = "exact" ]; then
            for prior in "${priors[@]}"; do
                run_task dataset="$d" model.M=10 training.risk="$r" training.batch_size=128 \
                    training.risk_type=stochastic model.pred=stumps-uniform model.prior="$prior"
            done
        else
            run_task dataset="$d" model.M=10 training.risk="$r" training.batch_size=128 \
                training.risk_type=stochastic model.pred=stumps-uniform model.prior=1
        fi
    done
done

for r in "${risks[@]}"; do
    for d in "${multiclass_datasets[@]}"; do
        for learnt_prior in "${learnt_priors[@]}"; do
            if [ "$r" = "des_dir" ]; then
                for order in "${orders[@]}"; do
                    for prior in "${priors[@]}"; do
                        run_task dataset="$d" model.M=100 training.risk="$r" training.MC_draws=1 \
                            training.batch_size=1024 training.risk_type=draw model.pred=rf \
                            training.order="$order" model.prior="$prior" \
                            pretraining.learnt_prior="$learnt_prior"
                    done
                done
            elif [ "$r" = "des_rivasplata_dir" ]; then
                for prior in "${priors[@]}"; do
                    run_task dataset="$d" model.M=100 training.risk="$r" training.MC_draws=1 \
                        training.batch_size=1024 training.risk_type=draw model.pred=rf \
                        model.prior="$prior" pretraining.learnt_prior="$learnt_prior"
                done
            elif [ "$r" = "MC" ]; then
                for prior in "${priors[@]}"; do
                    run_task dataset="$d" model.M=100 training.risk="$r" training.MC_draws=10 \
                        training.batch_size=1024 training.risk_type=stochastic model.pred=rf \
                        model.prior="$prior" pretraining.learnt_prior="$learnt_prior"
                done
            elif [ "$r" = "exact" ]; then
                for prior in "${priors[@]}"; do
                    run_task dataset="$d" model.M=100 training.risk="$r" training.batch_size=1024 \
                        training.risk_type=stochastic model.pred=rf model.prior="$prior" \
                        pretraining.learnt_prior="$learnt_prior"
                done
            else
                run_task dataset="$d" model.M=100 training.risk="$r" training.batch_size=1024 \
                    training.risk_type=stochastic model.pred=rf model.prior=1 \
                    pretraining.learnt_prior="$learnt_prior"
            fi
        done
    done
done

if [[ ${#pids[@]} -gt 0 ]]; then
    wait_batch
fi

if (( failures > 0 )); then
    echo "$failures of $count variability tasks failed. Inspect $log_dir." >&2
    exit 1
fi

echo "All $count variability tasks finished."
