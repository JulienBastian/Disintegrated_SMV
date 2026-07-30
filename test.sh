#!/bin/bash
#SBATCH --time=4-00:00:00
#SBATCH -N 1
#SBATCH --ntasks=16
#SBATCH -c 4
#SBATCH -p LONG3,LONG2,LONG,GPU-UJM

MAX_JOBS=16   # nombre de tâches simultanées
count=0

mkdir -p logs  # crée le dossier logs si nécessaire

run_task () {
    log_file="logs/run_${SLURM_JOB_ID}_$((count+1)).out"
    echo "Launching: $*  -> $log_file"
    srun -N1 -n1 -c4 python3 -u training.py "$@" > "$log_file" 2>&1 &
    ((count++))
    
    # si le nombre de jobs actifs atteint MAX_JOBS, on attend qu'au moins un se termine
    while (( $(jobs -r | wc -l) >= MAX_JOBS )); do
        sleep 30
    done
}

# ------------------ Boucle principale ------------------
for r in exact MC des_rivasplata_dir des_dir FO SO Rnd des_rivasplata_gauss des_gauss
do
    # data-independent priors
    for d in MUSH HABER SVMGUIDE TTT CODRNA PHIS ADULT SPLICE AUSTRALIAN
    do
        if [ "$r" = "des_dir" ]; then
            for order in 1.5; do
                for prior in 0.1 0.25 0.5; do
                    run_task dataset=$d model.M=10 training.risk=$r training.MC_draws=1 training.batch_size=128 \
                             training.risk_type=draw model.pred=stumps-uniform training.order=$order model.prior=$prior
                done
            done
        elif [ "$r" = "des_rivasplata_dir" ]; then
            for prior in 0.1 0.25 0.5; do
                run_task dataset=$d model.M=10 training.risk=$r training.MC_draws=1 training.batch_size=128 \
                         training.risk_type=draw model.pred=stumps-uniform model.prior=$prior
            done
        elif [ "$r" = "des_gauss" ]; then
            for order in 1.5; do
                for var in .1 .9 1.5; do
                    run_task dataset=$d model.M=10 training.risk=$r training.MC_draws=1 training.batch_size=128 \
                             training.risk_type=draw model.pred=stumps-uniform training.order=$order model.prior=0. model.var=$var training.lr=0.001
                done
            done
        elif [ "$r" = "des_rivasplata_gauss" ]; then
            for var in .1 .9 1.5; do
                run_task dataset=$d model.M=10 training.risk=$r training.MC_draws=1 training.batch_size=128 \
                         training.risk_type=draw model.pred=stumps-uniform model.prior=0. model.var=$var training.lr=0.001
            done
        elif [ "$r" = "MC" ]; then
            for prior in 0.1 0.25 0.5; do
                run_task dataset=$d model.M=10 training.risk=$r training.MC_draws=10 training.batch_size=128 \
                         training.risk_type=stochastic model.pred=stumps-uniform model.prior=$prior
            done
        elif [ "$r" = "exact" ]; then
            for prior in 0.1 0.25 0.5; do
                run_task dataset=$d model.M=10 training.risk=$r training.risk_type=stochastic training.batch_size=128 \
                         model.pred=stumps-uniform model.prior=$prior
            done
        else
            run_task dataset=$d model.M=10 training.risk=$r training.risk_type=stochastic training.batch_size=128 \
                     model.pred=stumps-uniform model.prior=1
        fi
    done

    # data-dependent priors
    for d in SENSORLESS PROTEIN PENDIGITS SHUTTLE MNIST FASHION
    do
        for learnt_prior in False True; do
            if [ "$r" = "des_dir" ]; then
                for order in 1.5; do
                    for prior in 0.1 0.25 0.5; do
                        run_task dataset=$d model.M=100 training.risk=$r training.MC_draws=1 training.batch_size=1024 \
                                 training.risk_type=draw model.pred=rf training.order=$order model.prior=$prior pretraining.learnt_prior=$learnt_prior
                    done
                done
            elif [ "$r" = "des_rivasplata_dir" ]; then
                for prior in 0.1 0.25 0.5; do
                    run_task dataset=$d model.M=100 training.risk=$r training.MC_draws=1 training.batch_size=1024 \
                             training.risk_type=draw model.pred=rf model.prior=$prior pretraining.learnt_prior=$learnt_prior
                done
            elif [ "$r" = "des_gauss" ]; then
                for order in 1.5; do
                    for var in .1 .9 1.5; do
                        run_task dataset=$d model.M=100 training.risk=$r training.MC_draws=1 training.batch_size=1024 \
                                 training.risk_type=draw model.pred=rf training.order=$order model.prior=0. model.var=$var pretraining.learnt_prior=$learnt_prior training.lr=0.001
                    done
                done
            elif [ "$r" = "des_rivasplata_gauss" ]; then
                for var in .1 .9 1.5; do
                    run_task dataset=$d model.M=100 training.risk=$r training.MC_draws=1 training.batch_size=1024 \
                             training.risk_type=draw model.pred=rf model.prior=0. model.var=$var pretraining.learnt_prior=$learnt_prior training.lr=0.001
                done
            elif [ "$r" = "MC" ]; then
                for prior in 0.1 0.25 0.5; do
                    run_task dataset=$d model.M=100 training.risk=$r training.MC_draws=10 training.batch_size=1024 \
                             training.risk_type=stochastic model.pred=rf model.prior=$prior pretraining.learnt_prior=$learnt_prior
                done
            elif [ "$r" = "exact" ]; then
                for prior in 0.1 0.25 0.5; do
                    run_task dataset=$d model.M=100 training.risk=$r training.risk_type=stochastic training.batch_size=1024 \
                             model.pred=rf model.prior=$prior pretraining.learnt_prior=$learnt_prior
                done
            else
                run_task dataset=$d model.M=100 training.risk=$r training.risk_type=stochastic training.batch_size=1024 \
                         model.pred=rf model.prior=1 pretraining.learnt_prior=$learnt_prior
            fi
        done
    done
done

wait
echo "All tasks finished."
