#!/bin/bash
#SBATCH --time=3-00:00:00
#SBATCH -c 8

for r in des_rivasplata_dir des_dir MC exact FO SO Rnd des_rivasplata_gauss des_gauss 
do
    #data dependent priors
    for d in SENSORLESS PROTEIN PENDIGITS SHUTTLE MNIST FASHION
    do
        for learnt_prior in False True
        do
            if [ "$r" = "des_dir" ] ; then
                for order in 1.1 1.25 1.5
                do
                    for prior in 0.1 0.25 0.5
                    do
                        python3 training.py dataset=$d model.M=100 training.risk=$r training.MC_draws=1 training.batch_size=1024 training.risk_type=draw model.pred=rf training.order=$order model.prior=$prior pretraining.learnt_prior=$learnt_prior
                    done
                done
            elif [ "$r" = "des_rivasplata_dir" ] ; then
                 for prior in 0.1 0.25 0.5
                do
                    python3 training.py dataset=$d model.M=100 training.risk=$r training.MC_draws=1 training.batch_size=1024 training.risk_type=draw model.pred=rf model.prior=$prior pretraining.learnt_prior=$learnt_prior
                done
            
            elif [ "$r" = "des_gauss" ] ; then
                for order in 1.1 1.25 1.5
                do
                    for prior in 0.
                    do
                        for var in .1 .9 1.5
                        do
                            python3 training.py dataset=$d model.M=100 training.risk=$r training.MC_draws=1 training.batch_size=1024 training.risk_type=draw model.pred=rf training.order=$order model.prior=$prior pretraining.learnt_prior=$learnt_prior
                        done
                    done
                done
            elif [ "$r" = "des_rivasplata_gauss" ] ; then
                for prior in 0.
                do
                    for var in .1 .9 1.5
                    do
                        python3 training.py dataset=$d model.M=100 training.risk=$r training.MC_draws=1 training.batch_size=1024 training.risk_type=draw model.pred=rf model.prior=$prior pretraining.learnt_prior=$learnt_prior
                    done
                done

            elif [ "$r" = "MC" ]; then
                for MC_draws in 10 
                do
                    for prior in 0.1 0.25 0.5 
                    do
                        python3 training.py dataset=$d model.M=100 training.risk=$r training.MC_draws=$MC_draws training.batch_size=1024 training.risk_type=stochastic model.pred=rf model.prior=$prior pretraining.learnt_prior=$learnt_prior
                    done
                done

            elif [ "$r" = "exact" ]; then
                for prior in 0.1 0.25 0.5
                do
                    python3 training.py dataset=$d model.M=100 training.risk=$r training.batch_size=1024 training.risk_type=stochastic model.pred=rf model.prior=$prior pretraining.learnt_prior=$learnt_prior
                done

            else
                python3 training.py dataset=$d model.M=100 training.risk=$r training.batch_size=1024 training.risk_type=stochastic model.pred=rf model.prior=1 pretraining.learnt_prior=$learnt_prior
            fi
        done
    done
done
