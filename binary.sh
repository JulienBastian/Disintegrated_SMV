#!/bin/bash
#SBATCH --time=3-00:00:00
#SBATCH -c 8

for r in des_rivasplata_dir des_dir des_rivasplata_gauss des_gauss  MC exact FO SO Rnd 
do
    # data-independent prior
    for d in MUSH HABER SVMGUIDE TTT CODRNA PHIS ADULT
    do
        if [ "$r" = "des_dir" ] ; then
            for order in 1.1 1.25 1.5
            do
                for prior in 0.1 0.25 0.5
                do
                    python3 training.py dataset=$d model.M=10 training.risk=$r training.MC_draws=1 training.batch_size=128 training.risk_type=draw model.pred=stumps-uniform training.order=$order model.prior=$prior
                done
            done
        elif [ "$r" = "des_rivasplata_dir" ] ; then
            for prior in 0.1 0.25 0.5
            do
                python3 training.py dataset=$d model.M=10 training.risk=$r training.MC_draws=1 training.batch_size=128 training.risk_type=draw model.pred=stumps-uniform model.prior=$prior
            done
        elif [ "$r" = "des_gauss" ]; then
            for order in 1.1 1.25 1.5
            do
                for M in 10
                do
                    for var in .1 .9 1.5
                    do
                        python3 training.py dataset=$d model.M=$M training.risk=$r training.MC_draws=1  training.batch_size=128 training.risk_type=draw model.pred=stumps-uniform training.order=$order model.prior=0. model.var=$var
                    done
                done
            done
        elif  [ "$r" = "des_rivasplata_gauss" ]; then
            for M in 10
            do
                for var in .1 .9 1.5
                do
                    python3 training.py dataset=$d model.M=$M training.risk=$r training.MC_draws=1 ttraining.batch_size=128 raining.risk_type=draw model.pred=stumps-uniform model.prior=0. model.var=$var
                done
            done

        elif [ "$r" = "MC" ]; then
            for MC_draws in 10 
            do
                for prior in 0.1 0.25 0.5
                do
                    python3 training.py dataset=$d model.M=10 training.risk=$r training.MC_draws=$MC_draws training.batch_size=128 training.risk_type=stochastic model.pred=stumps-uniform model.prior=$prior
                done
            done

        elif [ "$r" = "exact" ]; then
            for prior in 0.1 0.25 0.5
            do
                python3 training.py dataset=$d model.M=10 training.risk=$r training.batch_size=128 training.risk_type=stochastic model.pred=stumps-uniform model.prior=$prior
            done

        else
            python3 training.py dataset=$d model.M=10 training.risk=$r training.batch_size=128 training.risk_type=stochastic model.pred=stumps-uniform model.prior=1
        fi
    done
done
