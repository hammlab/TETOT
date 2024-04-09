for trial in "0"
do
    for src in "0" "1" "2" "3"
    do
        for trg in "0" "1" "2" "3"
        do
            if [ $src != $trg ]
            then 
                CUDA_VISIBLE_DEVICES=$1 python3 eval_erm_corrupted.py --SOURCE $src --TARGET $trg --TRIAL $trial
            fi
        done
    done
done
