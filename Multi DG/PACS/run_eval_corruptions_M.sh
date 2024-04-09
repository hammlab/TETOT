for trial in "0"
do
    for trg in "0" "1" "2" "3"
    do
        CUDA_VISIBLE_DEVICES=$1 python3 eval_erm_corrupted_M.py --TARGET $trg --TRIAL $trial
    done
done
