for trial in "0"
do
    for trg in  "0" "1" "2" 
    do
        echo CUDA_VISIBLE_DEVICES=$1 python3 vanilla_ERM_M.py --TARGET $trg --TRIAL $trial
        CUDA_VISIBLE_DEVICES=$1 python3 vanilla_ERM_M.py --TARGET $trg --TRIAL $trial
    done
done
