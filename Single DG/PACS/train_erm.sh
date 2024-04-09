for trial in "0"
do
    for src in "0" "1" "2" "3"
    do
        CUDA_VISIBLE_DEVICES=$1 python3 vanilla_erm.py --SOURCE $src --TRIAL $trial
    done
done
