# clean all
# for d in ./runs/ccsa/vgg16_alpha_search_coarse_grad_ft
# do
#     if [[ $d != *_best ]]
#     then
#         echo $d/*/checkpoints
#         rm -rf $d/*/checkpoints
#         # rm -rf $d/*/logs
#     fi
# done

# clean faulty (no report.json)
for D in ./runs/dsne/*
do
    if [[ $D != *_best ]]
    then
        for E in $D/*
        do
            FILE=$E/report.json
            if [ ! -f "$FILE" ]; then
                if [ -d "$E" ]; then
                    rm -rf $E
                    echo "$E deleted"
                fi
            fi
        done
    fi
done