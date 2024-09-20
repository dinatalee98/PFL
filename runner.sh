#!/bin/bash

slack_noti_url=$1
ALGORITHMS=( "proposed" "speed" "random" )
N_CLIENTS=( 50 100 200 )


for n_clients in "${N_CLIENTS[@]}"
do
    for algo in "${ALGORITHMS[@]}"
    do
        if [ $algo == 'random' ];
        then
            frac=0.08
        else
            frac=0.12
        fi

        python main.py --algorithm $algo --n_clients $n_clients --frac $frac
    done

    if [ ! -z $slack_noti_url ]; then  
        curl -X POST --data-urlencode "payload={\"channel\": \"lab-notification\", \"text\": \"$n_clients complete\"}" $slack_noti_url
    fi
done