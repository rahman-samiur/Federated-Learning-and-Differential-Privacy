#!/bin/bash

# Define the lists of hyperparameters to test
NOISE_MULTIPLIERS=(3.0)
CLIPPING_NORMS=(0.5 1.0)

# Loop through all combinations
for NM in "${NOISE_MULTIPLIERS[@]}"; do
    for CN in "${CLIPPING_NORMS[@]}"; do
        echo "--- Starting Run: Noise Multiplier = ${NM}, Clipping Norm = ${CN} ---"
        
        # Execute the Flower run, passing the new config values via command line
        # NOTE: Changed '--config' to '--run-config'
        flwr run . \
            --run-config "noise-multiplier=${NM}" \
            --run-config "clipping-norm=${CN}"
        
        echo "--- Run Complete for NM=${NM}, CN=${CN} ---"
    done
done

echo "All grid search runs complete."