#!/bin/bash

# Define the notebook to be run
NOTEBOOK_TEMPLATE="ToyModel_emulator.ipynb"

# Define the seed values (10 different seeds)
SEEDS=(40134)

# Define the training size values (from 10 to 100 in increments of 10)
TRAIN_SIZES=(40 50 60 70 80 90 100)

DETMAX=(True False)

# Create the output directory if it doesn't exist
mkdir -p output_notebooks

# Loop through each seed and training size
for seed in "${SEEDS[@]}"
do
    for train_size in "${TRAIN_SIZES[@]}"
    do
        for detmax in "${DETMAX[@]}"
        do
            echo "Running notebook with seed $seed and train size $train_size..."

            # Save the output notebook in the output_notebooks directory
            OUTPUT_NOTEBOOK="output_notebooks/notebook_seed_${seed}_train_${train_size}.ipynb"

            # Run the notebook with papermill, passing the parameters
            papermill "$NOTEBOOK_TEMPLATE" "$OUTPUT_NOTEBOOK" \
                -p seed "$seed" \
                -p train_size "$train_size" \
                -p dmax "$detmax"\
                -p skip_cell True

            echo "Finished running notebook with seed $seed and train size $train_size. Output saved to $OUTPUT_NOTEBOOK."
        done
    done
done

#echo "Deleting output_notebooks directory..."
rm -rf output_notebooks
#echo "Directory deleted."