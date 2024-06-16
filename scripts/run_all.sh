#!/bin/bash
scripts=( 
"./scripts/bwrs/ood_prior.sh" \
"./scripts/bwrs/in_dist_cross_validation.sh" \
"./scripts/bwrs/in_dist_vary_prior_ratio/LLMBar.sh" \
"./scripts/bwrs/in_dist_vary_prior_ratio/Hanna.sh" \
"./scripts/bwrs/in_dist_vary_prior_ratio/SummEval.sh" \
"./scripts/bwrs/in_dist_vary_prior_ratio/Meva.sh" \
"./scripts/bwrs/in_dist_vary_prior_ratio/MTBench.sh" \
"./scripts/bwrs/in_dist_vary_prior_ratio/LLMEval2.sh" \

"./scripts/bayesian_ds/ood_prior/Meva.sh" \
"./scripts/bayesian_ds/ood_prior/Hanna.sh" \
"./scripts/bayesian_ds/ood_prior/SummEval.sh" \

"./scripts/bayesian_ds/no_prior/LLMBar.sh" \
"./scripts/bayesian_ds/no_prior/SummEval.sh" \
"./scripts/bayesian_ds/no_prior/Hanna.sh" \
"./scripts/bayesian_ds/no_prior/Meva.sh" \
"scripts/bayesian_ds/no_prior/MTBench.sh" \
"scripts/bayesian_ds/no_prior/LLMEval2.sh" \

"./scripts/bayesian_ds/golden_labels_vary_prior_ratio/LLMBar.sh" \
"./scripts/bayesian_ds/golden_labels_vary_prior_ratio/Hanna.sh" \
"./scripts/bayesian_ds/golden_labels_vary_prior_ratio/SummEval.sh" \
"./scripts/bayesian_ds/golden_labels_vary_prior_ratio/Meva.sh" \
"./scripts/bayesian_ds/golden_labels_vary_prior_ratio/MTBench.sh" \
"./scripts/bayesian_ds/golden_labels_vary_prior_ratio/LLMEval2.sh" \
)

# No more than 16 parallel jobs
num_jobs=16

# Array to store PIDs of running scripts
declare -A pids

# Function to run a script and print its status
run_script() {
    local script="$1"
    if [ -x "$script" ]; then  # Check if the script is executable
        bash "$script" &
        local pid=$!
        pids[$pid]=$script
        echo "$script is running (PID: $pid)"
    else
        echo "Error: $script not found or is not executable."
    fi
}

# Function to check and manage running jobs
manage_jobs() {
    for pid in "${!pids[@]}"; do
        if ! kill -0 $pid 2>/dev/null; then  # Check if the process is still running
            echo "${pids[$pid]} has finished."
            unset pids[$pid]
        fi
    done
}

# Run all scripts 5 times
for i in {1..5}; do
    echo "Starting iteration $i"
    # Loop through the scripts and manage them efficiently
    for script in "${scripts[@]}"; do
        # Ensure an available slot
        while [ "${#pids[@]}" -ge "$num_jobs" ]; do
            manage_jobs
            sleep 1  # Sleep briefly to prevent excessive CPU usage
        done
        # Run the script
        run_script "$script"
    done
    # Final check for any remaining jobs
    while [ "${#pids[@]}" -gt 0 ]; do
        manage_jobs
        sleep 1
    done
    # move results to a new folder
    mkdir -p results_all/iteration_$i
    mv results/* results_all/iteration_$i
    echo "Iteration $i is finished"
done

echo "All scripts are finished"