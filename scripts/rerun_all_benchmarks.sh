#!/bin/bash
# Re-run all benchmarks on GPU and regenerate figures.
# Run from repo root after CUDA is available:
#   bash scripts/rerun_all_benchmarks.sh

set -e

echo "=== Checking CUDA ==="
python3 -c "import torch; assert torch.cuda.is_available(), 'CUDA not available!'; print('GPU:', torch.cuda.get_device_name(0))"

cd src

echo ""
echo "=== 1. Verify paper numbers (GPU) ==="
python3 verify_paper_numbers.py

echo ""
echo "=== 2. Metaheuristic benchmarks (N=10) ==="
python3 benchmark_metaheuristics.py \
    --dataset ../data/dataset_10_objects.json \
    --device cuda --max-scenarios 50 \
    --output ../results_metaheuristic_n10.json

echo ""
echo "=== 3. Metaheuristic benchmarks (N=40) ==="
python3 benchmark_metaheuristics.py \
    --dataset ../data/dataset_40_objects.json \
    --device cuda --max-scenarios 50 \
    --output ../results_metaheuristic_n40.json

echo ""
echo "=== 4. Metaheuristic benchmarks (N=200) ==="
python3 benchmark_metaheuristics.py \
    --dataset ../data/dataset_200_objects.json \
    --device cuda --max-scenarios 20 \
    --output ../results_metaheuristic_n200.json

echo ""
echo "=== 5. k-sensitivity ablation ==="
python3 benchmark_k_sensitivity.py \
    --device cuda --max-scenarios 20 \
    --output ../results_k_sensitivity.json

cd ..

echo ""
echo "=== 6. Generate figures ==="
python3 scripts/plot_k_pareto.py
python3 scripts/plot_pareto_scatter.py
python3 scripts/plot_timing_scaling_v2.py

echo ""
echo "=== ALL DONE ==="
echo "Results saved to:"
echo "  results_metaheuristic_n10.json"
echo "  results_metaheuristic_n40.json"
echo "  results_metaheuristic_n200.json"
echo "  results_k_sensitivity.json"
echo ""
echo "Figures saved to:"
echo "  references/figures/paper/fig_k_pareto.png"
echo "  references/figures/paper/fig_pareto_scatter.png"
echo "  references/figures/paper/fig_timing_scaling_v2.png"
echo ""
echo ">>> UPDATE references/revised_paper_v7.tex Table I GNN timings with GPU values <<<"
echo ">>> Look for lines marked % TODO-GPU in the tex file <<<"
