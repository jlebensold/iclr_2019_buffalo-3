# Run four denmo networks on MNIST, varying the width of the de layer.
(python comparisons.py run_denmo mnist 5 5      400) &
(python comparisons.py run_denmo mnist 25 25    400) &
(python comparisons.py run_denmo mnist 50 50    400) &
(python comparisons.py run_denmo mnist 100 100  400) &
