#!/bin/bash

echo "Starting all 7 dataset experiments sequentially..."

# 1. ECL
echo "Running ECL..."
bash scripts/multivariate_forecasting/ECL/SimpleTM.sh

# 2. ETTh1
echo "Running ETTh1..."
bash scripts/multivariate_forecasting/ETT/SimpleTM_h1.sh

# 3. ETTh2
echo "Running ETTh2..."
bash scripts/multivariate_forecasting/ETT/SimpleTM_h2.sh

# 4. ETTm1
echo "Running ETTm1..."
bash scripts/multivariate_forecasting/ETT/SimpleTM_m1.sh

# 5. ETTm2
echo "Running ETTm2..."
bash scripts/multivariate_forecasting/ETT/SimpleTM_m2.sh

# 6. Weather
echo "Running Weather..."
bash scripts/multivariate_forecasting/Weather/SimpleTM.sh

# 7. Traffic
echo "Running Traffic..."
bash scripts/multivariate_forecasting/Traffic/SimpleTM.sh

echo "All 7 dataset experiments have been completed!"
