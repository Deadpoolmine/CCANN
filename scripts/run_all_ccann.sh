echo This script runs all the tests in the "tests-ccann" directory,
echo and output the results to the "data" directory.
mkdir data

bash $(dirname $0)/tests-ccann/fig11.sh
bash $(dirname $0)/tests-ccann/fig12.sh
bash $(dirname $0)/tests-ccann/fig13.sh
bash $(dirname $0)/tests-ccann/fig14.sh
bash $(dirname $0)/tests-ccann/fig15.sh
bash $(dirname $0)/tests-ccann/fig16.sh
bash $(dirname $0)/tests-ccann/fig17.sh
bash $(dirname $0)/tests-ccann/fig18.sh
