# Generate ground truth for sift-small (10K) update experiments

The basic idea is to generate a larger sift workload with larger topK, and then filter it down to get the ground truth for sift-small update experiments.

```bash
# First, generate a sift-20K ground truth with top-100 neighbors, using SIFT-1M dataset

bash ./gen_subsets.sh CCANN /home/data/deadpool/ANN/SIFT-1M/data/sift_groundtruth.bin bigann-20k /home/data/deadpool/ANN/SIFT-1M/data/sift_query.bin 100

# Second, generate the ground truth for sift-small update experiments with 50K insertions and 5K insertions per batch, 10 batches, and top-10 neighbors

bash ./gen_ground_truth_for_update.sh CCANN /home/data/deadpool/ANN/SIFT-1M/data/sift_groundtruth.bin bigann-small $((5000*10)) 5000 10

# We get the ground truth files stored in /home/data/deadpool/ANN/SIFT-SMALL/update_gt/
```