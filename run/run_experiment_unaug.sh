#!/bin/bash
                           
#SBATCH --no-requeue
#SBATCH --partition=alpha
#SBATCH --nodes=1                   
#SBATCH --gres=gpu:1
#SBATCH --mem=16000
#SBATCH --mincpus=1
#SBATCH --time=48:00:00                             
#SBATCH --job-name=cifarnet_mcdropout_unaugmented_100_epochs
#SBATCH --mail-type=end
#SBATCH --mail-user=colin.simon@mailbox.tu-dresden.de
#SBATCH --output=output-cifarnet_mcdropout_unaugmented_100_epochs.out
#SBATCH --error=output-cifarnet_mcdropout_unaugmented_100_epochs.out

module --force purge                          				
module load modenv/hiera CUDA/11.7.0 GCCcore/11.3.0 Python/3.10.4

source lib.sh

echo "$job-name"

create_or_reuse_environment

cd /home/"$USER"/scratch/BaaL_EXPERIMENTS/

python ../cifarnet_mcdropout_unaugmented_cifar10.py --augment 0 --epoch 100
