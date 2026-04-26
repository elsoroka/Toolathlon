#!/bin/sh  
#SBATCH --job-name=deepseek-coder-6.7b-vllm
#SBATCH -p batch  
#SBATCH --nodes=1
#SBATCH -A marlowe-m000186-pm05
#SBATCH -G 1
#SBATCH -c 2
#SBATCH --mem-per-gpu=64G
#SBATCH --time=0:30:00
#SBATCH --output=vllm_deepseek-coder-6.7b-instruct_%A.out
#SBATCH --error=vllm_deepseek-coder-6.7b-instruct_error_%A.out
#SBATCH --export=ALL

module load conda
CONDA_BASE="/marlowe/apps/Mambaforge/24.3.0-0"
eval "$($CONDA_BASE/bin/conda shell.bash hook)"
conda activate /scratch/m000186/esoroka/

vllm serve "deepseek-ai/deepseek-coder-6.7b-instruct" --port 8000 &
echo "Waiting for vLLM to be ready..."                                                                                                        
  until curl -s http://localhost:8000/health > /dev/null 2>&1; do                                                                               
      sleep 5                                                                                                                                   
  done                                                                                                                                          
  echo "vLLM is ready"
 
curl -X POST "http://localhost:8000/v1/chat/completions" \
	-H "Content-Type: application/json" \
	--data '{
		"model": "deepseek-ai/deepseek-coder-6.7b-instruct",
		"messages": [
			{
				"role": "user",
				"content": "What is the capital of France?"
			}
		]
	}'
echo "Tested VLLM server"

python eval_client.py run \
  --mode private \
  --base-url http://localhost:8000/v1 \
  --model-name deepseek-coder-6.7b \
  --output-dir ./results \
  --server-host 47.253.6.47 \
  --api-key dummy \
  --workers 1 \
  --server-port 8080 \
  --ws-proxy-port 8081 \
  --task-list-file ./debug-tasks.txt \
  --skip-container-restart

echo "Done"
