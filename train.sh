monitor_gpu_processes() {
    while true; do
        gpu_processes=$(ps -ef | grep run.py | wc -l)

        if [ "$gpu_processes" -gt 1 ]; then
            echo "GPU $gpu_index $gpu_processes sleep 3min"
            sleep 180  
        else
            echo "GPU $gpu_index exit"
            break
        fi
    done
}
export CUDA_VISIBLE_DEVICES=0; nohup python run.py --index 4 --wandb_name all-0 --hidden_size 768 --train_batch_size 32 > train.log 2>&1 &
export CUDA_VISIBLE_DEVICES=1; nohup python run.py --index 4 --wandb_name all-1 --hidden_size 768 --train_batch_size 32 > train.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python run.py --index 4 --wandb_name all-2 --hidden_size 768 --train_batch_size 32 > train.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python run.py --index 4 --wandb_name all-3 --hidden_size 768 --train_batch_size 32 > train.log 2>&1 &
