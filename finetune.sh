python main.py \
	--base configs/stable-diffusion/v1-finetune_unfrozen.yaml \
	--actual_resume weights/sd-v1-1.ckpt \
	--name catalog_attempt_one \
    --gpus 0, \
	--data_root /home/ubuntu/data/lgs \
	--init_word=catalog \
	--train true
