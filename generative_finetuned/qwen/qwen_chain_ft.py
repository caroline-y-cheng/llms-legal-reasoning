import os, math
from datasets import load_dataset
from unsloth import FastModel
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling, EarlyStoppingCallback
from transformers import DataCollatorForLanguageModeling as BaseDataCollator
from torch.optim.lr_scheduler import LambdaLR
from trl import SFTTrainer, SFTConfig

import torch, random, numpy as np
seed = 3407
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
torch.cuda.manual_seed_all(seed)

models = {"qwen_4B_chain" : "Qwen/Qwen3-4B-Instruct-2507", "qwen_30B_chain" : "Qwen/Qwen3-30B-A3B-Instruct-2507"} 

for model_name, BASE_MODEL in models.items():
    SEQ_LEN=2048
    MAX_LR=3e-5 
    TOTAL_STEPS=200  
    WARMUP_RATIO=0.1
    WARMUP_STEPS= int(WARMUP_RATIO * TOTAL_STEPS) 
    MIN_LR_RATIO=0.1
    R,ALPHA,DROPOUT=16,32,0.1  #dropout for regularization
    PER_DEVICE_BS=2  

    SCHEDULER="cosine"

    WORLD_SIZE=int(os.getenv("WORLD_SIZE","1")) or 1
    GRAD_ACCUM=16  # effective batch = 2*16*1 = 32

    train_path = 'train_test_splits/chain_train_4.jsonl'
    eval_path = 'train_test_splits/chain_test_4.jsonl'

    model, tokenizer = FastModel.from_pretrained(
        model_name=BASE_MODEL, 
        load_in_4bit=True, 
        max_seq_length=SEQ_LEN,
    )



    model = FastModel.get_peft_model(
        model, 
        r=R, 
        lora_alpha=ALPHA, 
        lora_dropout=DROPOUT,  
        target_modules = ["q_proj","k_proj","v_proj","o_proj"],
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )

    from unsloth.chat_templates import get_chat_template
    tokenizer = get_chat_template(
        tokenizer,
        chat_template = "qwen3-instruct",
    )

    def format_record(examples):
        prompts = examples["prompt"]
        completions = examples["completion"]

        conversations = []
        for p, c in zip(prompts, completions):
            conversations.append([
                {"role": "user", "content": p},
                {"role": "assistant", "content": c},
            ])

        texts = [
            tokenizer.apply_chat_template(
                convo,
                tokenize=False,
                add_generation_prompt=False,
            )
            for convo in conversations
        ]

        return {"text": texts}

    ds = load_dataset("json", data_files={"train":train_path,"eval":eval_path})
    ds = ds.map(format_record, batched = True, remove_columns=ds["train"].column_names)

    # Tokenize the dataset
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, padding=False, max_length=SEQ_LEN)

    ds = ds.map(tokenize_function, batched=True, remove_columns=["text"])

    args = SFTConfig(
        output_dir="checkpoints",
        per_device_train_batch_size=PER_DEVICE_BS,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=GRAD_ACCUM,
        max_steps=TOTAL_STEPS,
        learning_rate=MAX_LR,
        lr_scheduler_type="cosine",  # cosine decay with warmup
        warmup_steps=WARMUP_STEPS,                 # set steps explicitly (not ratio)
        logging_steps=15, 
        # save_strategy="steps", save_steps=15,
        # eval_strategy="steps", eval_steps=45,
        load_best_model_at_end=False, save_total_limit=6,
        metric_for_best_model="eval_loss", greater_is_better=False,
        bf16=True, weight_decay=0.01, max_grad_norm=1.0, report_to=["none"],
    )

    trainer = SFTTrainer(
        model=model, 
        tokenizer=tokenizer,
        train_dataset=ds["train"], 
        eval_dataset=ds["eval"],
        args=args,
        # callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    from unsloth.chat_templates import train_on_responses_only
    trainer = train_on_responses_only(
        trainer,
        instruction_part = "<|im_start|>user\n",
        response_part = "<|im_start|>assistant\n",
    )

    # trainer.create_optimizer_and_scheduler(num_training_steps=TOTAL_STEPS)

    # dl = trainer.get_train_dataloader()

    trainer.train()

    trainer.evaluate()

    # Save model locally
    import os
    os.makedirs("models", exist_ok=True)
    model_save_path = "models/" + model_name

    model.save_pretrained(model_save_path)
    tokenizer.save_pretrained(model_save_path)
    print(f"Saved adapter at ./{model_save_path}")
