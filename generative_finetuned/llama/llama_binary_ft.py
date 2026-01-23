import os, math
from datasets import load_dataset
from unsloth import FastLanguageModel
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling, EarlyStoppingCallback
from transformers import DataCollatorForLanguageModeling as BaseDataCollator
from torch.optim.lr_scheduler import LambdaLR

import torch, random, numpy as np
seed = 3407
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
torch.cuda.manual_seed_all(seed)

models = {"llama_8B_binary" : "meta-llama/Meta-Llama-3.1-8B-Instruct", "llama_70B_binary" : "meta-llama/Meta-Llama-3.1-70B-Instruct"} 

for model_name, BASE_MODEL in models.items():
    SEQ_LEN=512
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

    train_path = 'train_test_splits/binary_train_4.jsonl'
    eval_path = 'train_test_splits/binary_test_4.jsonl'



    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL, 
        load_in_4bit=True, 
        max_seq_length=SEQ_LEN,
    )



    model = FastLanguageModel.get_peft_model(
        model, 
        r=R, 
        lora_alpha=ALPHA, 
        lora_dropout=DROPOUT,  
        target_modules = ["q_proj","k_proj","v_proj","o_proj"],
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )

    def format_record(ex):
        prompt = (ex.get("prompt") or "").strip()
        completion = (ex.get("completion") or "").strip()
        return {"text": f"""<|start_header_id|>system<|end_header_id|>
                You are a legal prediction assistant.

                <|start_header_id|>user<|end_header_id|>
                {prompt}

                <|start_header_id|>assistant<|end_header_id|>
                {completion}"""}


    ds = load_dataset("json", data_files={"train":train_path,"eval":eval_path})
    ds = ds.map(format_record, remove_columns=ds["train"].column_names)

    # Tokenize the dataset
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, padding=False, max_length=SEQ_LEN)

    ds = ds.map(tokenize_function, batched=True, remove_columns=["text"])

    args = TrainingArguments(
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

    # collator that masks prompt tokens 
    class CompletionOnlyDataCollator(BaseDataCollator):
        def __init__(self, tokenizer, response_template_tokens, **kwargs):
            super().__init__(tokenizer, **kwargs)
            self.response_template_tokens = response_template_tokens  # token IDs for "<|start_header_id|>assistant<|end_header_id|>\n"
        
        def __call__(self, features):
            # First, get the standard collation
            batch = super().__call__(features)
            labels = batch["labels"]
            input_ids = batch["input_ids"]
            
            # Mask prompt tokens (set to -100) - only keep loss on completion
            for i in range(len(labels)):
                # Find where assistant response starts
                response_start = self._find_response_start(input_ids[i])
                if response_start > 0:
                    # Mask everything before the assistant response
                    labels[i][:response_start] = -100
            
            return batch
        
        def _find_response_start(self, input_ids):
            """Find the position after the assistant header token sequence"""
            template = torch.tensor(self.response_template_tokens, device=input_ids.device, dtype=input_ids.dtype)
            template_len = len(template)
            seq_len = len(input_ids)
            
            if template_len == 0 or seq_len < template_len:
                return 0
            
            # Search for the template sequence using tensor operations
            for i in range(seq_len - template_len + 1):
                if torch.equal(input_ids[i:i+template_len], template):
                    # Return position after the template
                    return i + template_len
            
            # Fallback: if template not found, assume everything is completion
            return 0

    # Tokenize the assistant header to find its token IDs
    assistant_header = "<|start_header_id|>assistant<|end_header_id|>\n"
    assistant_header_tokens = tokenizer.encode(assistant_header, add_special_tokens=False)

    # Create data collator that masks prompt tokens (loss only on completions)
    data_collator = CompletionOnlyDataCollator(
        tokenizer=tokenizer,
        response_template_tokens=assistant_header_tokens,
        mlm=False,  
    )


    trainer = Trainer(
        model=model, 
        tokenizer=tokenizer,
        train_dataset=ds["train"], 
        eval_dataset=ds["eval"],
        args=args,
        data_collator=data_collator,
        # callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
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
