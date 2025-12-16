import os
import json
import random
import torch
import sys
import argparse
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    default_data_collator,
    TrainerCallback,
    EarlyStoppingCallback
)

CONFIG = {
    "system_prompt": "You are an expert Python programmer. Please read the problem carefully before writing any Python code.", 
    "base_model": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
    "datasets": {
        "DEEP": "Naholav/CodeGen-Deep-5K",
        "DIVERSE": "Naholav/CodeGen-Diverse-5K" 
    },
    "lora_config": {
        "r": 32,
        "lora_alpha": 64,
        "lora_dropout": 0.05,
        "bias": "none",
        "task_type": "CAUSAL_LM",
        "use_dora": True,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"]
    },
    "training_args": {
        "per_device_train_batch_size": 2,
        "gradient_accumulation_steps": 8,
        "learning_rate": 1e-4,
        "num_train_epochs": 3,
        "weight_decay": 0.01,
        "lr_scheduler_type": "cosine_with_restarts",
        "warmup_ratio": 0.05,
        "bf16": True,
        "logging_steps": 20,

        "eval_strategy": "steps",
        "eval_steps": 100,
        "save_strategy": "steps",
        "save_steps": 100,
        "save_total_limit": None,
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
        "per_device_eval_batch_size": 1,

        "max_grad_norm": 0.3,
        "gradient_checkpointing": False,
        "group_by_length": True,
        "report_to": "none"
    },
    "max_seq_length": 2048,
    "seed": 42
}

class TxtLoggerCallback(TrainerCallback):
    def __init__(self, log_path):
        self.log_path = log_path
        # Dosyayı sıfırdan oluştur
        with open(self.log_path, "w") as f:
            f.write("==== Training Log Started ====\n")

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None: return
        # Logları ekle
        with open(self.log_path, "a") as f:
            f.write(json.dumps(logs, indent=4))
            f.write("\n")

def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def preprocess_function(samples, tokenizer, max_seq_len):
    batch_input_ids = []
    batch_attention_mask = []
    batch_labels = []

    inputs = samples["input"]
    solutions = samples["solution"]

    for i, s in zip(inputs, solutions):
        messages_full = [{"role": "system", "content": CONFIG["system_prompt"]}, {"role": "user", "content": i}, {"role": "assistant", "content": s}]
        messages_prompt = [{"role": "system", "content": CONFIG["system_prompt"]}, {"role": "user", "content": i}]

        full_text = tokenizer.apply_chat_template(messages_full, tokenize=False, add_generation_prompt=False)
        prompt_text = tokenizer.apply_chat_template(messages_prompt, tokenize=False, add_generation_prompt=True)

        full_tok = tokenizer(full_text, truncation=True, max_length=max_seq_len, padding="max_length")
        prompt_tok = tokenizer(prompt_text, truncation=True, max_length=max_seq_len)

        input_ids = full_tok["input_ids"]
        attention_mask = full_tok["attention_mask"]
        label = list(input_ids)

        prompt_len = len(prompt_tok["input_ids"])
        for idx in range(min(prompt_len, len(label))):
            label[idx] = -100

        pad_id = tokenizer.pad_token_id
        for idx, tok in enumerate(label):
            if tok == pad_id: label[idx] = -100

        batch_input_ids.append(input_ids)
        batch_attention_mask.append(attention_mask)
        batch_labels.append(label)

    return {"input_ids": batch_input_ids, "attention_mask": batch_attention_mask, "labels": batch_labels}

# --- 3. ANA EĞİTİM FONKSİYONU ---

def run_training(args):
    dataset_key = args.dataset
    
    # Drive Ayarları (Argüman veya Colab kontrolü)
    if args.use_drive:
        try:
            from google.colab import drive
            print("Google Drive bağlanıyor...")
            drive.mount('/content/drive')
            drive_root = "/content/drive/MyDrive/LoRa_Egitim_Sonuclari"
        except ImportError:
            print("Uyarı: 'google.colab' bulunamadı. Drive yerine yerel klasör kullanılacak.")
            drive_root = "./LoRa_Egitim_Sonuclari"
    else:
        drive_root = args.output_dir

    os.makedirs(drive_root, exist_ok=True)

    # Config Güncellemeleri (Terminalden gelen parametrelerle)
    if args.epochs:
        CONFIG["training_args"]["num_train_epochs"] = args.epochs
    
    set_seed(CONFIG["seed"])
    
    if dataset_key not in CONFIG["datasets"]:
        raise ValueError(f"Hata: Geçersiz dataset key: {dataset_key}. Seçenekler: {list(CONFIG['datasets'].keys())}")

    dataset_name = CONFIG["datasets"][dataset_key]
    output_dir = os.path.join(drive_root, f"results_{dataset_key}")
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n==== EĞİTİM BAŞLIYOR: {dataset_key} ====")
    print(f"Kayıt Yeri: {output_dir}")
    print(f"Epoch Sayısı: {CONFIG['training_args']['num_train_epochs']}")

    tokenizer = AutoTokenizer.from_pretrained(CONFIG["base_model"], trust_remote_code=True, padding_side="right")
    if tokenizer.pad_token is None: tokenizer.pad_token = "<|endoftext|>"

    model = AutoModelForCausalLM.from_pretrained(CONFIG["base_model"], trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="auto")
    model.enable_input_require_grads()

    peft_config = LoraConfig(**CONFIG["lora_config"])
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    print(f"Dataset yükleniyor: {dataset_name}...")
    raw = load_dataset(dataset_name)
    if isinstance(raw, dict) and "train" in raw: full = raw["train"]
    else: full = raw

    if "split" in full.column_names:
        print(" Dataset içindeki 'split' etiketine göre ayrılıyor...")
        train_dataset = full.filter(lambda x: x.get("split") == "train")
        test_dataset = full.filter(lambda x: x.get("split") == "test")

        if len(test_dataset) == 0:
            print("UYARI: 'test' split boş geldi! Manuel split yapılıyor...")
            ds_split = train_dataset.train_test_split(test_size=0.05, seed=CONFIG["seed"])
            train_dataset = ds_split["train"]
            test_dataset = ds_split["test"]
    else:
        print(" Manuel split yapılıyor (%95 Train - %5 Test)...")
        ds_split = full.train_test_split(test_size=0.05, seed=CONFIG["seed"])
        train_dataset = ds_split["train"]
        test_dataset = ds_split["test"]

    print(f" Train Size: {len(train_dataset)} | Eval Size: {len(test_dataset)}")

    tokenized_train = train_dataset.map(lambda x: preprocess_function(x, tokenizer, CONFIG["max_seq_length"]), batched=True, batch_size=32, remove_columns=train_dataset.column_names)
    tokenized_eval = test_dataset.map(lambda x: preprocess_function(x, tokenizer, CONFIG["max_seq_length"]), batched=True, batch_size=32, remove_columns=test_dataset.column_names)

    training_args = CONFIG["training_args"]
    hf_args = TrainingArguments(output_dir=output_dir, **training_args)
    
    log_file = os.path.join(output_dir, f"training_logs_{dataset_key}.txt")

    trainer = Trainer(
        model=model,
        args=hf_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        tokenizer=tokenizer,
        data_collator=default_data_collator,
        callbacks=[
            TxtLoggerCallback(log_file),
            EarlyStoppingCallback(early_stopping_patience=2) 
        ]
    )

    print("🚀 Eğitim Başlıyor...")
    trainer.train()

    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    final_config = CONFIG.copy()
    final_config["training_args"]["output_dir"] = output_dir
    with open(os.path.join(output_dir, "training_config.json"), "w") as f: 
        json.dump(final_config, f, indent=4)

    print(f"✅ Bitti! Sonuçlar: {output_dir}")
    del model, trainer
    torch.cuda.empty_cache()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LoRA Training Script for CodeGen Models")

    parser.add_argument(
        "--dataset", 
        type=str, 
        required=True, 
        choices=["DEEP", "DIVERSE"], 
        help="Eğitilecek veri setini seçin: 'DEEP' veya 'DIVERSE'"
    )
    
    parser.add_argument(
        "--use_drive", 
        action="store_true", 
        default=True, 
        help="Google Drive'a kaydet (Colab ortamı için)"
    )

    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="./LoRa_Egitim_Sonuclari", 
        help="Sonuçların kaydedileceği yerel klasör (use_drive kullanılmazsa)"
    )

    parser.add_argument(
        "--epochs", 
        type=int, 
        help="Varsayılan epoch sayısını (3) değiştirmek isterseniz kullanın."
    )

    args = parser.parse_args()

    run_training(args)