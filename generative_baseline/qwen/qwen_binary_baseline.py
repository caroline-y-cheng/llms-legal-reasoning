import numpy as np
import os
import pandas as pd
from   pathlib import Path
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from unsloth import FastModel


models = {"qwen_4B_binary_baseline" : "Qwen/Qwen3-4B-Instruct-2507", "qwen_30B_binary_baseline" : "Qwen/Qwen3-30B-A3B-Instruct-2507"} 

for model_name, BASE_MODEL in models.items():
    formalism_dir = os.getcwd()
    labeled_data_path = os.path.join(formalism_dir, 'labeled_data', 'final_cleaned_paragraphs.csv')
    output_path = os.path.join(formalism_dir, 'results', model_name)
    generated_output_path = os.path.join(output_path, 'generations' + model_name)
    descriptive_errors_dir = os.path.join(output_path, 'errors')
    
    SEQ_LEN = 1024

    model, tokenizer = FastModel.from_pretrained(
        model_name=BASE_MODEL,
        load_in_4bit=True,
        max_seq_length=SEQ_LEN,
    )

    FastModel.for_inference(model)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    prompt_text = """Some paragraphs in court cases interpret statutes. In this type of paragraph, there is an analysis of a statute and a claim made about its meaning.\n\nIn the following paragraph, determine if legal interpretation occurs ("INTERPRETATION") or not ("NONE")."""

    def get_interp_type(prompt_text, text):
        full_prompt = f"{prompt_text}\n\"{text}\"\n\nYou must respond in a single word. Your options are \"INTERPRETATION\" or \"NONE\". What is the one word that describes this paragraph?"
        messages = [{'role': 'user', 'content': full_prompt}]
        formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(formatted, return_tensors="pt").to(device)

        output = model.generate(
            **inputs,
            max_new_tokens=5,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

        response = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        response = response.split()[0].upper()
        print(response)
        return response


    full_df = pd.DataFrame()
    interpretation_df = pd.read_csv(labeled_data_path)
    interpretation_df = interpretation_df[interpretation_df['class'].notna()]
    interpretation_df["interpretation"] = np.where(interpretation_df["class"].isin(["FORMAL", "GRAND"]), "interpretation", "none")

    macro_f1_l = []
    macro_precision_l = []
    macro_recall_l = []

    weighted_f1_l = []
    weighted_precision_l = []
    weighted_recall_l = []

    one_f1_l = []
    one_precision_l = []
    one_recall_l = []

    zero_f1_l = []
    zero_precision_l = []
    zero_recall_l = []

    for split in range(0, 5):

        split_id_file = os.path.join(formalism_dir, 'train_test_splits', f'split_{split}')

        with open(split_id_file, 'r') as file:
            train_ids = file.read().split("\n")

        interpretation_train_df = interpretation_df[interpretation_df["section_id"].isin(train_ids)]
        interpretation_test_df = interpretation_df[~interpretation_df["section_id"].isin(train_ids)]

        X_test = interpretation_test_df["paragraph"].to_list()
        y_test = interpretation_test_df["interpretation"].to_list()

        predicted_labels = [get_interp_type(prompt_text, text).upper() for text in X_test]

        def clean_predictions(prediction):
            prediction = prediction.strip()
            prediction = prediction.rstrip()
            prediction = prediction.strip('<\s>')
            prediction = prediction.strip('</')
            prediction = prediction.strip('</')
            prediction = prediction.strip('[')
            prediction = prediction.strip(']')
            prediction = prediction.strip('.')
            return prediction

        clean_predicted_labels = [clean_predictions(prediction) for prediction in predicted_labels]


        with open(os.path.join(generated_output_path, f'predictions_{split}.txt'), 'w') as file:
            for label in clean_predicted_labels:
                file.write(f"{label}\n")

    for split in range(0, 5):

        split_id_file = os.path.join(formalism_dir, 'train_test_splits', f'split_{split}')

        with open(split_id_file, 'r') as file:
            train_ids = file.read().split("\n")

        interpretation_train_df = interpretation_df[interpretation_df["section_id"].isin(train_ids)]
        interpretation_test_df = interpretation_df[~interpretation_df["section_id"].isin(train_ids)]

        X_test = interpretation_test_df["paragraph"].to_list()
        y_test = interpretation_test_df["interpretation"].to_list()

        with open(os.path.join(generated_output_path, f'predictions_{split}.txt'), 'r') as file:
            print(file)
            predicted_labels = [line.rstrip().lower() for line in file]

        print(y_test, predicted_labels)
        class_report = classification_report(y_test, predicted_labels, output_dict=True)

        sample_dict = {
            "model": "interpretation_generative",
            "split": split,

            "macro_f1": round(class_report["macro avg"]["f1-score"], 3),
            "macro_precision": round(class_report["macro avg"]["precision"], 3),
            "macro_recall": round(class_report["macro avg"]["recall"], 3),

            "weighted_f1": round(class_report["weighted avg"]["f1-score"], 3),
            "weighted_precision": round(class_report["weighted avg"]["precision"], 3),
            "weighted_recall": round(class_report["weighted avg"]["recall"], 3),

            "1_f1": round(class_report["interpretation"]["f1-score"], 3),
            "1_precision": round(class_report["interpretation"]["precision"], 3),
            "1_recall": round(class_report["interpretation"]["recall"], 3),

            "0_f1": round(class_report["none"]["f1-score"], 3),
            "0_precision": round(class_report["none"]["precision"], 3),
            "0_recall": round(class_report["none"]["recall"], 3),

        }

        new_row = pd.DataFrame(sample_dict, index = [0])
        full_df = pd.concat([full_df, new_row])

        macro_f1_l.append(class_report["macro avg"]["f1-score"])
        macro_precision_l.append(class_report["macro avg"]["precision"])
        macro_recall_l.append(class_report["macro avg"]["recall"])

        weighted_f1_l.append(class_report["weighted avg"]["f1-score"])
        weighted_precision_l.append(class_report["weighted avg"]["precision"])
        weighted_recall_l.append(class_report["weighted avg"]["recall"])

        one_f1_l.append(class_report["interpretation"]["f1-score"])
        one_precision_l.append(class_report["interpretation"]["precision"])
        one_recall_l.append(class_report["interpretation"]["recall"])

        zero_f1_l.append(class_report["none"]["f1-score"])
        zero_precision_l.append(class_report["none"]["precision"])
        zero_recall_l.append(class_report["none"]["recall"])

    macro_f1 = sum(macro_f1_l) / len(macro_f1_l)
    macro_precision = sum(macro_precision_l) / len(macro_precision_l)
    macro_recall = sum(macro_recall_l) / len(macro_recall_l)

    weighted_f1 = sum(weighted_f1_l) / len(weighted_f1_l)
    weighted_precision = sum(weighted_precision_l) / len(weighted_precision_l)
    weighted_recall = sum(weighted_recall_l) / len(weighted_recall_l)

    one_f1 = sum(one_f1_l) / len(one_f1_l)
    one_precision = sum(one_precision_l) / len(one_precision_l)
    one_recall = sum(one_recall_l) / len(one_recall_l)

    zero_f1 = sum(zero_f1_l) / len(zero_f1_l)
    zero_precision = sum(zero_precision_l) / len(zero_precision_l)
    zero_recall = sum(zero_recall_l) / len(zero_recall_l)

    model_dict = {
        "model": "interpretation_generative",
        "split": "averages",

        "macro_f1": round(macro_f1, 3),
        "macro_precision": round(macro_precision, 3),
        "macro_recall": round(macro_recall, 3),

        "weighted_f1": round(weighted_f1, 3),
        "weighted_precision": round(weighted_precision, 3),
        "weighted_recall": round(weighted_recall, 3),

        "1_f1": round(one_f1, 3),
        "1_precision": round(one_precision, 3),
        "1_recall": round(one_recall, 3),

        "0_f1": round(zero_f1, 3),
        "0_precision": round(zero_precision, 3),
        "0_recall": round(zero_recall, 3),
    }

    new_row = pd.DataFrame(model_dict, index = [0])
    full_df = pd.concat([full_df, new_row])

    full_df.to_csv(os.path.join(output_path, model_name + '.csv'))
