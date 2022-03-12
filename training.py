import argparse
import ast
import os
import random
import warnings

import numpy as np
import pandas as pd
import tez
import torch
import torch.nn as nn
from sklearn import metrics
from torch.nn import functional as F
from tqdm import tqdm
from transformers import AdamW, AutoConfig, AutoModel, AutoTokenizer, get_linear_schedule_with_warmup
from tez.callbacks import EarlyStopping

warnings.filterwarnings("ignore")


def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--output", type=str, default="../model", required=False)
    parser.add_argument("--input", type=str, default="../input", required=False)
    parser.add_argument("--max_len", type=int, default=196, required=False)
    parser.add_argument("--batch_size", type=int, default=8, required=False)
    parser.add_argument("--valid_batch_size", type=int, default=8, required=False)
    parser.add_argument("--epochs", type=int, default=20, required=False)
    parser.add_argument("--accumulation_steps", type=int, default=1, required=False)
    return parser.parse_args()


class SimDataset:
    def __init__(self, inputs, target, tokenizer, max_len):
        self.inputs = inputs
        self.target = target
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, item):
        inputs = self.inputs[item]

        ids = inputs["input_ids"]
        mask = inputs["attention_mask"]

        return {
            "ids": torch.tensor(ids, dtype=torch.long),
            "mask": torch.tensor(mask, dtype=torch.long),
            "targets": torch.tensor(self.target[item], dtype=torch.long),
        }


class SimModel(tez.Model):
    def __init__(self, model_name, num_train_steps, learning_rate, num_labels, steps_per_epoch):
        super().__init__()
        self.learning_rate = learning_rate
        self.model_name = model_name
        self.num_train_steps = num_train_steps
        self.num_labels = num_labels
        self.steps_per_epoch = steps_per_epoch
        self.step_scheduler_after = "batch"

        hidden_dropout_prob: float = 0.1
        layer_norm_eps: float = 1e-7

        config = AutoConfig.from_pretrained(model_name)

        config.update(
            {
                "output_hidden_states": True,
                "hidden_dropout_prob": hidden_dropout_prob,
                "layer_norm_eps": layer_norm_eps,
                "add_pooling_layer": True,
                "num_labels": self.num_labels,
            }
        )

        self.transformer = AutoModel.from_pretrained(model_name, config=config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.2)
        self.dropout3 = nn.Dropout(0.3)
        self.dropout4 = nn.Dropout(0.4)
        self.dropout5 = nn.Dropout(0.5)
        self.output = nn.Linear(config.hidden_size, self.num_labels)

    def fetch_optimizer(self):
        param_optimizer = list(self.named_parameters())
        no_decay = ["bias", "LayerNorm.bias"]
        optimizer_parameters = [
            {
                "params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                "weight_decay": 0.001,
            },
            {
                "params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        opt = AdamW(optimizer_parameters, lr=self.learning_rate)
        return opt

    def fetch_scheduler(self):
        sch = get_linear_schedule_with_warmup(
            self.optimizer, num_warmup_steps=0, num_training_steps=self.num_train_steps
        )
        return sch

    def loss(self, outputs, targets):
        return nn.CrossEntropyLoss()(outputs, targets)

    def monitor_metrics(self, outputs, targets):
        outputs = torch.argmax(outputs, dim=1).cpu().detach().numpy()
        targets = targets.cpu().detach().numpy()
        f1 = metrics.f1_score(targets, outputs, average="micro")
        return {"f1": f1}

    def forward(self, ids, mask, targets=None):

        transformer_out = self.transformer(ids, mask)
        pooled_output = transformer_out.pooler_output
        pooled_output = self.dropout(pooled_output)

        logits1 = self.output(self.dropout1(pooled_output))
        logits2 = self.output(self.dropout2(pooled_output))
        logits3 = self.output(self.dropout3(pooled_output))
        logits4 = self.output(self.dropout4(pooled_output))
        logits5 = self.output(self.dropout5(pooled_output))

        logits = (logits1 + logits2 + logits3 + logits4 + logits5) / 5
        logits = torch.softmax(logits, dim=-1)
        loss = 0

        if targets is not None:
            loss1 = self.loss(logits1, targets)
            loss2 = self.loss(logits2, targets)
            loss3 = self.loss(logits3, targets)
            loss4 = self.loss(logits4, targets)
            loss5 = self.loss(logits5, targets)
            loss = (loss1 + loss2 + loss3 + loss4 + loss5) / 5
            f1_1 = self.monitor_metrics(logits1, targets)["f1"]
            f1_2 = self.monitor_metrics(logits2, targets)["f1"]
            f1_3 = self.monitor_metrics(logits3, targets)["f1"]
            f1_4 = self.monitor_metrics(logits4, targets)["f1"]
            f1_5 = self.monitor_metrics(logits5, targets)["f1"]
            f1 = (f1_1 + f1_2 + f1_3 + f1_4 + f1_5) / 5
            metric = {"f1": f1}
            return logits, loss, metric

        return logits, loss, {}


def fix_s1s2(data):
    new_s1 = []
    new_s2 = []
    for idx, row in tqdm(data.iterrows(), total=len(data)):
        if row["s1"].startswith("["):
            try:
                temp_s1 = " ".join(ast.literal_eval(row["s1"]))
            except SyntaxError:
                temp_s1 = row["s1"][1:-1]
        else:
            temp_s1 = row["s1"]

        if row["s2"].startswith("["):
            try:
                temp_s2 = " ".join(ast.literal_eval(row["s2"]))
            except SyntaxError:
                temp_s2 = row["s2"][1:-1]
        else:
            temp_s2 = row["s2"]

        new_s1.append(temp_s1)
        new_s2.append(temp_s2)
    data["s1"] = new_s1
    data["s2"] = new_s2
    return data


if __name__ == "__main__":
    args = parse_args()
    seed_everything(42)
    os.makedirs(args.output, exist_ok=True)

    df = pd.read_csv(os.path.join(args.input, "train_folds.csv"))  # .head(1000)
    df = fix_s1s2(df)

    label_mapping = {
        "association": 0,
        "disagreement": 1,
        "unbiased": 2,
    }
    df.category = df.category.map(label_mapping)

    train_df = df[df["kfold"] != args.fold].reset_index(drop=True)
    valid_df = df[df["kfold"] == args.fold].reset_index(drop=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    train_inputs = []
    for s1, s2 in tqdm(zip(train_df.s1.values, train_df.s2.values), total=len(train_df)):
        temp_inputs = tokenizer.encode_plus(
            s1,
            s2,
            add_special_tokens=True,
            max_length=args.max_len,
            padding="max_length",
            truncation=True,
        )
        train_inputs.append(temp_inputs)

    valid_inputs = []
    for s1, s2 in tqdm(zip(valid_df.s1.values, valid_df.s2.values), total=len(valid_df)):
        temp_inputs = tokenizer.encode_plus(
            s1,
            s2,
            add_special_tokens=True,
            max_length=args.max_len,
            padding="max_length",
            truncation=True,
        )
        valid_inputs.append(temp_inputs)

    train_dataset = SimDataset(
        inputs=train_inputs,
        target=train_df.category.values,
        tokenizer=tokenizer,
        max_len=args.max_len,
    )

    valid_dataset = SimDataset(
        inputs=valid_inputs,
        target=valid_df.category.values,
        tokenizer=tokenizer,
        max_len=args.max_len,
    )

    num_train_steps = int(len(train_dataset) / args.batch_size / args.accumulation_steps * args.epochs)

    model = SimModel(
        model_name=args.model,
        num_train_steps=num_train_steps,
        learning_rate=args.lr,
        num_labels=3,
        steps_per_epoch=len(train_dataset) / args.batch_size,
    )

    es = EarlyStopping(
        model_path=os.path.join(args.output, f"model_{args.fold}.bin"),
        patience=5,
        mode="max",
        delta=0.001,
        save_weights_only=True,
        monitor="valid_f1",
    )

    model.fit(
        train_dataset,
        train_bs=args.batch_size,
        valid_dataset=valid_dataset,
        valid_bs=args.valid_batch_size,
        device="cuda",
        epochs=args.epochs,
        callbacks=[es],
        fp16=True,
        accumulation_steps=args.accumulation_steps,
    )