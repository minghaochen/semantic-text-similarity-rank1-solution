import argparse
import ast
from cgi import test
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
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=False)
    parser.add_argument("--output_name", type=str, required=True)
    parser.add_argument("--input", type=str, default="../input", required=False)
    parser.add_argument("--max_len", type=int, default=196, required=False)
    parser.add_argument("--batch_size", type=int, default=8, required=False)
    return parser.parse_args()


class SimDataset:
    def __init__(self, inputs, tokenizer, max_len):
        self.inputs = inputs
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, item):
        inputs = self.inputs[item]

        ids = inputs["input_ids"]
        mask = inputs["attention_mask"]
        # token_type_ids = inputs["token_type_ids"]

        return {
            "ids": ids,
            "mask": mask,
            # "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
        }


class SimModel(tez.Model):
    def __init__(self, model_name, num_labels):
        super().__init__()
        self.model_name = model_name
        self.num_labels = num_labels

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
        self.output = nn.Linear(config.hidden_size, self.num_labels)

    def forward(self, ids, mask, token_type_ids=None):

        if token_type_ids is not None:
            transformer_out = self.transformer(ids, mask, token_type_ids)
        else:
            transformer_out = self.transformer(ids, mask)
        pooled_output = transformer_out.pooler_output
        logits = self.output(pooled_output)
        logits = torch.softmax(logits, dim=-1)
        return logits, 0, {}


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


class Collate:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        output = dict()
        output["ids"] = [sample["ids"] for sample in batch]
        output["mask"] = [sample["mask"] for sample in batch]

        # calculate max token length of this batch
        batch_max = max([len(ids) for ids in output["ids"]])

        # add padding
        if self.tokenizer.padding_side == "right":
            output["ids"] = [s + (batch_max - len(s)) * [self.tokenizer.pad_token_id] for s in output["ids"]]
            output["mask"] = [s + (batch_max - len(s)) * [0] for s in output["mask"]]
        else:
            output["ids"] = [(batch_max - len(s)) * [self.tokenizer.pad_token_id] + s for s in output["ids"]]
            output["mask"] = [(batch_max - len(s)) * [0] + s for s in output["mask"]]

        # convert to tensors
        output["ids"] = torch.tensor(output["ids"], dtype=torch.long)
        output["mask"] = torch.tensor(output["mask"], dtype=torch.long)

        return output


if __name__ == "__main__":
    args = parse_args()
    seed_everything(42)

    test_df = pd.read_csv(os.path.join(args.input, "test.csv"))  # .head(1000)
    test_df = fix_s1s2(test_df)

    test_df["len1"] = test_df["s1"].apply(lambda x: len(x.split()))
    test_df["len2"] = test_df["s2"].apply(lambda x: len(x.split()))
    test_df["len"] = test_df["len1"] + test_df["len2"]

    # sort by length
    test_df = test_df.sort_values(by="len", ascending=True).reset_index(drop=True)

    # drop len1, len2, len
    test_df = test_df.drop(["len1", "len2", "len"], axis=1)

    test_ids = test_df["id"].values
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    test_inputs = []
    for s1, s2 in tqdm(zip(test_df.s1.values, test_df.s2.values), total=len(test_df)):
        temp_inputs = tokenizer.encode_plus(
            s1,
            s2,
            add_special_tokens=True,
            max_length=args.max_len,
            padding="do_not_pad",
            truncation=True,
        )
        test_inputs.append(temp_inputs)

    test_dataset = SimDataset(
        inputs=test_inputs,
        tokenizer=tokenizer,
        max_len=args.max_len,
    )

    model = SimModel(
        model_name=args.model,
        num_labels=3,
    )

    model.load(args.model_path, weights_only=True)
    coll = Collate(tokenizer)
    preds_iter = model.predict(test_dataset, batch_size=args.batch_size, n_jobs=-1, collate_fn=coll)

    test_preds = []
    for preds in preds_iter:
        test_preds.append(preds)

    test_preds = np.vstack(test_preds)
    test_preds_cat = np.argmax(test_preds, axis=1)

    test_preds_df = pd.DataFrame(
        data={"id": test_ids, "category": test_preds_cat},
        columns=["id", "category"],
    )
    test_preds_df["category"] = test_preds_df["category"].map(
        {
            0: "association",
            1: "disagreement",
            2: "unbiased",
        }
    )
    test_preds_df.to_csv(args.output_name + "_submission.csv", index=False)

    test_preds_raw_df = pd.DataFrame(
        data={
            "id": test_ids,
            "cat1": test_preds[:, 0],
            "cat2": test_preds[:, 1],
            "cat3": test_preds[:, 2],
        },
        columns=["id", "cat1", "cat2", "cat3"],
    )
    test_preds_raw_df.to_csv(args.output_name + "_raw.csv", index=False)