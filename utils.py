
import pandas as pd
from sklearn import model_selection



# divide the data into 5 folds
df = pd.read_csv("train.csv")

df["kfold"] = -1
y = df.category.values

kf = model_selection.StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for f, (t_, v_) in enumerate(kf.split(y, y)):
    df.loc[v_, "kfold"] = f

df.to_csv("train_folds.csv", index=False)

# 字符串处理
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

# use the tokenizer to transform the data
tokenizer.encode_plus(
    s1,
    s2,
    add_special_tokens=True,
    max_length=args.max_len,
    padding="max_length",
    truncation=True,
)
