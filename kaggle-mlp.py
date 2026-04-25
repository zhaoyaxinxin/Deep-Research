import hashlib
import os
import random
import requests
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

# ============================================================
# 1. 可调超参数
# ============================================================
CONFIG = {
    # 模型结构
    "model_type": "mlp",
    "hidden_units": 512,
    "dropout": 0.3,

    # 训练设置
    "k": 5,
    "num_epochs": 500,
    "learning_rate": 0.001,
    "weight_decay": 0.001,
    "batch_size": 128,

    # 早停设置
    "patience": 40,
    "min_delta": 1e-5,

    # 其他
    "seed": 42,
    "output_path": "submission_mlp.csv"
}

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(CONFIG["seed"])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)

# ============================================================
# 2. 下载与读取数据
# ============================================================
DATA_HUB = {{}}
DATA_URL = "http://d2l-data.s3-accelerate.amazonaws.com/"

DATA_HUB["kaggle_house_train"] = (
    DATA_URL + "kaggle_house_pred_train.csv",
    "585e9cc93e70b39160e7921475f9bcd7d31219ce"
)

DATA_HUB["kaggle_house_test"] = (
    DATA_URL + "kaggle_house_pred_test.csv",
    "fa19780a7b011d9b009e8bff8e99922a8ee2eb90"
)

def download(name, cache_dir=os.path.join("..", "data")):
    assert name in DATA_HUB
    url, sha1_hash = DATA_HUB[name]
    os.makedirs(cache_dir, exist_ok=True)
    fname = os.path.join(cache_dir, url.split("/")[-1])

    if os.path.exists(fname):
        sha1 = hashlib.sha1()
        with open(fname, "rb") as f:
            while True:
                data = f.read(1048576)
                if not data:
                    break
                sha1.update(data)
        if sha1.hexdigest() == sha1_hash:
            return fname

    print(f"正在下载 {fname} ...")
    r = requests.get(url, stream=True, verify=True)
    r.raise_for_status()
    with open(fname, "wb") as f:
        f.write(r.content)
    return fname

train_data = pd.read_csv(download("kaggle_house_train"))
test_data = pd.read_csv(download("kaggle_house_test"))

print("train_data:", train_data.shape)
print("test_data:", test_data.shape)

# ============================================================
# 3. 数据预处理
# ============================================================
all_features = pd.concat((train_data.iloc[:, 1:-1], test_data.iloc[:, 1:]))

numeric_features = all_features.dtypes[all_features.dtypes != "object"].index

# 数值缺失标记：先记录缺失，再做标准化和填充
for col in numeric_features:
    all_features[col + "_is_missing"] = all_features[col].isnull().astype(np.float32)

# 数值特征标准化
all_features[numeric_features] = all_features[numeric_features].apply(
    lambda x: (x - x.mean()) / x.std()
)

# 标准化后缺失值填 0，等价于用均值填充
all_features[numeric_features] = all_features[numeric_features].fillna(0)

# 类别变量独热编码，并显式记录类别缺失
all_features = pd.get_dummies(all_features, dummy_na=True)
all_features = all_features.astype(np.float32)

n_train = train_data.shape[0]

train_features = torch.tensor(all_features[:n_train].values, dtype=torch.float32).to(device)
test_features = torch.tensor(all_features[n_train:].values, dtype=torch.float32).to(device)

# 使用 log 房价作为训练标签
train_labels = torch.tensor(
    np.log(train_data.SalePrice.values).reshape(-1, 1),
    dtype=torch.float32
).to(device)

in_features = train_features.shape[1]

print("feature_num:", in_features)
print("train_features:", train_features.shape)
print("test_features:", test_features.shape)

# ============================================================
# 4. MLP 模型定义
# ============================================================
class HousePriceMLP(nn.Module):
    def __init__(self, in_features, hidden_units=512, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_units),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_units, hidden_units // 2),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_units // 2, 1)
        )

    def forward(self, x):
        return self.net(x)

def get_net():
    return HousePriceMLP(
        in_features=in_features,
        hidden_units=CONFIG["hidden_units"],
        dropout=CONFIG["dropout"]
    ).to(device)

loss = nn.MSELoss()
# ============================================================
# 5. 评价指标与训练函数
# ============================================================
def rmse_on_log_price(net, features, labels):
    net.eval()
    with torch.no_grad():
        rmse = torch.sqrt(loss(net(features), labels))
    net.train()
    return rmse.item()


def train(net, train_features, train_labels,
          valid_features=None, valid_labels=None,
          num_epochs=500, learning_rate=0.01,
          weight_decay=0.001, batch_size=64,
          patience=40, min_delta=1e-5):

    train_ls = []
    valid_ls = []

    train_iter = DataLoader(
        TensorDataset(train_features, train_labels),
        batch_size=batch_size,
        shuffle=True
    )

    optimizer = torch.optim.Adam(
        net.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )

    best_valid_loss = float("inf")
    best_state = None
    wait = 0

    for epoch in range(num_epochs):
        net.train()
        for X, y in train_iter:
            optimizer.zero_grad()
            l = loss(net(X), y)
            l.backward()
            optimizer.step()

        train_loss = rmse_on_log_price(net, train_features, train_labels)
        train_ls.append(train_loss)

        if valid_features is not None and valid_labels is not None:
            valid_loss = rmse_on_log_price(net, valid_features, valid_labels)
            valid_ls.append(valid_loss)

            if valid_loss < best_valid_loss - min_delta:
                best_valid_loss = valid_loss
                best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
                wait = 0
            else:
                wait += 1

            if wait >= patience:
                break

    if best_state is not None:
        net.load_state_dict(best_state)
        net.to(device)

    return train_ls, valid_ls
# ============================================================
# 6. K 折交叉验证
# ============================================================
def get_k_fold_data(k, i, X, y):
    assert k > 1
    fold_size = X.shape[0] // k
    X_train, y_train = None, None

    for j in range(k):
        idx = slice(j * fold_size, (j + 1) * fold_size)
        X_part = X[idx, :]
        y_part = y[idx]

        if j == i:
            X_valid, y_valid = X_part, y_part
        elif X_train is None:
            X_train, y_train = X_part, y_part
        else:
            X_train = torch.cat([X_train, X_part], dim=0)
            y_train = torch.cat([y_train, y_part], dim=0)

    return X_train, y_train, X_valid, y_valid


def k_fold(k, X_train, y_train):
    train_l_sum = 0.0
    valid_l_sum = 0.0
    first_train_ls = None
    first_valid_ls = None

    for i in range(k):
        data = get_k_fold_data(k, i, X_train, y_train)
        net = get_net()

        train_ls, valid_ls = train(
            net,
            *data,
            num_epochs=CONFIG["num_epochs"],
            learning_rate=CONFIG["learning_rate"],
            weight_decay=CONFIG["weight_decay"],
            batch_size=CONFIG["batch_size"],
            patience=CONFIG["patience"],
            min_delta=CONFIG["min_delta"]
        )

        train_l_sum += train_ls[-1]
        valid_l_sum += valid_ls[-1]

        if i == 0:
            first_train_ls = train_ls
            first_valid_ls = valid_ls

        print(
            f"折 {i + 1}: "
            f"训练 RMSE(log)={train_ls[-1]:.6f}, "
            f"验证 RMSE(log)={valid_ls[-1]:.6f}, "
            f"训练轮数={len(train_ls)}"
        )

    return train_l_sum / k, valid_l_sum / k, first_train_ls, first_valid_ls
# ============================================================
# 7. 执行 K 折验证
# ============================================================
train_l, valid_l, train_curve, valid_curve = k_fold(
    CONFIG["k"],
    train_features,
    train_labels
)

print(
    f"{CONFIG['k']}-折验证: "
    f"平均训练 RMSE(log)={train_l:.6f}, "
    f"平均验证 RMSE(log)={valid_l:.6f}"
)

plt.figure(figsize=(6, 4))
plt.plot(range(1, len(train_curve) + 1), train_curve, label="train")
plt.plot(range(1, len(valid_curve) + 1), valid_curve, label="valid")
plt.xlabel("epoch")
plt.ylabel("RMSE on log price")
plt.yscale("log")
plt.legend()
plt.show()
# ============================================================
# 8. 全量训练并生成提交文件
# ============================================================
def train_and_pred(train_features, test_features, train_labels, test_data):
    net = get_net()

    train_ls, _ = train(
        net,
        train_features,
        train_labels,
        valid_features=None,
        valid_labels=None,
        num_epochs=CONFIG["num_epochs"],
        learning_rate=CONFIG["learning_rate"],
        weight_decay=CONFIG["weight_decay"],
        batch_size=CONFIG["batch_size"],
        patience=CONFIG["patience"],
        min_delta=CONFIG["min_delta"]
    )

    print(f"全量训练 RMSE(log)={train_ls[-1]:.6f}, 训练轮数={len(train_ls)}")

    plt.figure(figsize=(6, 4))
    plt.plot(range(1, len(train_ls) + 1), train_ls, label="train")
    plt.xlabel("epoch")
    plt.ylabel("RMSE on log price")
    plt.yscale("log")
    plt.legend()
    plt.show()

    net.eval()
    with torch.no_grad():
        # 模型预测的是 log(SalePrice)，需要 exp 还原
        preds = torch.exp(net(test_features)).detach().cpu().numpy()

    submission = pd.concat([
        test_data["Id"],
        pd.Series(preds.reshape(-1), name="SalePrice")
    ], axis=1)

    submission.to_csv(CONFIG["output_path"], index=False)
    print(f"已生成提交文件: {CONFIG['output_path']}")

    return submission

submission = train_and_pred(
    train_features,
    test_features,
    train_labels,
    test_data
)

submission.head()
