import pandas as pd
from sklearn.model_selection import train_test_split
import torch.nn as nn
from sklearn.model_selection import ShuffleSplit
import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
# скачиваем датасэт и записываем его
URL = "https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.csv"
df = pd.read_csv(URL)
df.to_csv("housing.csv", index=False)

# у нас есть пропуски значений признкака totalbedrooms, их надо заполнить, например медианным значением

print("Пропусков до:", df["total_bedrooms"].isna().sum())

median = df["total_bedrooms"].median()
df.fillna({"total_bedrooms": median}, inplace=True)

print("Пропусков сейчаc:", df["total_bedrooms"].isna().sum())

# на это этапе мы делаем нашу выборку на тренировочную и тестовую, чтобы начать делать scaling и закодирование
# категорильных значений

TARGET = "median_house_value"
CAT_COL = "ocean_proximity"

X = df.drop(columns=[TARGET])
Y = df[TARGET]

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y,
    test_size=0.20,
    random_state=42,
    stratify=X[CAT_COL]
)

# здесь мы начинаем кодировать признаки и делать масштабирование
categories = sorted(X_train[CAT_COL].unique())
print("Категории в train:", categories)


# функция one-hot coding для категориальных признаков: идея такая, мы создаем столбец из нулей, и 1ставим когда
# встречаем определенный класс

def ohe_stable(df, categories):
    out = pd.get_dummies(df, columns=["ocean_proximity"], dtype=int)

    for cat in categories:
        col = f"ocean_proximity_{cat}"
        if col not in out.columns:
            out[col] = 0
        # делаем правильный порядок столбцов, чтобы модель правильно работала
        ohe_columns = [f"ocean_proximity_{cat}" for cat in categories]
        base_columns = [c for c in df.columns if c not in ["ocean_proximity"]]

        return out[base_columns + ohe_columns]


X_train_onehot = ohe_stable(X_train, categories)
X_test_onehot = ohe_stable(X_test, categories)
print("первые 3 строки train с ohe:\n", X_train_onehot.head(3))
# мы закодировали признаки, теперь займемся масштабом

num_cols = [c for c in X_train_onehot.columns if not c.startswith("ocean_proximity_")]
# подготовка массивов со средним и отклонением
means = X_train_onehot[num_cols].mean()
stds = X_train_onehot[num_cols].std(ddof=0).replace(0, 1.0)


def scale_with_train(df, num_cols, means, stds):
    df_scaled = df.copy()
    df_scaled[num_cols] = (df_scaled[num_cols] - means) / stds
    return df_scaled


X_train_scaled = scale_with_train(X_train_onehot, num_cols, means, stds)
X_test_scaled = scale_with_train(X_test_onehot, num_cols, means, stds)

print("первые 3 строки train со scaling:\n", X_train_scaled.head(3))

# теперь мы переводим наши данные из Pandas в Tensor, чтобы работать с PyTorch

X_train_tensor = torch.tensor(X_train_scaled.values, dtype=torch.float32)
Y_train_tensor = torch.tensor(Y_train.values, dtype=torch.float32).view(-1, 1)

X_test_tensor = torch.tensor(X_test_scaled.values, dtype=torch.float32)
Y_test_tensor = torch.tensor(Y_test.values, dtype=torch.float32).view(-1, 1)

# Объединяем в один массив Tensor
train_ds = TensorDataset(X_train_tensor, Y_train_tensor)
test_ds = TensorDataset(X_test_tensor, Y_test_tensor)

train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

print("Пример одного батча:")
for xb, yb in train_loader:
    print("X shape:", xb.shape, "y shape:", yb.shape)
    break


# начинаем строить нашу модель линейной регрессии


class LinearRegressor(nn.Module):
    def __init__(self, in_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, 1)

    def forward(self, x):
        return self.linear(x)


# здесь мы передаём в модель сколько у нас признаков
in_dim = X_train_scaled.shape[1]
model = LinearRegressor(in_features=in_dim)

# здесб выбираем, видеокарту или процессор будем использоваь
device = torch.device("cuba" if torch.cuda.is_available() else "cpu")
model = model.to(device)


# критерий - лосс функция, мы берем RMSE, оптимайзер - как будем убавлять веса
class RMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()  # внутри используем стандартный MSE

    def forward(self, y_pred, y_true):
        return torch.sqrt(self.mse(y_pred, y_true))  # берём корень из MSE


criterion = RMSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

# epochs = 5
# for epoch in range(1, epochs + 1):
#    # цикл по эпохам (каждый проход = полный прогон по train выборке)
#    model.train()
#    # переводим модель в режим обучения (важно для некоторых слоёв, напр. Dropout/BatchNorm)
#
#    running_loss = 0.0  # суммарный лосс за эпоху
#    n_samples = 0  # общее число объектов, пройденных за эпоху
#
#    for xb, yb in train_loader:
#        # xb = признаки (batch_size, in_features)
#        # yb = целевая переменная (batch_size, 1)
#
#        # переносим данные на то же устройство (CPU/GPU), где лежит модель
#        xb = xb.to(device)
#        yb = yb.to(device)
#
#        # прямой проход: считаем предсказания для батча
#        preds = model(xb)
#
#        # считаем функцию потерь по предсказаниям и истинным y
#        loss = criterion(preds, yb)
#
#        # обнуляем старые градиенты (иначе будут копиться от прошлых шагов)
#        optimizer.zero_grad()
#
#        # считаем новые градиенты (обратное распространение)
#        loss.backward()
#
#        # оптимизатор обновляет веса модели по найденным градиентам
#        optimizer.step()
#
#        # аккумулируем сумму ошибок (loss.item() = скалярный лосс для батча)
#        # умножаем на размер батча, чтобы учесть все примеры
#        running_loss += loss.item() * xb.size(0)
#
#        # обновляем количество обработанных объектов
#        n_samples += xb.size(0)
#
#        # считаем средний лосс по всем объектам в эпохе
#    epoch_loss = running_loss / n_samples
#
#    # печатаем номер эпохи и средний train MSE
#    print(f"Epoch {epoch:02d} | train MSE: {epoch_loss:.4f}")

# 1) Делим исходный train на sub-train / val (например, 80/20)
ss = ShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
(train_idx, val_idx) = next(ss.split(X_train_tensor))

X_subtrain = X_train_tensor[train_idx]
y_subtrain = Y_train_tensor[train_idx]
X_val = X_train_tensor[val_idx]
y_val = Y_train_tensor[val_idx]

subtrain_loader = DataLoader(TensorDataset(X_subtrain, y_subtrain), batch_size=64, shuffle=True)
val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=64, shuffle=False)


# 2) Вспомогательные функции: одна эпоха обучения и оценка RMSE
@torch.no_grad()
def evaluate_rmse(model, loader, device):
    model.eval()
    se_sum, n = 0.0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        preds = model(xb)
        se_sum += ((preds - yb) ** 2).sum().item()
        n += yb.numel()
    return (se_sum / n) ** 0.5  # RMSE


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        preds = model(xb)
        loss = criterion(preds, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


# 3) Сетка по lr и цикл подбора
#candidate_lrs = np.logspace(-4, 0, 20)
#epochs = 25
#
#results = []
#for lr in candidate_lrs:
#    # новая модель для каждого lr (важно заново инициализировать веса!)
#    model_lr = LinearRegressor(in_features=X_train_tensor.shape[1]).to(device)
#    optimizer = torch.optim.SGD(model_lr.parameters(), lr=lr)  # можно SGD для наглядности
#    # если используешь наш RMSE как loss:
#    # criterion уже определён как RMSELoss(); иначе можно взять nn.MSELoss()
#
#    for _ in range(epochs):
#        train_one_epoch(model_lr, subtrain_loader, optimizer, criterion, device)
#
#    val_rmse = evaluate_rmse(model_lr, val_loader, device)
#    results.append((lr, val_rmse))
#    print(f"lr={lr:.0e} -> val RMSE: {val_rmse:,.1f}")
#
## 4) Выбор лучшего lr
#best_lr, best_rmse = sorted(results, key=lambda x: x[1])[0]
#print(f"\nBest lr: {best_lr:.0e} | best val RMSE: {best_rmse:,.1f}")

# ---- финальное обучение на всём train с лучшим lr и оценка на test ----
best_lr = 1e-1  # из твоего перебора

# переинициализируем модель и оптимизатор
model = LinearRegressor(in_features=X_train_tensor.shape[1]).to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=best_lr)  # можно и Adam, но оставим SGD для прозрачности

# переопределим train_loader на весь train (если у тебя он уже есть — используй его)
from torch.utils.data import TensorDataset, DataLoader
full_train_loader = DataLoader(TensorDataset(X_train_tensor, Y_train_tensor), batch_size=64, shuffle=True)

# обучим подольше, чтобы «дойти»
epochs = 100
for epoch in range(1, epochs + 1):
    model.train()
    for xb, yb in full_train_loader:
        xb, yb = xb.to(device), yb.to(device)
        preds = model(xb)
        loss = criterion(preds, yb)  # RMSELoss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    if epoch % 20 == 0:
        print(f"[epoch {epoch:03d}] train RMSE (last batch): {loss.item():,.1f}")

# --- оценка на test: RMSE, MAE, R^2 ---
import math
model.eval()
with torch.no_grad():
    se_sum, ae_sum, n = 0.0, 0.0, 0
    y_all, yhat_all = [], []
    for xb, yb in test_loader:
        xb, yb = xb.to(device), yb.to(device)
        yhat = model(xb)
        se_sum += ((yhat - yb) ** 2).sum().item()
        ae_sum += (yhat - yb).abs().sum().item()
        n += yb.numel()
        y_all.append(yb.cpu())
        yhat_all.append(yhat.cpu())

    mse = se_sum / n
    rmse = math.sqrt(mse)
    mae = ae_sum / n

    import torch
    y_all = torch.cat(y_all)
    yhat_all = torch.cat(yhat_all)
    sst = ((y_all - y_all.mean()) ** 2).sum().item()
    sse = ((y_all - yhat_all) ** 2).sum().item()
    r2 = 1.0 - sse / sst

print(f"[TEST] RMSE: {rmse:,.1f} | MAE: {mae:,.1f} | R²: {r2:.4f}")