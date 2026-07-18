import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from torchmetrics import Accuracy

train = pd.read_csv("titanic/train.csv")
test = pd.read_csv("titanic/test.csv")

train_data = train.drop(columns=["Survived"])
y_train = torch.tensor(train["Survived"].values, dtype=torch.float).unsqueeze(dim=1)

df = pd.DataFrame(train_data["Sex"])
le = LabelEncoder()
df['sex_encoded'] = le.fit_transform(df['Sex'])

X_train = torch.tensor(df['sex_encoded'].values, dtype=torch.float32).unsqueeze(dim=1)

df_test = pd.DataFrame(test["Sex"])
df_test['sex_encoded'] = le.transform(df_test['Sex'])

X_test = torch.tensor(df_test['sex_encoded'].values, dtype=torch.float).unsqueeze(dim=1)

class LRTitanic(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(in_features=1, out_features=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.linear(x)
    
RANDOM_SEED = 42

device = 'cuda' if torch.cuda.is_available() else 'cpu'

torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED)

model_0 = LRTitanic().to(device)

X_train, y_train = X_train.to(device), y_train.to(device)
X_test = X_test.to(device)

loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(params=model_0.parameters(),lr=0.1)

accuracy_fn = Accuracy(task="binary").to(device)

epochs = 500
for epoch in range(epochs):
    model_0.train()
    y_logits = model_0(X_train)
    y_preds = torch.sigmoid(y_logits)
    loss = loss_fn(y_logits, y_train)
    acc = accuracy_fn(y_preds, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    model_0.eval()
    with torch.inference_mode():
        test_logits = model_0(X_test)
        test_preds = torch.round(torch.sigmoid(test_logits))

    if epoch % 50 == 0:
        print(f"Epoch: {epoch} | Loss: {loss:.5f} | Accuracy: {acc:.5f}")

    submission = pd.DataFrame({
        'PassengerId': test['PassengerId'],
        'Survived': test_preds.type(torch.int).squeeze().cpu().numpy()
    })
    submission.to_csv('titanicSubmission.csv', index=False)