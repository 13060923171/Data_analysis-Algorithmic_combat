import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import spacy
from torchtext.data import Field, LabelField, TabularDataset, BucketIterator
import pandas as pd
from torchtext.vocab import GloVe


class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        batch_size, seq_len = x.size()
        # Initialize hidden and cell states
        h0 = torch.zeros(1, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(1, batch_size, self.hidden_size).to(x.device)

        # Pass the input through LSTM layer
        out, (hidden, cell) = self.lstm(x.view(batch_size, seq_len, -1), (h0, c0))
        # Only last hidden state is used as features for classification
        out = self.fc(hidden[-1])

        return out


nlp = spacy.load("en_core_web_sm")
def tokenizer(text):
    return [token.text for token in nlp(text)]

# 数据预处理
text_field = Field(tokenize=tokenizer, include_lengths=True)
label_field = Field(sequential=False, is_target=True)

fields = [('text', text_field), ('category', label_field)]

train_data, test_data = TabularDataset.splits(
    path='data/',
    train='train_data.csv',
    test='test_data.csv',
    format='csv',
    fields=fields,
)

text_field.build_vocab(train_data, vectors=GloVe(name='6B', dim=100))
label_field.build_vocab(train_data)

# 创建迭代器
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_iterator, test_iterator = BucketIterator.splits(
    datasets=(train_data, test_data),
    batch_size=64,
    sort_key=lambda x: len(x.text),
    sort_within_batch=True,
    device=device,
)

# Initialize model
model = LSTMClassifier(input_size=100, hidden_size=50, output_size=10)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# 训练模型
num_epochs = 10
for epoch in range(num_epochs):
    for batch in train_iterator:
        x, lengths = batch.text
        y = batch.category
        print(x)
        print(lengths)
        print(y)
        # Forward pass
        outputs = model(x)

        # Compute loss
        loss = criterion(outputs, y)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print progress
        # ...