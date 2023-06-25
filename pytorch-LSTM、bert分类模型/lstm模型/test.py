import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.data import Field, TabularDataset, BucketIterator
from torchtext.vocab import GloVe
import spacy

# 定义模型
class TextClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(TextClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(0.5)

    def forward(self, text):
        embedded = self.dropout(self.embedding(text))

        output, _ = self.lstm(embedded)
        hidden = torch.cat((output[-1, :, :hidden_dim], output[0, :, hidden_dim:]), dim=1)
        return self.fc(hidden)


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

# 定义模型和优化器
vocab_size = len(text_field.vocab)
embedding_dim = 100
hidden_dim = 256
output_dim = len(label_field.vocab)
model = TextClassifier(vocab_size, embedding_dim, hidden_dim, output_dim).to(device)
optimizer = optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()

# 训练模型
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    for batch in train_iterator:
        optimizer.zero_grad()
        text, text_lengths = batch.text
        predictions = model(text, text_lengths.cpu())
        loss = criterion(predictions, batch.label)
        loss.backward()
        optimizer.step()

# 测试模型
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for batch in test_iterator:
        text, text_lengths = batch.text
        predictions = model(text, text_lengths.cpu()).squeeze(1)
        predicted_labels = torch.argmax(predictions, dim=1)
        correct += (predicted_labels == batch.label).sum().item()
        total += batch.label.size(0)

accuracy = correct / total
print(f'Test Accuracy: {accuracy * 100:.2f}%')