import torch
from transformers import BertModel
from torch import nn
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
input_data = '新装三天空调室内漏水'
inputs = tokenizer.encode_plus(
    input_data,
    None,
    add_special_tokens=True,
    max_length=512,
    padding='max_length',
    return_token_type_ids=False,
    return_attention_mask=True,
    return_tensors='pt'
)
input_ids = inputs['input_ids']
attention_mask = inputs['attention_mask']

class BertClassifier(nn.Module):
    def __init__(self, dropout=0.5):
        super(BertClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-chinese')
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(768, 5)
        self.relu = nn.ReLU()

    def forward(self, input_id, mask):
        _, pooled_output = self.bert(input_ids= input_id, attention_mask=mask,return_dict=False)
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)
        final_layer = self.relu(linear_output)
        return final_layer

# 读取模型
model = torch.load('full_model.pkl')
# 将模型设置为评估状态
model.eval()
# 将输入数据传递给模型进行预测


output = model(input_ids, attention_mask)
_, predicted = torch.max(output.data, 1)
print(predicted)