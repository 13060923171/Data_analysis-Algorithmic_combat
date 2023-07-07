import pandas as pd
# import pytorch_lightning as pl
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.models import TemporalFusionTransformer
from pytorch_forecasting.metrics import QuantileLoss
# from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
import lightning.pytorch as pl
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import LearningRateMonitor
# 加载数据
data = pd.read_csv("BAJAJFINSV.csv")

# 数据预处理
data["date"] = pd.to_datetime(data["date"])
data = data.sort_values("date")
data["time_idx"] = data.groupby("Symbol").cumcount()
data["price"] = (data["Open"] + data["Close"]) / 2


# 定义时间序列数据集
#预测30天的数据
max_encoder_length = 30
# 使用1年的历史数据
max_prediction_length = 365

training_cutoff = data["time_idx"].max() - max_prediction_length

training = TimeSeriesDataSet(
    data[lambda x: x.time_idx <= training_cutoff],
    time_idx="time_idx",
    target="price",
    group_ids=["Symbol"],
    min_encoder_length=0,
    max_encoder_length=max_encoder_length,
    min_prediction_length=1,
    max_prediction_length=max_prediction_length,
    static_categoricals=["Symbol"],
)

validation = TimeSeriesDataSet.from_dataset(training, data,predict=True, stop_randomization=True)
batch_size = 128
train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size * 10, num_workers=0)

early_stop_callback = EarlyStopping(
    monitor="val_loss",
    min_delta=1e-4,
    patience=10,
    verbose=False,
    mode="min"
)

logger = TensorBoardLogger("lightning_logs") # 记录到tensorboard# 创建训练器
# 初始化模型
trainer = pl.Trainer(
    max_epochs=30,
    gradient_clip_val=0.1,
    # callbacks=[early_stop_callback],
    limit_train_batches=30,  # 每30个批次运行一次验证
    # fast_dev_run=True,   # 注释进去以快速检查bug
    logger=logger,
)

tft = TemporalFusionTransformer.from_dataset(
    training,
    learning_rate=0.03,
    hidden_size=16,  # 影响最大的网络规模
    attention_head_size=1,
    dropout=0.1,
    hidden_continuous_size=8,
    output_size=7, # QuantileLoss默认有7个量纲
    loss=QuantileLoss(),
    log_interval=10, # 每10个批次记录一次例子
    reduce_on_plateau_patience=4, # 自动减少学习。
)
tft.size() # 模型中29.6k个参数# 适合网络
trainer.fit(
    tft,
    train_dataloader,
    val_dataloaders=val_dataloader
)

# 预测
predictions = tft.predict(val_dataloader)
