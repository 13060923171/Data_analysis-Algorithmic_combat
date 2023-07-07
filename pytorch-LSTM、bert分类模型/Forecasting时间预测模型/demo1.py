from pytorch_forecasting.data.examples import get_stallion_data
from pytorch_forecasting.data import TimeSeriesDataSet,GroupNormalizer
import lightning.pytorch as pl
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_forecasting.metrics import QuantileLoss
from pytorch_forecasting.models import TemporalFusionTransformer
import numpy as np
from pytorch_forecasting.metrics import MAE
import torch

data = get_stallion_data()

#添加时间索引
data["time_idx"] = data["date"].dt.year * 12 + data["date"].dt.month
data["time_idx"] -= data["time_idx"].min()
# 类别必须是字符串
data["month"] = data.date.dt.month.astype(str).astype("category")
data["log_volume"] = np.log(data.volume + 1e-8)
data["avg_volume_by_sku"] = (
    data
        .groupby(["time_idx", "sku"], observed=True)
        .volume.transform("mean")
)
data["avg_volume_by_agency"] = (
    data
        .groupby(["time_idx", "agency"], observed=True)
        .volume.transform("mean")
)
# 我们想把特殊的日子编码为一个变量，因此需要先把一个热度倒过来。
# 因此需要先进行反向的一键编码
special_days = [
    "easter_day", "good_friday", "new_year", "christmas",
    "labor_day", "independence_day", "revolution_day_memorial",
    "regional_games", "fifa_u_17_world_cup", "football_gold_cup",
    "beer_capital", "music_fest"
]
data[special_days] = (
    data[special_days]
        .apply(lambda x: x.map({0: "-", 1: x.name}))
        .astype("category")
)
max_prediction_length = 6   #预测6个月
max_encoder_length = 24 # 使用24个月的历史数据
training_cutoff = data["time_idx"].max() - max_prediction_length
# 创建验证集(predict=True)，这意味着要预测每个系列的最后一个最大预测长度的时间点
training = TimeSeriesDataSet(
    data[lambda x: x.time_idx <= training_cutoff],
    time_idx="time_idx",
    target="volume",
    group_ids=["agency", "sku"],
    min_encoder_length=0,  # 允许没有历史的预测
    max_encoder_length=max_encoder_length,
    min_prediction_length=1,
    max_prediction_length=max_prediction_length,
    static_categoricals=["agency", "sku"],
    static_reals=[
        "avg_population_2017",
        "avg_yearly_household_income_2017"
    ],
    time_varying_known_categoricals=["special_days", "month"],
    # 一组分类变量可以被视为一个变量
    variable_groups={"special_days": special_days},
    time_varying_known_reals=[
        "time_idx",
        "price_regular",
        "discount_in_percent"
    ],
    time_varying_unknown_categoricals=[],
    time_varying_unknown_reals=[
        "volume",
        "log_volume",
        "industry_volume",
        "soda_volume",
        "avg_max_temp",
        "avg_volume_by_agency",
        "avg_volume_by_sku",
    ],
    target_normalizer=GroupNormalizer(
        # 使用softplus，beta=1.0，并按组进行规范化处理
        groups=["agency", "sku"]
    ),
    # 作为特征添加
    add_relative_time_idx=True,
    # 添加为特征
    add_target_scales=True,
    # 添加为特性
    add_encoder_length=True,
)

# 为模型创建数据加载器
validation = TimeSeriesDataSet.from_dataset(
    training, data, predict=True, stop_randomization=True
)
batch_size = 128
train_dataloader = training.to_dataloader(
    train=True, batch_size=batch_size, num_workers=0
)
val_dataloader = validation.to_dataloader(
    train=False, batch_size=batch_size * 10, num_workers=0
)
early_stop_callback = EarlyStopping(
    monitor="val_loss",
    min_delta=1e-4,
    patience=10,
    verbose=False,
    mode="min"
)
logger = TensorBoardLogger("lightning_logs") # 记录到tensorboard# 创建训练器
trainer = pl.Trainer(
    max_epochs=50,
    # 设置为0或1以使用GPU加速训练，设置为None以使用CPU
    gradient_clip_val=0.1,
    callbacks=[early_stop_callback],
    limit_train_batches=30,  # 每30个批次运行一次验证
    # 当验证损失连续10次没有改善时，停止训练
    logger=logger,
)
# 初始化模型
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

# 计算验证集上的MAE
best_model_path = trainer.checkpoint_callback.best_model_path
# 计算验证集的平均绝对误差
best_tft = TemporalFusionTransformer.load_from_checkpoint(best_model_path)
actuals = torch.cat([y[0] for x, y in iter(val_dataloader)])
predictions = best_tft.predict(val_dataloader)
#MAE越小，说明模型的预测性能越好。
mae = MAE()(predictions, actuals)
print(f"MAE: {mae}")



