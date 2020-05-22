import tensorflow.keras as keras
from metrics.core import BinaryAccuracy, BinaryMCC, BinarySensitivity, BinarySpecificity
from process.core import build_test_ds
from constant.index import DataIdx, MetricIdx
from data import loader
import numpy as np

threshold = 0.7

folds_data, test_data = loader.get_data(5)
extra_train_data = loader.parse_csv_data("resource/Added.Neg.Sample.Train.csv")
extra_test_data = loader.parse_csv_data("resource/Added.Neg.Sample.Test.csv")
results = []
for i in range(5):
	_, dev_data = loader.get_fold(folds_data=folds_data, fold_index=i)
	dev_ds = build_test_ds(
		mlp_x=dev_data[DataIdx.MLP_FEATURE],
		rnn_x=dev_data[DataIdx.RNN_FEATURE],
		y=dev_data[DataIdx.LABEL]
	)

	dev_ds = dev_ds.concatenate(build_test_ds(
		mlp_x=extra_train_data[DataIdx.MLP_FEATURE],
		rnn_x=extra_train_data[DataIdx.RNN_FEATURE],
		y=extra_train_data[DataIdx.LABEL]
	))

	model: keras.models.Model = keras.models.load_model(
		"saved_model/f{}.h5".format(i+1),
		compile=False
	)

	model.compile(
		optimizer="nadam",
		loss=keras.losses.binary_crossentropy,
		metrics=[
			BinaryAccuracy(threshold),
			BinaryMCC(threshold),
			BinarySensitivity(threshold),
			BinarySpecificity(threshold)
		])

	result = model.evaluate(dev_ds, verbose=0)
	results.append(result)
	print(result)
	# break

results = np.array(results)
print(np.mean(results, axis=0))

model: keras.models.Model = keras.models.load_model(
		"saved_model/final_model.h5",
		compile=False
	)

model.compile(
	optimizer="nadam",
	loss=keras.losses.binary_crossentropy,
	metrics=[
		BinaryAccuracy(threshold),
		BinaryMCC(threshold),
		BinarySensitivity(threshold),
		BinarySpecificity(threshold)
	])

test_ds = build_test_ds(
		mlp_x=test_data[DataIdx.MLP_FEATURE],
		rnn_x=test_data[DataIdx.RNN_FEATURE],
		y=test_data[DataIdx.LABEL]
	)

test_ds = test_ds.concatenate(build_test_ds(
		mlp_x=extra_test_data[DataIdx.MLP_FEATURE],
		rnn_x=extra_test_data[DataIdx.RNN_FEATURE],
		y=extra_test_data[DataIdx.LABEL]
	))

result = model.evaluate(test_ds, verbose=0)
print(result)