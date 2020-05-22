import tensorflow.keras as keras
from metrics.core import BinaryAccuracy, BinaryMCC, BinarySensitivity, BinarySpecificity

model = keras.models.load_model(
	"20200522-130629.h5",
#	 custom_objects={
#		"binary_acc": BinaryAccuracy(),
#		"binary_mcc": BinaryMCC(),
#		"binary_sen": BinarySensitivity(),
#		"binary_spec": BinarySpecificity()
#	},
	compile=False
)
print(model.summary())
