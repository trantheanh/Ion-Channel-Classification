from constant.index import MetricIdx

"""# Build Logging Funtion"""


def log_result(train_result, dev_result, test_result):
    print("\nDATASET, ACC, MCC, SEN, SPEC")

    print("TRAIN: {}, {}, {}, {}".format(
        train_result[MetricIdx.ACC],
        train_result[MetricIdx.MCC],
        train_result[MetricIdx.SEN],
        train_result[MetricIdx.SPEC])
    )

    print("DEV: {}, {}, {}, {}".format(
        dev_result[MetricIdx.ACC],
        dev_result[MetricIdx.MCC],
        dev_result[MetricIdx.SEN],
        dev_result[MetricIdx.SPEC])
    )

    print("TEST: {}, {}, {}, {}".format(
        test_result[MetricIdx.ACC],
        test_result[MetricIdx.MCC],
        test_result[MetricIdx.SEN],
        test_result[MetricIdx.SPEC])
    )