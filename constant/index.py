import pandas as pd


class MetricIdx:
    F1 = -5
    ACC = -4
    MCC = -3
    SEN = -2
    SPEC = -1


class DataIdx:
    MLP_FEATURE = 0
    RNN_FEATURE = 1
    LABEL = 2