import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from tpot.builtins import StackingEstimator
from tpot.export_utils import set_param_recursive
from sklearn.preprocessing import FunctionTransformer
from copy import copy

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1)
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'], random_state=11)

# Average CV score on the training set was: 0.8611693722636728
exported_pipeline = make_pipeline(
    make_union(
        VarianceThreshold(threshold=0.0001),
        FunctionTransformer(copy)
    ),
    RandomForestClassifier(bootstrap=True, criterion="entropy", max_features=0.05, min_samples_leaf=16, min_samples_split=17, n_estimators=100)
)
# Fix random state for all the steps in exported pipeline
set_param_recursive(exported_pipeline.steps, 'random_state', 11)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
