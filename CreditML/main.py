import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from tpot import TPOTClassifier


def preprocess(data):
    data.loc[data['MonthlyIncome'].isna(), 'MonthlyIncome'] = 1
    data.loc[data['NumberOfDependents'].isna(), 'NumberOfDependents'] = 0  # np.nanmedian(x['NumberOfDependents'])
    return data


df = pd.read_csv('data/cs-training.csv', header=0, index_col=0)
# df = df[:50000]

y = df['SeriousDlqin2yrs']
x = df.drop('SeriousDlqin2yrs', axis=1)
x = preprocess(x)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.2, stratify=y, random_state=11)

# TPOT = TPOTClassifier(
#     generations=100,
#     population_size=24,
#     scoring='roc_auc',
#     cv=4,
#     n_jobs=12,
#     random_state=11,
#     periodic_checkpoint_folder='checkpoint',
#     early_stop=20,
#     verbosity=2,
# )
#
# TPOT.fit(x, y)
# exit(0)

# Average CV score on the training set was: 0.8620023074527077
exported_pipeline = GradientBoostingClassifier(learning_rate=0.1, max_depth=5, max_features=0.05, min_samples_leaf=19, min_samples_split=2, n_estimators=100, subsample=0.7500000000000001)

# exported_pipeline.fit(x_train, y_train)
#
# print(confusion_matrix(
#     y_test,
#     exported_pipeline.predict(x_test)
# ))
# exit(0)

exported_pipeline.fit(x, y)

submission_df = pd.read_csv('data/cs-test.csv', header=0, index_col=0)
submission_x = submission_df.drop('SeriousDlqin2yrs', axis=1)
submission_x = preprocess(submission_x)
submission_y = exported_pipeline.predict_proba(submission_x)[:, 1]

final_df = pd.DataFrame({
    'Id': submission_x.index,
    'Probability': submission_y,
})

final_df.to_csv('data/submission.csv', index=False)
