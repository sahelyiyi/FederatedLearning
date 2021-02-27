import json
import datetime
import tensorflow_datasets as tfds

from deep_learning_lasso.deep_learning_utils import *
from deep_learning_lasso.deep_learning_utils import EPOCHS
from deep_learning_lasso.models import get_NN_model
from deep_learning_lasso.train_models.train_utils import NpEncoder, get_dataframe_and_datasets


n_models = 200

# train different models with different train/validation/test datasets and save its data
for i in range(n_models):

    # get the model
    model = get_NN_model()

    # prepare train, validation, and test datasets
    ds, metadata = tfds.load(
        "cats_vs_dogs",
        shuffle_files=True,
        with_info=True,
    )

    train_ds, train_df, validation_ds, validate_df, test_ds, test_df = get_dataframe_and_datasets(ds)

    # train the mode for the selected train and validation datasets
    start = datetime.datetime.now()
    model.fit(train_ds, epochs=EPOCHS, validation_data=validation_ds)
    print(datetime.datetime.now() - start)

    # calculate the accuracy of the model for the test dataset
    pred_labels = model.predict(test_ds).flatten()
    pred_labels[pred_labels <= 0] = 0
    pred_labels[pred_labels > 0] = 1

    true_labels = []
    for obj in test_ds:
        true_labels += list(np.array(obj[1]))
    true_labels = np.array(true_labels)

    score = np.where(true_labels == pred_labels)[0].shape[0] / test_df.shape[0]
    print ('\n\n\nscore is: ', score)

    model_data = {
        'score': score,
        'train_df': train_df['filename'].values,
        'validate_df': validate_df['filename'].values,
        'test_df': test_df['filename'].values,
        'weights': model.get_weights()[-2:],
    }

    # save trained data
    with open('deep_learning_data/new_deeplearning_%d.json' % i, 'w') as f:
        f.write(json.dumps(model_data, cls=NpEncoder))
