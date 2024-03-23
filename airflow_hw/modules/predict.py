import os
import dill
import pandas as pd


path = os.environ.get('PROJECT_PATH', '..')


def predict():
    best_model = sorted(os.listdir(f'{path}/data/models'))[-1]
    with open(f'{path}/data/models/{best_model}', 'rb') as file:
        model = dill.load(file)

    test_data_path = f'{path}/data/test'
    predictions = []
    for filename in os.listdir(test_data_path):
        test_data = pd.read_json(os.path.join(test_data_path, filename), typ='series')
        test_data_df = pd.DataFrame(test_data).T
        test_predictions = model.predict(test_data_df)
        predictions.extend(test_predictions)
    predictions_df = pd.DataFrame(predictions, columns=['Prediction'])
    predictions_df.to_csv(os.path.join(f'{path}/data/predictions', 'predictions.csv'), index=False)


if __name__ == '__main__':
    predict()
