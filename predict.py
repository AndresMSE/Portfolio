import sys; print('Python', sys.version)
import pandas as pd
import datetime
from preprop import preprocess_test
from joblib import load


def predict():
    """
    Function to import the data, preprocess it and get predictions
    """
    # Data load
    name = input('CSV file name...  ')
    file = './'+name+'.csv'
    # Preprocessing
    input_val,id = preprocess_test(file)
    # Predictions
    model = load('RandomF.joblib')
    predictions = model.predict(input_val)
    results = pd.DataFrame(predictions, columns=['passholder_type'])
    results = results.join(id, how='left')
    result_enc = {0:'Annual Pass', 1:'Flex Pass', 2:'Monthly Pass', 3:'One Day Pass', 4: 'Testing',5:'Walk-up'}
    results.passholder_type = results.passholder_type.map(result_enc)
    #Export predictions
    results.to_csv(f'{name}-predictions.csv', index=False)
if __name__ == '__main__':
    predict()