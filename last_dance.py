from autogluon.tabular import TabularPredictor
import pandas as pd 
from sklearn.model_selection import train_test_split

data = pd.read_csv('Tweets.csv')
data = data[['airline_sentiment','text']]

train, test = train_test_split(data, test_size=0.33, random_state=42)

predictor = TabularPredictor(label='airline_sentiment').fit(train_data=train)
predictions = predictor.predict(test)

print((predictions == test['airline_sentiment']).mean())