import sys
import matplotlib.pyplot as plt
import seaborn as sns
color = sns.color_palette()
%matplotlib inline
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
import plotly.express as px
import pandas as pd
import numpy as np
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords

df = pd.read_csv(r"Reviews.csv")

fig = px.histogram(df,x="Score")

fig.update_traces(marker_color="turquoise",marker_line_color='rgb(8,48,107)',marker_line_width=1.5)
fig.update_layout(title_text='Product Score')
fig.show()

df = df[df['Score'] != 3]
df['Sentiment'] = df['Score'].apply(lambda rating : +1 if rating > 3 else -1)
df.head()

positive = df[df['Sentiment'] == 1]
negative = df[df['Sentiment'] == -1]

df['Sentiment'] = df['Sentiment'].replace({-1 : 'negative'})

df['Sentiment'] = df['Sentiment'].replace({1 : 'positive'})

fig = px.histogram(df , x = 'Sentiment')

fig.update_traces(marker_color = 'indianRed',marker_line_color = 'rgb(8,48,107)' , marker_line_width = 1.5)

fig.update_layout(title_text = 'Product Sentiment')

fig.show()

def remove_punc(text):
    final = "".join( u for u in text if u not in ("?", ".", ";", ":", "!",'"'))
    return final
df['Text'] = df['Text'].apply(remove_punc)
df = df.dropna(subset=['Summary'])    
df['Summary'] = df['Summary'].apply(remove_punc)

dfnew = df[['Summary','Sentiment']]
dfnew.head()

index = df.index

df['random'] = np.random.randn(len(index))

train = df[df['random'] <= 0.8]
test = df[df['random'] > 0.8]

vectorizer = CountVectorizer(token_pattern = r'\b\w+\b')
train_matrix = vectorizer.fit_transform(train['Summary'])
test_matrix = vectorizer.transform(test['Summary'])

lr = LogisticRegression()

X_train = train_matrix
X_test = test_matrix
Y_train = train['Sentiment'] 
Y_test = test['Sentiment']

lr.fit(X_train,Y_train)

predictions = lr.predict(X_test)

new = np.asarray(Y_test)

confusion_matrix = (predictions,Y_test)

print(classification_report(predictions,Y_test))
