import pandas as pd
df=pd.read_csv("news.csv")
print(df.head())


import spacy

nlp=spacy.load("en_core_web_sm")

def clean_text(text):
    doc=nlp(text.lower())

    tokens=[]

    for token in doc:
        if token.is_stop==False and token.is_alpha:
            tokens.append(token.lemma_)

    return" ".join(tokens)

df["clean_text"]=df["text"].apply(clean_text)


from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer=TfidfVectorizer(max_features=5000)

X=vectorizer.fit_transform(df["clean_text"])
y=df["label"]


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)

model=LogisticRegression()
model.fit(X_train,y_train)

pred=model.predict(X_test)

print("Accuracy:",accuracy_score(y_test,pred))


news="NASA confirms alien on the ground"

cleaned=clean_text(news)

vector=vectorizer.transform([cleaned])

prediction=model.predict(vector)

print("Prediction:",prediction[0])