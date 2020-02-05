import numpy as np
import pandas as pd
import re
from nltk.stem.porter import PorterStemmer #NLP method
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2
import pickle
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

final_model = pickle.load(open('estimator.pkl', 'rb'))
vectorizer= pickle.load(open('vectorizer.pkl', 'rb'))

#=============== PREDICT  1  ==================================
dataset_new = pd.read_csv('reviews_by_course.csv')

# dataset_new = dataset_new.drop(columns = "Id")

dataset_new = dataset_new[dataset_new['CourseId'].str.contains('machine-learning', na = False)]
dataset_new = dataset_new.drop(columns = "CourseId")
dataset_new = dataset_new.dropna() #drop all NaN or empty values of the reviews 

Nscale=1 # the scaling data

# #reduce dataset by taking a 1/3 of all values for each label
# dataset_new = dataset_new.groupby('Label').apply(lambda x: x.sample(n=int(len(x.index)/100))).reset_index(drop = True)
dataset_new = dataset_new.groupby('Label').apply(lambda x: x.sample(n=int(len(x.index)/Nscale))).reset_index(drop = True)
# print(dataset_new)

# dataset_new = dataset_new[dataset_new['CourseId'].str.contains('html')]

dataset_labels=dataset_new.drop(columns="Review")
dataset_new = dataset_new.drop(columns = "Label")
print(dataset_new)

# print(dataset_labels)

for i in dataset_new.index:
    review=re.sub('[^a-zA-Z\']', ' ', dataset_new['Review'][i]).lower() #make uppercase lowercase
    review=re.sub("'","",review) #remove special characters
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review.split()] #identify stem
    review = ' '.join(review) #add white spaces between words
    dataset_new.loc[i, 'Review'] = review

# vectorizer = TfidfVectorizer(sublinear_tf=True, stop_words='english', min_df=20, ngram_range=(1, 2))
X_new= vectorizer.transform(dataset_new['Review']).toarray()
#=============== PREDICT  2  ==================================
predicted = final_model.predict(X_new)

# print(np.size(predicted))
# print(predicted)
# print(np.size(dataset_labels))
err=[]
Y=[]
P=[]
for i in range(np.size(predicted)):
    P.append(predicted[i])
    R=dataset_labels['Label'][i]
    err.append(P[i]-R)
    Y.append(i)
#print(err)
# plt.hist(err, weights=np.ones(len(err)) / len(err))
kwargs = dict(histtype='stepfilled', alpha=0.3, normed=True, bins=40)
plt.hist(P, weights=np.ones(len(err)) / len(err), alpha=0.3)
plt.hist(dataset_labels['Label'], weights=np.ones(len(err)) / len(err), alpha=0.3)
plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
plt.show()
