import numpy as np
import pandas as pd
import re
from nltk.stem.porter import PorterStemmer #NLP method
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2
import pickle

#================== ESTIMATOR/VECTORIZER SAVE FUNCTION using PICKLE ============
def save_object(obj, filename):
    with open(filename, 'wb') as output:  # Overwrites any existing file.
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)
        print("estimator has been saved")
#===============================================================================
#### NATURAL LANGUAGE PROCESSING comparing reviews with given scores, and how accurately the words reflect the score given.

#NOTE = the validation of parameters + hyperparamters is based on an example on KAGGLE
dataset = pd.read_csv('reviews.csv') #dataset here is a scraped data from COUSERA reviews + ratings
dataset = dataset.drop(columns = "Id")

#reduce dataset by taking a 1/3 of all values for each label
dataset = dataset.groupby('Label').apply(lambda x: x.sample(n=int(len(x.index)/3))).reset_index(drop = True)
# print(dataset['Label'].value_counts())

#####################################################################
#  pre-processing data
#
#     Convert it to lowercase
#     Remove special characters, such as "!", "?", "." and ","
#     Map words into their stems, so that "liked", "liking" and "likes" are all mapped to "like", as they all indicate similar intentions
#     Remove all stopwords, words that are not relevant to the inference process, i.e.: "the", "why" and "for"
#####################################################################

for i in dataset.index:
    review=re.sub('[^a-zA-Z\']', ' ', dataset['Review'][i]).lower() #make uppercase lowercase
    review=re.sub("'","",review) #remove special characters
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review.split()] #identify stem
    review = ' '.join(review) #add white spaces between words
    dataset.loc[i, 'Review'] = review

# print(dataset)

# use a vectorizer to parse our corpus into a matrix that displays a metric based on how many times each word appears in each review (Term Frequency) and how many times a word appears across all reviews (Inverse Document Frequency). In our vectorizer, we've set the following parameters:
#
#     min_df: This makes it so that words that only appear less than x times in total aren't included in the resulting matrix (used to avoid misspellings)
#     ngram_range: This makes it so that we're not looking only at single words, but also groupings of adjacent words (used to make our search more complete)
vectorizer = TfidfVectorizer(sublinear_tf=True, stop_words='english', min_df=20, ngram_range=(1, 2))
X = vectorizer.fit_transform(dataset['Review']).toarray()
y = dataset.iloc[:, -1].values

# look at which terms best represent each possible rating (1 to 5)
N = 5
for category in range(1, 6):
    term_chi2 = chi2(X, y == category)
    indices = np.argsort(term_chi2[0])
    feature_names = np.array(vectorizer.get_feature_names())[indices]
    print("{} Star Rating's most common terms:\n\t{}".format(category, '\n\t'.join(feature_names[-N:])))
    print()
##############################################################################
#reduce dimensiotnality of data by checking if words match output star rating by looking at p-values
feature_p_values = chi2(X,y)[1]
new_vocabulary = np.extract(feature_p_values < 0.05, vectorizer.get_feature_names())
vectorizer = TfidfVectorizer(sublinear_tf = True, stop_words='english', vocabulary = new_vocabulary)
X = vectorizer.fit_transform(dataset['Review']).toarray()

print(X.shape)
############################################################################
# cross-validate the algorithms we want to compare.
#    comparing:
#     The Naive Bayes Classifier
#     The Random Forest Classifier
#     The Decision Tree Classifier
###### VALIDATION/testing stufff
from sklearn.model_selection import cross_validate
from sklearn.exceptions import ConvergenceWarning
import warnings

warnings.simplefilter("ignore", ConvergenceWarning)
scores = {} #dict
metrics = ['accuracy', 'f1_macro', 'f1_weighted']

from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
# from sklearn.svm import SVC

naive_bayes = MultinomialNB()
random_forest = RandomForestClassifier(n_estimators=60, criterion='gini')
decision_tree = DecisionTreeClassifier(splitter='best', criterion='gini')
# svm = SVC(max_iter=2000, gamma='auto')
scores["Naive Bayes"] = cross_validate(naive_bayes, X, y, scoring=metrics, cv=10, return_estimator=True)
scores["Random Forest"] = cross_validate(random_forest, X, y, scoring=metrics, cv=10, return_estimator=True)
scores["Decision Tree"] = cross_validate(decision_tree, X, y, scoring=metrics, cv=10, return_estimator=True)
# #print estimator scores
flag = False
for algorithm in scores:
    if flag : print()
    flag = True
    print(algorithm, 'Fit Time:', scores[algorithm]['fit_time'].mean())
    print(algorithm, 'Score Time:', scores[algorithm]['score_time'].mean())
    for metric in metrics:
        metric_name = ' '.join([s.capitalize() for s in metric.split('_')])
        print(algorithm, metric_name+':', scores[algorithm]['test_'+metric].mean())

#================ TRY TO IMPROVE CROSS VALIDATION RESULTS ======================
#by comparing which words most accurately predict the outcome, use those words for prediction and therefore reduce the total dataset that is used
print("passed estimator scoring")

total_importances = np.zeros(X.shape[1])
for algorithm in scores:
    best_instance = np.argmax(scores[algorithm]['test_accuracy'])
    if hasattr(scores[algorithm]['estimator'][best_instance], 'feature_importances_'):
        feature_importances = scores[algorithm]['estimator'][best_instance].feature_importances_
        score = scores[algorithm]['test_accuracy'][best_instance]
        total_importances = np.add(total_importances, np.multiply(score, feature_importances))

best_features = np.extract(total_importances > total_importances.mean(), vectorizer.get_feature_names())

vectorizer = TfidfVectorizer(sublinear_tf=True, stop_words='english', vocabulary=best_features)
X = vectorizer.fit_transform(dataset['Review']).toarray()

#run validation again after reducing
scores["Naive Bayes"] = cross_validate(naive_bayes, X, y, scoring=metrics, cv=10, return_estimator=True)
scores["Random Forest"] = cross_validate(random_forest, X, y, scoring=metrics, cv=10, return_estimator=True)
scores["Decision Tree"] = cross_validate(decision_tree, X, y, scoring=metrics, cv=10, return_estimator=True)


#print estimator scores
flag = False
for algorithm in scores:
    if flag : print()
    flag = True
    print(algorithm, 'Fit Time:', scores[algorithm]['fit_time'].mean())
    print(algorithm, 'Score Time:', scores[algorithm]['score_time'].mean())
    for metric in metrics:
        metric_name = ' '.join([s.capitalize() for s in metric.split('_')])
        print(algorithm, metric_name+':', scores[algorithm]['test_'+metric].mean())

#======= DATA HAS NOW BEEN TRANSFORMED/OPTIMISED - now choose best hyperparameters
# #-----------------------------
from sklearn.model_selection import GridSearchCV
#-----------------------------
#RandomForestClassifier
random_forest = RandomForestClassifier() ## now that the estimator has been chosen,
#we need to optimise hyperparamters and choose that verison
parameters = [
    {
        'n_estimators': [10, 20, 30, 40, 50, 60],
        'criterion': ['gini', 'entropy']
    }
]
grid_search = GridSearchCV(random_forest, parameters, scoring='f1_macro', cv=10)
grid_search_fit = grid_search.fit(X, y)
print(grid_search_fit.best_score_)
print(grid_search_fit.best_params_)
final_model=grid_search_fit.best_estimator_  ##this is the best estimator config
# #-------------------------------------

#=============== TRAIN using best estimator and save ============
final_model.fit(X,y)
save_object(final_model, 'estimator.pkl')
save_object(vectorizer, 'vectorizer.pkl')
