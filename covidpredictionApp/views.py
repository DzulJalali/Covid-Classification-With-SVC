from django.shortcuts import render
import pandas as pd
from sklearn.model_selection import train_test_split
from collections import Counter
from imblearn.under_sampling import RandomUnderSampler
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier as rfc
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import numpy as np

# Create your views here.

def home(request):
    return render(request, 'home.html', {})

def getPredict(demam, kecapean, batuk, sufas, sateng, tagel, sakit, hiter, pilek, diare, timeng):
    dataset_baru = pd.read_csv(r'E:\Applications\Jupyter Notebook\SistemPakar\tubes\dataset_covid_classification.csv')

    dataset_baru = dataset_baru.sample(frac=1, random_state=12)
    dataset_baru = dataset_baru.iloc[0:5000, :]

    dataset_baru['labels'] = dataset_baru['labels'].replace([0],'Positive')
    dataset_baru['labels'] = dataset_baru['labels'].replace([1],'Negative')

    x = dataset_baru.iloc[:,0:11]
    y = dataset_baru['labels']

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=10)
    undersample = RandomUnderSampler(sampling_strategy='majority')
    x_train_under, y_train_under = undersample.fit_resample(x_train, y_train)

    pca = PCA(n_components=11)
    pca_attributes = pca.fit_transform(x_train_under)
    pca.explained_variance_ratio_

    pca = PCA(n_components = 11)
    x_train_pca = pca.fit_transform(x_train_under)
    x_test_pca = pca.fit_transform(x_test)

    knn = KNeighborsClassifier()
    svc = SVC()
    # tree = DecisionTreeClassifier()
    # randfor = rfc()
    models = [knn, svc]

    scores = []
    for model in models:
        score = np.mean(cross_val_score(model ,x_train_pca, y_train_under,cv=5))
        scores.append(score)

    svc.fit(x_train_pca, y_train_under)

    pred = model.predict(np.array([demam, kecapean, batuk, sufas, sateng, tagel, sakit, hiter, pilek, diare, timeng]).reshape(1,-1))
    # pred = (pred)

    print(pred[0])
    
    if pred == 'Positive':
        return 0
    elif pred == 'Negative':
        return 1
    else :
        return 'error!'
    


def result(request):
    CHECKBOX_MAPPING = {'on':True,
                'off':False,}

    #variable inputan
    demam = CHECKBOX_MAPPING.get(request.GET.get('demam', False))
    kecapean = CHECKBOX_MAPPING.get(request.GET.get('capek', False))
    batuk = CHECKBOX_MAPPING.get(request.GET.get('baker', False))
    sufas = CHECKBOX_MAPPING.get(request.GET.get('sufas', False))
    sateng = CHECKBOX_MAPPING.get(request.GET.get('sateng', False))
    tagel = CHECKBOX_MAPPING.get(request.GET.get('tagel', False))
    sakit = CHECKBOX_MAPPING.get(request.GET.get('sakit', False))
    hiter = CHECKBOX_MAPPING.get(request.GET.get('hiter', False))
    pilek = CHECKBOX_MAPPING.get(request.GET.get('pilek', False))
    diare = CHECKBOX_MAPPING.get(request.GET.get('diare', False))
    timeng = CHECKBOX_MAPPING.get(request.GET.get('timeng', False))

    result = getPredict(demam, kecapean, batuk, sufas, sateng, tagel, sakit, hiter, pilek, diare, timeng)
    return render(request, 'result.html', {"result":result})