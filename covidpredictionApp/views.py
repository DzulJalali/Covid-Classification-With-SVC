from django.shortcuts import render
import pickle
import numpy as np


# Create your views here.

def home(request):
    return render(request, 'home.html', {})

def getPredict(susahnafas, demam, baker, sateng, hipertensi, lelah, jjln, contact, kumpul, public, keluarga):
    # dataset_baru = pd.read_csv(r'E:\Applications\Jupyter Notebook\SistemPakar\tubes\dataset_covid_classification.csv')
    
    model = pickle.load(open(r'E:\Applications\Jupyter Notebook\SistemPakar\TB\covid_dataset.sav', "rb"))
    scaled = pickle.load(open(r'E:\Applications\Jupyter Notebook\SistemPakar\TB\scaler.sav', "rb"))
    prediction = model.predict(scaled.transform([
        [susahnafas, demam, baker, sateng, hipertensi, lelah, jjln, contact, kumpul, public, keluarga]
    ]))
    
    
    if prediction == 1:
        return 0
    elif prediction == 0:
        return 1
    else :
        return 'error!'
    


def result(request):
    CHECKBOX_MAPPING = {'on':True,
                'off':False,}

    #variable inputan
    susahnafas = CHECKBOX_MAPPING.get(request.GET.get('susahnafas', False))
    demam = CHECKBOX_MAPPING.get(request.GET.get('demam', False))
    baker = CHECKBOX_MAPPING.get(request.GET.get('baker', False))
    sateng = CHECKBOX_MAPPING.get(request.GET.get('sateng', False))
    hipertensi = CHECKBOX_MAPPING.get(request.GET.get('hipertensi', False))
    lelah = CHECKBOX_MAPPING.get(request.GET.get('lelah', False))
    jjln = CHECKBOX_MAPPING.get(request.GET.get('jjln', False))
    contact = CHECKBOX_MAPPING.get(request.GET.get('contact', False))
    kumpul = CHECKBOX_MAPPING.get(request.GET.get('kumpul', False))
    public = CHECKBOX_MAPPING.get(request.GET.get('public', False))
    keluarga = CHECKBOX_MAPPING.get(request.GET.get('keluarga', False))

    result = getPredict(susahnafas, demam, baker, sateng, hipertensi, lelah, jjln, contact, kumpul, public, keluarga)
    
    return render(request, 'result.html', {"result":result})