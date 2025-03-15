from django.shortcuts import render

def index(request):
    return render(request,'classifier_app/index.html')
def contact(request):
    return render(request, 'classifier_app/contacts.html') 
def topics_detail(request):
    return render(request, 'classifier_app/topics-detail.html') 
def classification(request):
    return render(request, 'classifier_app/classification.html')
def unsuper(request):
    return render(request, 'classifier_app/unsuper.html')
def naivebase(request):
    return render(request, 'classifier_app/naivebase.html')
def polynomialregression(request):
    return render(request, 'classifier_app/polynomialregression.html')
def logisticregression(request):
    return render(request, 'classifier_app/logisticregression.html')
def linearregression(request):
    return render(request, 'classifier_app/linearregression.html')
def knearest(request):
    return render(request, 'classifier_app/knearest.html')
def Decisiontrees(request):
    return render(request, 'classifier_app/Decisiontrees.html')

def supportvectormachine(request):
    return render(request, 'classifier_app/supportvectormachine.html')
def Model(request):
    return render(request, 'classifier_app/Model.html')
def reinforce(request):
    return render(request, 'classifier_app/reinforce.html')
def regression(request):
    return render(request, 'classifier_app/regression.html') 
def randomforest(request):
    return render(request, 'classifier_app/randomforest.html') 
def streamlit_view(request):
    return render(request, 'streamlit.html')
