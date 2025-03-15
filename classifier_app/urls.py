from django.urls import path
from .import views

urlpatterns = [
     path('',views.index,name='index'),
     path('contact/',views.contact,name='contact'),
     path('topics-detail.html',views.topics_detail,name='topics'),
     path('classification.html',views.classification,name='classification'),
     path('unsuper.html',views.unsuper,name='unsuper'),
     path('reinforce.html',views.reinforce,name='unsuper'),
     path('supportvectormachine.html',views.supportvectormachine,name='supportvec'),
     path('randomforest.html',views.randomforest ,name='supportvec'),
     path('polynomialregression.html',views.polynomialregression,name='polynomialreg'),
     path('naivebase.html',views.naivebase,name='naivebase'),
     path('logisticregression.html',views.logisticregression,name='logisticreg'),
     path('linearregression.html',views.linearregression,name='linearreg'),
     path('knearest.html',views.knearest,name='knearest'),
     path('Decisiontrees.html',views.Decisiontrees,name='decisiontree'),
     path('regression.html',views.regression,name='regression'),
     path('Model.html',views.Model,name='model'),
     path('streamlit/', views.streamlit_view, name='streamlit'),
]
