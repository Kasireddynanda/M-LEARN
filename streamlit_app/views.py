import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import base64
from io import BytesIO
from django.http import JsonResponse
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB


try:
    from xgboost import XGBClassifier
    xgboost_available = True
except ImportError:
    xgboost_available = False

# Function to select classifier
from django.shortcuts import render
def upload_view(request):
    return render(request, 'streamlitapp/upload.html')
def get_classifier(name):
    classifiers = {
        "Logistic Regression": LogisticRegression(),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "Random Forest": RandomForestClassifier(n_estimators=100),
        "Decision Tree": DecisionTreeClassifier(),
        "SVM": SVC(probability=True),
        "Gradient Boosting": GradientBoostingClassifier(),
        "Naive Bayes": GaussianNB()
    }
    
    if xgboost_available:
        classifiers["XGBoost"] = XGBClassifier()
    
    return classifiers.get(name, LogisticRegression())  # Default classifier

# Upload Dataset View
def upload_dataset(request):
    if request.method == 'POST' and request.FILES.get('dataset'):
        file = request.FILES['dataset']
        dataset = pd.read_csv(file)
        request.session['dataset_csv'] = dataset.to_csv(index=False)
        request.session['columns'] = list(dataset.columns)
        return render(request, 'upload.html', {'columns': dataset.columns})
    return render(request, 'upload.html')

# Train Model View
def train_model(request):
    if 'dataset_csv' not in request.session:
        return JsonResponse({'error': 'No dataset uploaded'}, status=400)
    
    dataset_csv = request.session['dataset_csv']
    from io import StringIO
    dataset = pd.read_csv(StringIO(dataset_csv))
    
    target_column = request.POST.get('target_column')
    if target_column not in dataset.columns:
        return JsonResponse({'error': 'Invalid target column'}, status=400)

    X = dataset.drop(columns=[target_column])
    y = dataset[target_column]
    
    if y.dtype == 'O':  
        y = pd.factorize(y)[0]
    
    classifier_name = request.POST.get('classifier')
    clf = get_classifier(classifier_name)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    
    cm = confusion_matrix(y_test, y_pred)
    fig1, ax1 = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="coolwarm", ax=ax1)
    ax1.set_xlabel("Predicted Label")
    ax1.set_ylabel("True Label")
    ax1.set_title(f"Confusion Matrix - {classifier_name}")
    
    buffer1 = BytesIO()
    fig1.savefig(buffer1, format="png")
    buffer1.seek(0)
    heatmap_base64 = base64.b64encode(buffer1.getvalue()).decode("utf-8")
    buffer1.close()
    plt.close(fig1)
    
    if X.shape[1] > 1:
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)
        
        fig2, ax2 = plt.subplots(figsize=(6, 5))
        scatter = ax2.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', alpha=0.6)
        ax2.set_xlabel("PCA Component 1")
        ax2.set_ylabel("PCA Component 2")
        ax2.set_title("PCA Dot Plot")
        plt.colorbar(scatter, ax=ax2)
        
        buffer2 = BytesIO()
        fig2.savefig(buffer2, format="png")
        buffer2.seek(0)
        dotplot_base64 = base64.b64encode(buffer2.getvalue()).decode("utf-8")
        buffer2.close()
        plt.close(fig2)
    else:
        dotplot_base64 = None
    
    return JsonResponse({
        'classifier': classifier_name,
        'accuracy': round(acc, 2),
        'heatmap': heatmap_base64,
        'dotplot': dotplot_base64
    })

