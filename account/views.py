from django.shortcuts import render,redirect
from django.contrib.auth.models import User
from django.contrib.auth import authenticate,login,logout
from .models import Contact
import re
import pandas as pd
from urllib.parse import urlparse
from tensorflow.keras.models import load_model # type: ignore
from sklearn.preprocessing import StandardScaler
import joblib
from django.contrib.auth import get_user_model
import os
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from datetime import datetime
from django.utils.timezone import now
from django.conf import settings
from django.core.files.storage import FileSystemStorage
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from tensorflow.keras.layers import (Input, Conv1D, Dense, Flatten, Dropout, Multiply,Softmax, Reshape, TimeDistributed, BatchNormalization) # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
from tensorflow.keras.models import Model # type: ignore

def signup(request):
    if request.method == "POST":
        username = request.POST.get("username")
        email = request.POST.get("email")
        password = request.POST.get("password")
        confirm_password = request.POST.get("confirm_password")
        if password != confirm_password:
            return render(request,"signup.html",{"error":"password not matching!"})
        user = User.objects.create_user(username = username,email = email,password = password)
        return redirect("login")
    return render(request,"signup.html")


def admin_login(request):
    if request.method == 'POST':
        username = request.POST['username']
        password = request.POST['password']
        user = authenticate(request, username=username, password=password)
        if user is not None and user.is_superuser:  # check if admin
            login(request, user)
            return redirect('admin_home')
        else:
            return render(request, 'admin_login.html', {'error': 'Invalid admin credentials'})
    return render(request, 'admin_login.html')

User = get_user_model()
import io
import base64
from matplotlib.figure import Figure
from django.utils.timezone import now, localdate
from .models import UrlDataset, ModelAccuracy, Prediction

def generate_base64_plot(fig):
    buffer = io.BytesIO()
    fig.savefig(buffer, format='png')
    buffer.seek(0)
    return base64.b64encode(buffer.getvalue()).decode('utf-8')

def admin_home(request):
    # Summary data
    total_users = User.objects.count()
    today = localdate()
    predictions_today = Prediction.objects.filter(created_at__date=today).count()
    phishing_today = Prediction.objects.filter(created_at__date=today, prediction_result='phishing').count()

    latest_accuracy = ModelAccuracy.objects.last()
    accuracy_score = f"{latest_accuracy.accuracy:.2f}%" if latest_accuracy else "N/A"

    # Chart 1: Model Accuracy
    from collections import OrderedDict

    # Get all entries ordered by latest first
    accs_all = ModelAccuracy.objects.all().order_by('-trained_on')

    # Keep only latest accuracy per model (preserve insertion order)
    latest_accs = OrderedDict()
    for acc in accs_all:
        if acc.model_name not in latest_accs:
            latest_accs[acc.model_name] = acc

    accs = list(latest_accs.values())

    names = [m.model_name for m in accs]
    values = [m.accuracy for m in accs]

    fig1 = Figure()
    ax1 = fig1.subplots()

    model_colors = {
        'Random Forest': 'orange',
        'CNN': 'skyblue',
        'ABS-CNN': 'seagreen',
    }
    colors = [model_colors.get(m.model_name, 'gray') for m in accs]
    bars = ax1.bar(names, values, color=colors)

    ax1.set_title('Model Accuracy Comparison')
    ax1.set_ylabel('Accuracy (%)')

    # Add values on top of bars
    for bar, acc in zip(bars, values):
        ax1.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 1.5,
                 f"{acc:.1f}%",
                 ha='center',
                 va='bottom',
                 fontsize=10,
                 color='black')

    fig1.tight_layout()
    accuracy_chart = generate_base64_plot(fig1)


    # Chart 2: Prediction Results
    from collections import defaultdict
    from django.db.models.functions import TruncDate
    from django.db.models import Count
    from .models import Contact 

    stats = (
    Prediction.objects
    .annotate(date=TruncDate('created_at'))
    .values('date', 'prediction_result')
    .annotate(count=Count('id'))
    .order_by('date')
)

    data = defaultdict(lambda: {'phishing': 0, 'legitimate': 0})
    for row in stats:
        date = row['date']
        label = row['prediction_result']
        data[date][label] = row['count']

    dates = sorted(data.keys())
    phishing = [data[d]['phishing'] for d in dates]
    legit = [data[d]['legitimate'] for d in dates]

    fig2 = Figure(figsize=(8, 4))
    ax2 = fig2.subplots()
    ax2.plot(dates, phishing, marker='o', label='Phishing', color='red')
    ax2.plot(dates, legit, marker='o', label='Legitimate', color='green')
    # Add value labels to line chart points
    for i, date in enumerate(dates):
        ax2.text(date, phishing[i] - 0.2, str(phishing[i]), ha='center', color='red', fontsize=9)
        ax2.text(date, legit[i] - 0.2, str(legit[i]), ha='center', color='green', fontsize=9)

    ax2.set_title('Prediction Results Over Time')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Count')
    ax2.legend()
    fig2.autofmt_xdate()
    fig2.tight_layout()
    prediction_chart = generate_base64_plot(fig2)

    # Get users and dataset
    users = User.objects.all().order_by('-date_joined')
    dataset = UrlDataset.objects.all().order_by('-uploaded_at')[:100]
    activities = UserActivity.objects.all().order_by('-timestamp')
    contacts = Contact.objects.all().order_by('-id')

    return render(request, 'admin_home.html', {
        'users': users,
        'dataset': dataset,
        'model_accuracy': accuracy_score,
        'total_users': total_users,
        'predictions_today': predictions_today,
        'phishing_today': phishing_today,
        'accuracy_chart': accuracy_chart,
        'prediction_chart': prediction_chart,
        'activities': activities,
        'contacts': contacts,
    })


from .models import UserActivity
from django.utils.timezone import now

def get_client_ip(request):
    x_forwarded_for = request.META.get('HTTP_X_FORWARDED_FOR')
    if x_forwarded_for:
        ip = x_forwarded_for.split(',')[0]
    else:
        ip = request.META.get('REMOTE_ADDR')
    return ip

def login_page(request):
    if request.method == "POST":
        username = request.POST.get("username")
        email = request.POST.get("email")
        password = request.POST.get("password")
        user = authenticate(request, username=username, email=email, password=password)
        if user is not None:
            login(request, user)
            UserActivity.objects.create(user=user, action='login', ip_address=get_client_ip(request))
            return redirect("project_home")
        return render(request,"login.html",{"error":"user not found"})
    return render(request,"login.html")

def logout_page(request):
    if request.user.is_authenticated:
        # Optionally: Log the logout event
        from .models import UserActivity
        from .views import get_client_ip
        UserActivity.objects.create(user=request.user, action='logout', ip_address=get_client_ip(request))

        is_admin = request.user.is_superuser
        logout(request)
        
        if is_admin:
            return redirect('admin_login')  # name of your admin login url
        else:
            return redirect('login')  # normal user login
    return redirect('login')

from .models import UserActivity
from django.contrib.auth.models import User

def login_info(request):
    activities = UserActivity.objects.select_related('user').order_by('-timestamp')
    users = User.objects.all()

    if request.method == 'POST':
        if 'delete_user' in request.POST:
            user_id = request.POST.get('delete_user')
            User.objects.filter(id=user_id).delete()

        if 'deactivate_user' in request.POST:
            user_id = request.POST.get('deactivate_user')
            user = User.objects.get(id=user_id)
            user.is_active = False
            user.save()

        if 'activate_user' in request.POST:
            user_id = request.POST.get('activate_user')
            user = User.objects.get(id=user_id)
            user.is_active = True
            user.save()

        return redirect('admin_home')  # reload page

    return render(request, 'admin_home.html', {'activities': activities, 'users': users})

def home(request):
    return render(request,"home.html")
def contactus(request):
    return render(request,"contactus.html")
def dataset(request):
    return render(request,"dataset.html")
from django.contrib import messages
def contactus_form(request):
    if request.method == "POST":
        name = request.POST.get("name")
        email = request.POST.get("email")
        message = request.POST.get("message")
        Contact.objects.create(
            name = name,
            email = email,
            message = message,
        )
        messages.success(request, "Your message has been sent successfully!")
        return redirect("contactus")
    return render(request,"contactus.html")

def project_home(request):
    if not request.user.is_authenticated:
        return redirect("login")
    result = None
    if request.method == 'POST':
        url = request.POST.get('url')
        result = predict_single_url(url)
        # Store prediction in the database
        if request.user.is_authenticated:
            confidence = float(re.findall(r"([0-9.]+)%", result)[0]) / 100
            label = 'phishing' if 'Phishing' in result else 'legitimate'
            from .models import Prediction
            Prediction.objects.create(
                user=request.user,
                url_input=url,
                prediction_result=label,
                confidence=confidence,
                created_at=now()
            )

            #logging the activity
            from .models import UserActivity
            UserActivity.objects.create(
                user=request.user,
                action=f"Predicted {label} for {url}",
                ip_address=get_client_ip(request)
            )
    return render(request, 'project_home.html', {'result': result})

# function to ectract the important features from the url 
def extract_advanced_features(url):
    parsed = urlparse(url)
    domain = parsed.netloc
    path = parsed.path
    query = parsed.query

    shortening_services = ['bit.ly', 'tinyurl.com', 'goo.gl', 't.co', 'ow.ly', 'is.gd']
    domain_keywords = domain.split('.')

    return {
        'url_length': len(url),
        'special_chars': len(re.findall(r"[/?=&]", url)),
        'suspicious_keywords': sum(kw in url.lower() for kw in ['login', 'signin', 'account', 'verify', 'secure']),
        'has_https': 1 if parsed.scheme == 'https' else 0,
        'num_dots': url.count('.'),
        'num_digits': len(re.findall(r"\\d", url)),
        'domain_length': len(domain),
        'uses_ip': 1 if re.match(r'^\\d{1,3}(\\.\\d{1,3}){3}$', domain) else 0,
        'has_at_symbol': 1 if '@' in url else 0,
        'has_redirect': 1 if '//' in path else 0,
        'num_subdomains': domain.count('.') - 1,
        'suspicious_tld': 1 if any(domain.endswith(tld) for tld in ['.zip', '.tk', '.ml', '.ga', '.cf']) else 0,
        'has_http_in_domain': 1 if 'http' in domain else 0,
        'shortening_service': 1 if any(svc in domain for svc in shortening_services) else 0,
        'domain_in_path': 1 if any(dk in path for dk in domain_keywords) else 0,
        'num_parameters': len(query.split('&')) if query else 0,
        'prefix_suffix': 1 if '-' in domain else 0,
        'page_rank': 5  
    }
# predicts and returns the output 
def predict_single_url(url):
    model = joblib.load(r'C:\Users\Admin\OneDrive\Desktop\major_project_demo\detection_model.pkl')
    scaler = joblib.load(r'C:\Users\Admin\OneDrive\Desktop\major_project_demo\scaler.pkl')
    features = extract_advanced_features(url)
    df = pd.DataFrame([features])
    X = scaler.transform(df)
    X = X.reshape(1, X.shape[1], 1)
    prediction = model.predict(X)[0][0]
    if prediction >= 0.5:
        return f"Phishing Website ({prediction * 100:.2f}% confidence)"
    else:
        return f"Legitimate Website ({(1 - prediction) * 100:.2f}% confidence)"
    
#code for retrainig and uploading the dataset
def retrain_model():
    queryset = UrlDataset.objects.all()
    df = pd.DataFrame(list(queryset.values('website', 'status', 'page_rank')))
    df['status'] = df['status'].apply(lambda x: 1 if x == 'phishing' else 0)
    
    feature_data = df['website'].apply(lambda x: pd.Series(extract_advanced_features(x)))
    feature_data['page_rank'] = df['page_rank']
    
    X = feature_data
    y = df['status'].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1], 1)

    joblib.dump(scaler, 'scaler.pkl')

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weights = {0: weights[0], 1: weights[1]}

    input_layer = Input(shape=(X_scaled.shape[1], 1))
    x = Conv1D(64, kernel_size=2, activation='relu', padding='same')(input_layer)
    x = BatchNormalization()(x)
    x = Conv1D(64, kernel_size=2, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    attention = TimeDistributed(Dense(1, activation='tanh'))(x)
    attention = Flatten()(attention)
    attention = Softmax()(attention)
    attention = Reshape((X_scaled.shape[1], 1))(attention)
    attention_mul = Multiply()([x, attention])
    x = Flatten()(attention_mul)
    x = Dropout(0.3)(x)
    x = Dense(64, activation='relu')(x)
    output = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=input_layer, outputs=output)
    model.compile(optimizer=Adam(0.001), loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=40, batch_size=32, validation_split=0.2, class_weight=class_weights)
    loss, accuracy = model.evaluate(X_test, y_test)
    ModelAccuracy.objects.create(model_name="ABS-CNN", accuracy=accuracy * 100, trained_on=now())
def upload_dataset(request):
    if request.method == 'POST' and request.FILES['csv_file']:
        csv_file = request.FILES['csv_file']
        df = pd.read_csv(csv_file)
        for _, row in df.iterrows():
            UrlDataset.objects.create(
                website=row['website'],
                status=row['status'],
                page_rank=row['page_rank']
            )
        retrain_model()
        return redirect('admin_home')



