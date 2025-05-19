import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load Data
data = pd.read_csv('phikitha.csv')

# Preprocessing
data['status'] = data['status'].apply(lambda x: 1 if x == 'phishing' else 0)
def extract_url_features(url):
    import re
    url_length = len(url)
    num_special_chars = len(re.findall(r"[/?=&]", url))
    num_suspicious_keywords = sum(keyword in url.lower() for keyword in ["login", "signin", "account", "verify", "secure"])
    return url_length, num_special_chars, num_suspicious_keywords

# Applying the function to each URL
data[['url_length', 'special_chars', 'suspicious_keywords']] = data['website'].apply(lambda x: pd.Series(extract_url_features(x)))

# Dropping the original 'website' column
data.drop(columns=['website'], inplace=True)

# Normalizing the features
scaler = StandardScaler()
data[['page_rank', 'url_length', 'special_chars', 'suspicious_keywords']] = scaler.fit_transform(
    data[['page_rank', 'url_length', 'special_chars', 'suspicious_keywords']]
)
updated_dataset = data.to_csv("updated_dataset.csv")
# Splitting the data
X = data.drop(columns=['status'])
y = data['status']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Building and Training
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
test = pd.DataFrame([{"page_rank":2,"url_length":10,"special_chars":2,"suspicious_keywords":0,}])
test_scaled = scaler.transform(test)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
print(accuracy)
print(report)

# Save the Model
# joblib.dump(model, 'phishing_detection_model.pkl')
# # print(model.feature_importances_)
# if y_pred == 1:
#     print("phishing")
# else:
#     print("legitimate")



