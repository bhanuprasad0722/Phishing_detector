# ---------------------------
# STEP 1: Imports
# ---------------------------
import pandas as pd
import joblib
import numpy as np
import re
from urllib.parse import urlparse
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import class_weight
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Conv1D, Dense, Flatten, Dropout, Multiply,
                                     Softmax, Reshape, TimeDistributed, BatchNormalization)
from tensorflow.keras.optimizers import Adam

# ---------------------------
# STEP 2: Feature Engineering
# ---------------------------
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
        'num_digits': len(re.findall(r"\d", url)),
        'domain_length': len(domain),
        'uses_ip': 1 if re.match(r'^\d{1,3}(\.\d{1,3}){3}$', domain) else 0,
        'has_at_symbol': 1 if '@' in url else 0,
        'has_redirect': 1 if '//' in path else 0,
        'num_subdomains': domain.count('.') - 1,
        'suspicious_tld': 1 if any(domain.endswith(tld) for tld in ['.zip', '.tk', '.ml', '.ga', '.cf']) else 0,
        'has_http_in_domain': 1 if 'http' in domain else 0,
        'shortening_service': 1 if any(svc in domain for svc in shortening_services) else 0,
        'domain_in_path': 1 if any(dk in path for dk in domain_keywords) else 0,
        'num_parameters': len(query.split('&')) if query else 0,
        'prefix_suffix': 1 if '-' in domain else 0
    }

# ---------------------------
# STEP 3: Load and Prepare Data
# ---------------------------
df = pd.read_csv('phikitha.csv')
df['status'] = df['status'].apply(lambda x: 1 if x == 'phishing' else 0)

feature_data = df['website'].apply(lambda x: pd.Series(extract_advanced_features(x)))
df = pd.concat([df, feature_data], axis=1).drop(columns=['website'])

features = feature_data.columns.tolist() + ['page_rank']
scaler = StandardScaler()
X = scaler.fit_transform(df[features])
X = X.reshape(X.shape[0], X.shape[1], 1)
y = df['status'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Handle class imbalance
weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights = {0: weights[0], 1: weights[1]}

# ---------------------------
# STEP 4: Define ABS-CNN Model
# ---------------------------
input_layer = Input(shape=(X.shape[1], 1))

x = Conv1D(64, kernel_size=2, activation='relu', padding='same')(input_layer)
x = BatchNormalization()(x)
x = Conv1D(64, kernel_size=2, activation='relu', padding='same')(x)
x = BatchNormalization()(x)

attention = TimeDistributed(Dense(1, activation='tanh'))(x)
attention = Flatten()(attention)
attention = Softmax()(attention)
attention = Reshape((X.shape[1], 1))(attention)
attention_mul = Multiply()([x, attention])

x = Flatten()(attention_mul)
x = Dropout(0.3)(x)
x = Dense(64, activation='relu')(x)
output = Dense(1, activation='sigmoid')(x)

model = Model(inputs=input_layer, outputs=output)
model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# ---------------------------
# STEP 5: Train and Evaluate
# ---------------------------
model.fit(X_train, y_train, epochs=40, batch_size=32, validation_split=0.2, class_weight=class_weights)
loss, accuracy = model.evaluate(X_test, y_test)
print(f"🎯 Final Accuracy: {accuracy * 100:.2f}%")
# joblib.dump(model, 'detection_model.pkl')
# joblib.dump(scaler,'scaler.pkl')