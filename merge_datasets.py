import pandas as pd

# =========================
# 1. LOAD ENRON DATASET
# =========================
enron = pd.read_csv("dataset/enron_spam_data.csv")

print("Enron columns:", enron.columns)

# Combine Subject + Message
enron['text'] = enron['Subject'] + " " + enron['Message']

# Rename label column
enron = enron.rename(columns={'Spam/Ham': 'label'})

# Keep required columns
enron = enron[['label', 'text']]


# =========================
# 2. LOAD SPAMASSASSIN DATASET
# =========================
spamassassin = pd.read_csv("dataset/spam_assassin.csv")

print("SpamAssassin columns:", spamassassin.columns)

# Rename column
spamassassin = spamassassin.rename(columns={'target': 'label'})

# Keep required columns
spamassassin = spamassassin[['label', 'text']]


# =========================
# 3. LOAD SMS DATASET (spam.csv)
# =========================
sms = pd.read_csv("dataset/spam.csv", encoding='latin-1')

print("SMS columns:", sms.columns)

# Rename columns (VERY IMPORTANT)
sms = sms.rename(columns={
    'v1': 'label',
    'v2': 'text'
})

# Keep required columns
sms = sms[['label', 'text']]


# =========================
# STANDARDIZE LABELS
# =========================
def fix_labels(df):
    df['label'] = df['label'].astype(str).str.lower()

    df['label'] = df['label'].map({
        'spam': 1,
        'ham': 0,
        '1': 1,
        '0': 0
    })

    return df

enron = fix_labels(enron)
spamassassin = fix_labels(spamassassin)
sms = fix_labels(sms)


# =========================
# MERGE ALL DATASETS
# =========================
df = pd.concat([enron, spamassassin, sms], ignore_index=True)


# =========================
# CLEAN DATA
# =========================
df.dropna(inplace=True)
df.drop_duplicates(inplace=True)

# Shuffle dataset
df = df.sample(frac=1, random_state=42)


# =========================
# SAVE FINAL DATASET
# =========================
df.to_csv("dataset/final.csv", index=False)

print("✅ Datasets merged successfully!")
print("Final shape:", df.shape)
print("\nLabel distribution:")
print(df['label'].value_counts())