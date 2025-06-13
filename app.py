import streamlit as st
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from PIL import Image




st.title('Welcome to wild animals')

image = Image.open('data/dataset-cover.png')

st.image(image)

# st.set_page_config(
#         layout="wide",
#         initial_sidebar_state="auto",
#         page_title="Welcome to wild animals",
#         page_icon=image,

#     )


st.markdown('##### You have met an animal, we will determine its behavior')

# df = pd.read_csv()

def split_data(df: pd.DataFrame):
    y = df['Adaptation_Signal']
    X = df.drop(columns=['Adaptation_Signal'])
    return X, y


def open_data(path="data/urban_wildlife_adaptation_english.csv"):
    df = pd.read_csv(path)
    df['Species'] = df['Species'].apply(lambda x:
                                    1 if x == 'Güvercin'
                                    else 2 if x == 'Rakun'
                                    else 3 if x == 'Sincap'
                                    else 4)
    df['Observation_Time'] = df['Observation_Time'].apply(lambda x:
                                    1 if x == 'Akşam'
                                    else 2 if x == 'Öğleden Sonra'
                                    else 3 if x == 'Gece'
                                    else 4)
    df['Location_Type'] = df['Location_Type'].apply(lambda x:
                                    1 if x == 'Ticari'
                                    else 2 if x == 'Park'
                                    else 3 if x == 'Endüstriyel'
                                    else 4)
    df['Adaptation_Signal'] = df['Adaptation_Signal'].apply(lambda x:
                                    1 if x == 'Alışkanlık'
                                    else 2 if x == 'Kaçınma'
                                    else 3 if x == 'Sömürü'
                                    else 4)  
    df = df.drop('Animal_ID', axis=1)

    return df

def normalize_data(df: pd.DataFrame, test=True):
    df_norm = df.copy()

    cols_to_normalize = ['Noise_Level_dB', 'Human_Density',
                        'Food_Source_Score', 'Shelter_Quality_Score',
                        'Estimated_Daily_Distance_km']

    # Нормализация
    scaler = MinMaxScaler()
    df_norm[cols_to_normalize] = scaler.fit_transform(df_norm[cols_to_normalize])

    if test:
        X_df, y_df = split_data(df_norm)
    else:
        X_df = df

    if test:
        return X_df, y_df
    else:
        return X_df



df = open_data()

# st.write(df.head())

X_df, y_df = normalize_data(df, test=True)

# st.write(X_df)

model_ovr = LogisticRegression(multi_class='ovr')
model_ovr.fit(X_df, y_df)

# test_prediction = model_ovr.predict(X_df)
# accuracy = accuracy_score(test_prediction, y_df)





# # accuracy_score(y_test, pred_ovr)

def sidebar_input_features():
    with st.sidebar:
        st.header('Under what circumstances did you encounter an animal')
        who = st.selectbox(
            'What kind of animal did you encounter',
            ['Raccoon', 'Pigeon', 'Fox', 'Squirrel']
        )
        when = st.selectbox(
            'When did this happen',
            ['Morning', 'Afternoon', 'Evening', 'Night']
        )
        where = st.selectbox(
            'Where did this happen',
            ['Commercial', 'Park', 'Industrial', 'Residential']
        )
        noise = st.slider('Noise Level (dB)', min_value=0, max_value=100, value=0, step=1)
        density = st.slider('Human Density', min_value=0, max_value=100, value=0, step=1)

        food = st.radio(
            'Food Source Score',
            ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10'],
            horizontal=True
        )
        shelter_quality = st.radio(
            'Shelter Quality Score',
            ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10'],
            horizontal=True
        )
        behavior_anomaly = st.radio(
            'Behavior Anomaly Score',
            ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10'],
            horizontal=True
        )
        distance = st.number_input('Estimated Daily Distance (km)', min_value=0.0, step=0.1)

        translatetion = {
            "Evening": 1,
            "Afternoon": 2,
            "Night": 3,
            "Morning": 4,

            "Commercial": 1,
            "Park": 2,
            "Industrial": 3,
            "Residential": 4,

            "Pigeon": 1,
            "Raccoon": 2,
            "Squirrel": 3,
            "Fox": 4,
        }

        data = {
            "Species": translatetion[who],
            "Observation_Time": translatetion[when],
            "Location_Type": translatetion[where],
            "Noise_Level_dB": int(noise),
            "Human_Density": int(density),
            "Food_Source_Score": int(food),
            "Shelter_Quality_Score": int(shelter_quality),
            "Behavior_Anomaly_Score": int(behavior_anomaly),
            "Estimated_Daily_Distance_km": float(distance),
        }

    df = pd.DataFrame(data, index=[0])
    return df



user_input_df = sidebar_input_features()

train_df = open_data()
train_X_df, _ = split_data(train_df)
full_X_df = pd.concat((user_input_df, train_X_df), axis=0)
norm_X_df = normalize_data(full_X_df, test=False)
user_X_df = norm_X_df[:1]


prediction = model_ovr.predict(user_X_df)[0]
prediction_proba = model_ovr.predict_proba(user_X_df)[0]

# encode_prediction_proba = {
#     1: 'Habituation',
#     2: 'Avoidance',
#     3: 'Exploitation',
#     4: 'Innovation'
# }

# prediction_data = {}
# for key, value in encode_prediction_proba.items():
#         prediction_data.update({value: prediction_proba[key-1]})

# st.write("## Предсказание")
# st.write(prediction)

# st.write("## Вероятность предсказания")
# st.write(prediction_data)


# Предсказываем вероятности
# Предсказываем вероятности
prediction_proba = model_ovr.predict_proba(user_X_df)[0]

# Словарь: номер класса -> название
class_names = {
    1: 'Habituation',
    2: 'Avoidance',
    3: 'Exploitation',
    4: 'Innovation'
}

# Получаем номер класса с максимальной вероятностью
predicted_class = int(model_ovr.predict(user_X_df)[0])

# Максимальная вероятность
max_prob = max(prediction_proba)

# Переводим вероятность в проценты
percentage = max_prob * 100

# Выводим результат

st.write('\n\n')

st.markdown("### Prediction")
st.markdown(f"#### **{class_names[predicted_class]}** — {percentage:.2f}%")