import pandas as pd
from surprise import Reader, Dataset, SVD
from surprise.model_selection import cross_validate
import mysql.connector

db = mysql.connector.connect(
    host="localhost",
    user="root",
    password="1234",
    database="mentors"
)

ratings = pd.read_csv('./extracted_data.csv')  # ratings 데이터를 미리 읽어옴

data = ratings

def train_model():
    reader = Reader(rating_scale=(1, 5))
    svd = SVD(random_state=0)
    df = Dataset.load_from_df(data, reader)
    cross_validate(svd, df, measures=['RMSE', 'MAE'], cv=5, verbose=True)
    trainset = df.build_full_trainset()
    svd.fit(trainset)
    return svd


model = train_model()  # 서버 시작 시 모델 학습 수행

def convert_to_number(word):
    number_dict = {'ONE': 1, 'TWO': 2, 'THREE': 3, 'FOUR': 4, 'FIVE': 5}
    return number_dict.get(word)

def training(reviewer_id):
    global model
    global data

    if not any(data['reviewer_id'].astype(str).str.contains(reviewer_id)):
        cursor = db.cursor()
        cursor.execute("SELECT reviewer_id, mentoring_id, rating FROM reviews where reviewer_id =" + reviewer_id)
        reviewers = cursor.fetchall()
        cursor.close()
        converted_reviewers = [(reviewer[0], reviewer[1], convert_to_number(reviewer[2])) for reviewer in reviewers]
        new_data_df = pd.DataFrame(converted_reviewers, columns=['reviewer_id', 'mentoring_id', 'rating'])
        data = pd.concat([data, new_data_df], axis=0)
        model = train_model()

def filter(reviewer_id):
    global model
    global data

    if not any(data['reviewer_id'].astype(str).str.contains(reviewer_id)):
        return []

    cursor = db.cursor()
    cursor.execute("SELECT distinct(mentoring_id) FROM reviews")
    mentoring_ids = cursor.fetchall()
    cursor.execute("SELECT mentoring_id FROM mypages where mentee_id = " +reviewer_id)
    mentees = cursor.fetchall()
    cursor.close()

    my_array = []
    for mentoringId in mentoring_ids:
        if mentoringId in mentees:
            continue
        my_array.append(model.predict(reviewer_id, mentoringId[0]))
    my_array.sort(key=lambda x: x.est, reverse=True)
    top_10_predictions = my_array[:10]
    result = []
    for prediction in top_10_predictions:
        result.append(prediction.iid)

    return result

