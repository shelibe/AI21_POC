import datetime
import json
import os
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import spacy
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split

from prod_data_analysis import get_prod_recommendations, get_prod_occurrences, create_validation_sample_from_DB

nlp = spacy.load("en_core_web_sm", disable=["tok2vec", "tagger", "parser", "attribute_ruler", "lemmatizer", "ner"])
nlp.enable_pipe("senter")

labels_maps = dict(Resistance="Care Resistance",
                   Improvement="Physical improvement",
                   Urinary="Urinary System", Neutral="neutral",
                   Vital="Vital Signs",
                   Emergency="EMS", Psychological="Psychological", Senses="Senses", Medication="Medication",
                   Hospital="Hospital",
                   Activity="ADLs", Exercise="Exercise", Positive="Positive Interaction", Cardiac="Cardiac",
                   Inadequate="Inadequate care", Plan="Care Plan", Gastrointestinal="Gastrointestinal",
                   Cognitive="Cognitive",
                   Sleep="Sleep", Fall="Fall", Anomaly="Physical Anomaly", Skin="Skin", Safety="Safety",
                   Equipment="Equipment",
                   Chronic="Chronic Condition", Pain="Pain", Labs="Labs", Respiratory="Respiratory",
                   help="Call for Help", TV="TV")

# Dataset input files
file_path = './data/'
annotation_input_csv_file = './data/input/label_redefenition_ai21.csv'
AWS_transcriptions_input_validation_csv_file = './data/input/AWS_Transcriptions_validation_DB.csv'

# AI21 run config
API_KEY = os.environ.get('AI21_API_KEY')
NUM_RESULTS = 1
TEMPERATURE = 0


def create_dataset(csv_file):
    df = pd.read_csv(csv_file)
    df = df.drop_duplicates()
    df = df.dropna()
    df['text'] = df['text'].apply(lambda x: x.lower())
    df['context'] = df['context'].apply(lambda x: x.lower())
    list_of_labels = ', '.join(labels_maps.keys())
    df['prompt'] = df.apply(
        lambda x: 'Classify this sentence using the provided context to one of the following classes: ['
                  + list_of_labels +
                  ']\nContext: ' + x['context'] +
                  '.\nSentence: ' + x['text'] + '.\nClass: \n', axis=1)
    return df


def prepare_train_test_datasets(df):
    df['completion'] = df['label'].apply(lambda x: find_key(labels_maps, x))
    X_train, X_test, y_train, y_test = train_test_split(df['prompt'], df['completion'],
                                                        stratify=df['completion'],
                                                        shuffle=True,
                                                        test_size=0.25)
    training_set = pd.concat([X_train, y_train], axis=1)
    test_set = pd.concat([X_test, y_test], axis=1)

    # dataset statistics
    print("Train Dataset statistics per class: \n")
    print(calculate_dataset_stas(training_set))
    print("Validation Dataset statistics per class: \n")
    print(calculate_dataset_stas(test_set))
    train_generated_dataset_csv_file = './data/generated_datasets/train_ai21_' + time.strftime('%Y%m%d%H%M%S') + '.csv'

    training_set.to_csv(train_generated_dataset_csv_file, index=False, columns=['prompt', 'completion'], header=True,
                        sep=',')
    validate_generated_dataset_csv_file = './data/generated_datasets/validation_ai21_' + time.strftime(
        '%Y%m%d%H%M%S') + '.csv'
    test_set.to_csv(validate_generated_dataset_csv_file, index=False, columns=['prompt', 'completion'], header=True,
                    sep=',')


def create_train_test_datasets(csv_file):
    prepare_train_test_datasets(create_dataset(csv_file))


def calculate_dataset_stas(dataset):
    print(dataset.groupby("completion")["completion"].count())


def predict(validate_dataset, prediction_output_csv_file, ground_truth=True):
    df = pd.read_csv(validate_dataset)
    print("Start prediction: {}".format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    df['y_pred'], df['y_pred_prop'] = zip(*df['prompt'].apply(lambda x: call_ai21_api(x)))
    print("Prediction completed: {}. \nStore predictions to CSV file: \n".format(
        datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    if ground_truth:
        predictions = pd.concat([df['prompt'], df['completion'], df['y_pred'], df['y_pred_prop']], axis=1)
        columns_list = ['prompt', 'completion', 'y_pred', 'y_pred_prop']
        calculate_performance_metrics(df['completion'], df['y_pred'])
    else:
        predictions = pd.concat([df['uuid'], df['prompt'], df['y_pred'], df['y_pred_prop']], axis=1)
        columns_list = ['uuid', 'prompt', 'y_pred', 'y_pred_prop']
    if not prediction_output_csv_file:
        prediction_output_csv_file = f"{file_path}/output_prediction/{asr_provider}_'ai21_predict'_{start_date}_{end_date}_{limit}_{time.strftime('%Y%m%d%H%M%S')}.csv "
        predictions.to_csv(prediction_output_csv_file, index=False, columns=columns_list,
                           header=True, sep=',')
    print("prediction file name: {}".format(prediction_output_csv_file))
    return prediction_output_csv_file


def call_ai21_api(text):
    res = None
    try:
        res = requests.post("https://api.ai21.com/studio/v1/j1-grande/Sensi_jumbo/complete",
                            headers={"Authorization": "Bearer {}".format(API_KEY)},
                            json={
                                "prompt": text,
                                "numResults": NUM_RESULTS,
                                "temperature": TEMPERATURE, })
    except requests.exceptions.RequestException as e:
        print(str(e))
    return handle_response(res)


def handle_response(res):
    if not res:
        print('response is empty')
        return 'error', 0
    try:
        res_json = json.loads(res.text)['completions'][0]
        predicted_class = res_json['data']['text']
        if (predicted_class):
            print(predicted_class)
            raw_logprob = res_json['data']['tokens'][0]['generatedToken']['raw_logprob']
            raw_probability = np.power(10, raw_logprob)
            return predicted_class, raw_probability
        else:
            return None, None
    except ValueError:
        print('Failed extracting response')


def calculate_performance_metrics(y_true, y_pred):
    print("Calculate prediction metrics: \n")
    print(classification_report(y_true, y_pred))
    print("Confusion matrix: \n")
    labels = list(set(np.unique(y_true)).union(set(np.unique(y_pred))))
    confusion_mtx = confusion_matrix(y_true, y_pred, normalize='true', labels=labels)
    confusion_mtx_display = ConfusionMatrixDisplay(confusion_matrix=confusion_mtx,
                                                   display_labels=labels)
    confusion_mtx_display.plot()
    plt.show()
    print(confusion_mtx)


def find_key(input_dict, value):
    for key, val in input_dict.items():
        if val == value: return key
    return "None"


def load_DB_validation_dataset_from_CSV(transcriptions_validation_DB, DB_dataset):
    df = pd.read_csv(transcriptions_validation_DB)
    prepare_DB_validate_dataset_CSV(df, DB_dataset)


def prepare_DB_validate_dataset_CSV(df, DB_dataset):
    df['text'] = df['context'].apply(lambda x: context_to_sentences(x))
    df = df.explode('text')
    db_dataset = pd.concat([df['uuid'], df['text'], df['context']], axis=1)
    db_dataset.to_csv(DB_dataset, index=False, columns=['uuid', 'text', 'context'], header=True, sep=',')
    df_db = create_dataset(DB_dataset)
    df_db.to_csv(DB_dataset, index=False, columns=['uuid', 'prompt'], header=True, sep=',')


def context_to_sentences(context):
    doc = nlp(context)
    return list(doc.sents)


def analyze_prediction_results(prediction_file, classes, start_date, end_date, asr_provider, limit):
    df = pd.read_csv(prediction_file)
    df['y_pred'] = df['y_pred'].apply(lambda x: x if x in labels_maps.keys() else 'Neutral')
    ai21_predict_df = pd.concat([df['uuid'], df['y_pred'], df['y_pred_prop']], axis=1)
    print("ai21 recommendations distribution per class:")
    print("############################################")
    print(ai21_predict_df.groupby("y_pred")["y_pred"].count())
    ai21_rec_no_neutral_sentence_level_df = df[df['y_pred'] != 'Neutral']
    ai21_rec_no_neutral_context_level_df = ai21_rec_no_neutral_sentence_level_df.groupby('uuid')['uuid']
    prod_rec_df = get_prod_recommendations(start_date, end_date, asr_provider, limit)
    if prod_rec_df is not None:
        prod_actual_rec_df = prod_rec_df[~prod_rec_df['event_type_id'].isin([116, 121, 122])]
        if not prod_actual_rec_df.empty:
            print("\nProduction recommendations distribution per class:")
            print("############################################")
            print(prod_actual_rec_df.groupby("event_type_id")["event_type_id"].count())
    analyze_recommendations(len(df), ai21_rec_no_neutral_sentence_level_df, ai21_rec_no_neutral_context_level_df,
                            prod_actual_rec_df)
    analyze_occurrences(prod_actual_rec_df, ai21_rec_no_neutral_sentence_level_df)
    print("{} Recommendations:".format(''.join(classes)))
    print("############################################")
    print_label_data(df, classes)


def print_label_data(df, classes):
    df['check'] = df['y_pred'].apply(lambda x: True if x in classes else False)
    class_df = df[df['check'] == True]['uuid']
    if class_df.any():
        print(class_df)
    else:
        print("None")


def print_entity(message, enteties, enteties_len):
    print("\n" + message + " {}: ".format(enteties_len))
    if enteties_len != 0:
        print("Job uuids {}: ".format(enteties))


def analyze_occurrences(prod_actual_rec_df, ai21_predict_rec_df):
    print("\nOccurrences:")
    print("############################################")
    prod_occurrences_df = get_prod_occurrences(start_date, end_date, asr_provider, limit)
    if prod_occurrences_df is not None:
        merged = pd.merge(prod_occurrences_df, prod_actual_rec_df, on='uuid', how='outer', indicator=True)
        occ_no_ai21_rec = merged[merged['_merge'] == 'left_only']
        occ_no_ai21_rec_len = len(occ_no_ai21_rec)
        print_entity("Occurrences without ai21 recommendations", occ_no_ai21_rec, occ_no_ai21_rec_len)
        occ_with_ai21_rec = pd.merge(prod_occurrences_df, ai21_predict_rec_df, on='uuid', how='inner')
        occ_with_ai21_rec_len = len(occ_with_ai21_rec)
        print_entity("Occurrences with ai21 recommendations", occ_with_ai21_rec, occ_with_ai21_rec_len)
    else:
        print("No occurrences detected in production\n")


def analyze_recommendations(df_len, ai21_predict_rec_df, ai21_rec_no_neutral_context_level_df, prod_actual_occ_df):
    print("\nRecommendations:")
    print("############################################")
    print("ai21 Recommendations - sentence level: {} , sentence level - without Neutral: {}, job level: {}".format(
        df_len, len(ai21_predict_rec_df), len(ai21_rec_no_neutral_context_level_df)))

    merged = pd.merge(prod_actual_occ_df, ai21_predict_rec_df, on='uuid', how='outer', indicator=True)
    rec_only_prod = merged[merged['_merge'] == 'left_only']
    rec_only_prod_len = len(rec_only_prod)
    print_entity("Recommendations detected only by prod", rec_only_prod, rec_only_prod_len)
    rec_only_ai21 = merged[merged['_merge'] == 'right_only']
    rec_only_ai21_len = len(rec_only_ai21)
    print_entity("Recommendations detected only by ai21", rec_only_ai21, rec_only_ai21_len)
    rec_both = pd.merge(prod_actual_occ_df, ai21_predict_rec_df, on='uuid', how='inner')
    rec_both_len = len(rec_both)
    print_entity("Recommendations detected by both", rec_both, rec_both_len)


def prod_sample_analyzer(DB_validation_sample=None, validation_sample_csv_file=None, run_predict=False, analyzer=True,
                         start_date=None, end_date=None, prediction_output_csv_file=None,
                         asr_provider='AWS', limit='1', label='help'):
    # Optional: Creates validation dataset from DB either by fetching data from DB or loading CSV file
    transcription_generated_input_csv_file = f"{file_path}/generated_datasets/{asr_provider}_'DB_validation_dataset'_{start_date}_{end_date}_{limit}_{time.strftime('%Y%m%d%H%M%S')}.csv"
    if DB_validation_sample == 'DB':
        validation_sample_df = create_validation_sample_from_DB(start_date, end_date,
                                                                asr_provider, limit)
        prepare_DB_validate_dataset_CSV(validation_sample_df, transcription_generated_input_csv_file)
    if DB_validation_sample == 'File':
        load_DB_validation_dataset_from_CSV(validation_sample_csv_file,
                                            transcription_generated_input_csv_file)
    # Optional: Predict
    if run_predict:
        prediction_output_csv_file = predict(transcription_generated_input_csv_file, prediction_output_csv_file,
                                             ground_truth=False)
    # Optional: Analyze results
    if analyzer:
        analyze_prediction_results(prediction_output_csv_file, label, start_date,
                                   end_date, asr_provider, limit)


if __name__ == '__main__':
    ############ Train/Validate AI21###############
    #create_train_test_datasets(annotation_input_csv_file)
    # predict(validate_generated_dataset_csv_file, prediction_output_csv_file)
    ############ DB compare #######################
    # predict_file_name = aws_prediction_output_csv_file
    # file_path = os.path.dirname(predict_file_name) + '/'
    # file_basename = 'ai21_predict'
    # file_extension = os.path.splitext(predict_file_name)[1]
    start_date = datetime.date(2023, 2, 19)
    end_date = datetime.date(2023, 2, 20)
    asr_provider = 'AWS'
    limit = '2'

    prod_sample_analyzer(DB_validation_sample='DB', run_predict=True,
                         analyzer=True, start_date=start_date, end_date=end_date, asr_provider=asr_provider,
                         limit=limit, label='help', prediction_output_csv_file=None)
##Only analyze - Need to provide prediction file
# prod_sample_analyzer(DB_validation_sample=None, run_predict=False,
#                      analyzer=True, start_date=start_date, end_date=end_date, asr_provider=asr_provider,
#                      limit=limit, label='help',
#                      prediction_output_csv_file='./data/output_prediction/AWS_ai21_predict_2023-02-19_2023-02-20_10_20230303031713.csv')
