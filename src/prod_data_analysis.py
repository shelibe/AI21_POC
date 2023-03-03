import os
import time
import pandas as pd
import psycopg2

HOST=os.environ.get('DB_HOST')
DB="postgres"
USER=os.environ.get('DB_USER')
PASSWORD=os.environ.get('DB_PASSWORD')
PORT=os.environ.get('DB_PORTD')
generated_files_path='./data/db_queries_res/'

def execute_query_create_datafram(query,query_params, df_columns,entity_name,asr_provider,start_date,end_date, limit):
    try:
        conn = psycopg2.connect(host=HOST,database=DB,user=USER,password=PASSWORD,port=PORT)
        cur = conn.cursor()
        cur.execute(query, query_params)
        results = cur.fetchall()
        df = pd.DataFrame(results, columns=df_columns)
        results_csv_file = f"{generated_files_path}{asr_provider}_{entity_name}_{start_date}_{end_date}_{limit}_{time.strftime('%Y%m%d%H%M%S')}.csv"

        df.to_csv(results_csv_file, index=False, columns=df_columns,
                              header=True, sep=',')
        cur.close()
        conn.close()
    except Exception as e:
        print(str(e))
        cur.close()
        conn.close()
    return df

def get_prod_recommendation_query(col, limit, with_str=False):
    with_query="WITH params (from_date,to_date,provider) as (values (%s,%s,%s)) "
    prod_recommendations_query = "select "+col +" from cl_compare_occurrence_recommendations, params \
                   where api=params.provider and start_t between params.from_date and params.to_date \
                   limit " +limit
    return (with_query+ prod_recommendations_query if with_str else prod_recommendations_query)

def create_validation_sample_from_DB(start_date, end_date, asr_provider, limit):
    query = get_prod_recommendation_query('job_uuid,transcription', limit, with_str=True)
    query_params = (start_date, end_date, asr_provider)
    return execute_query_create_datafram(query, query_params,df_columns=['uuid','context'],entity_name='validation_sample',
                                         asr_provider=asr_provider,start_date=start_date,end_date=end_date, limit=limit)

def create_validation_sample_from_DB(start_date, end_date, asr_provider, limit):
    query = get_prod_recommendation_query('job_uuid,transcription', limit, with_str=True)
    query_params = (start_date, end_date, asr_provider)
    return execute_query_create_datafram(query, query_params, df_columns=['uuid', 'context'],
                                         entity_name='validation_sample',
                                         asr_provider=asr_provider, start_date=start_date, end_date=end_date,
                                         limit=limit)


def get_prod_occurrences(start_date, end_date, asr_provider, limit):
    query = "WITH params (from_date,to_date,provider) as (values (%s,%s,%s)) \
            select occ.id, occ.event_type_id, tocc.name as event_name, rec.job_uuid \
            from  (select * from cl_compare_occurrence_recommendations, params \
                   where api=params.provider and job_uuid in (" + get_prod_recommendation_query('job_uuid', limit) + ")) as rec, \
                  (select * from cl_occurrences, params where device_id in \
                    (select id from cl_devices where customer_id not in (55, 4, 51, 3, 88, 52, 69, 56, 48, 93, 49, 99, 50, 2)) \
                    and event_type_id not in (121, 122, 116)\
                    and date(start_t) between params.from_date and params.to_date) as occ  join tagging_classes tocc on occ.event_type_id = tocc.id\
            where rec.device_id = occ.device_id \
            and (rec.start_t::timestamp, rec.end_t::timestamp) overlaps (occ.start_t::timestamp, occ.end_t::timestamp) \
            and (occ.start_t::timestamp, occ.end_t::timestamp) overlaps (rec.start_t::timestamp, rec.end_t::timestamp) \
            "
    query_params = (start_date, end_date, asr_provider)
    return execute_query_create_datafram(query, query_params, df_columns=['uuid','occurrence_id','event_id','event_name'],
                                         entity_name='occurrences',
                                         asr_provider=asr_provider, start_date=start_date, end_date=end_date,
                                         limit=limit)


def get_prod_recommendations(start_date, end_date, asr_provider, limit):
        query_params = (start_date, end_date, asr_provider)
        query = get_prod_recommendation_query('job_uuid, event_type_id, transcription', limit, with_str=True)
        return execute_query_create_datafram(query, query_params,
                                             df_columns=['uuid', 'event_type_id', 'transcription'],
                                             entity_name='recommendations',
                                             asr_provider=asr_provider, start_date=start_date, end_date=end_date,
                                             limit=limit)
