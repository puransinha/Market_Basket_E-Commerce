
import psycopg2  # import the postgres library
import sys
import pandas as pd

# connect to the database
conn = psycopg2.connect(host='localhost',
                        dbname='postgres',
                        user='mydb',
                        password='123abc',
                        port='5432')
# create a cursor object
# cursor object is used to interact with the database
cur = conn.cursor()

cur.execute("""truncate table "public".emp_data;""")


def pg_load_table(file_path, table_name, dbname, host, port, user, pwd):
    '''
    This function upload csv to a target table
    '''
    try:
        cur = conn.cursor()
        f = open(file_path, "r")
        # cur.copy_expert("copy {} from STDIN CSV HEADER QUOTE '\"'".format(table_name), f)
        cur.copy_from(f, "emp_data")
        cur.execute("commit;")
        print("Loaded data into {}".format(table_name))
        conn.close()
        print("DB connection closed.")

    except Exception as e:
        print("Error: {}".format(str(e)))
        sys.exit(1)


file_path = '/home/lns-lp-037/Documents/dt1.csv'
table_name = 'emp_data'
dbname = 'postgres'
host = 'localhost'
port = '5432'
user = 'mydb'
pwd = '123abc'
pg_load_table(file_path, table_name, dbname, host, port, user, pwd)

# psql -h port -d db -U user -c "\copy products from 'products.csv' with delimiter as ',' csv header;"

