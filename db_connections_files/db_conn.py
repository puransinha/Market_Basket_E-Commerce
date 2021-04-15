#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  5 20:45:40 2021

@author: lns-lp-037
"""
## in case of using Ubuntu ... need to start the services using the command ... sudo service mongod start
import psycopg2  # import the postgres library
import pymongo
from pymongo import MongoClient
import sys
import pandas as pd

def conn_db_postgres():
    try:
        # Connect to an existing database
        connection = psycopg2.connect(user="postgres",
                                      password="root",
                                      host="localhost",
                                      port="5432",
                                      database="postgres")

        # Create a cursor to perform database operations
        cursor = connection.cursor()
        # Print PostgreSQL details
        print("PostgreSQL server information")
        print(connection.get_dsn_parameters(), "\n")
        # Executing a SQL query
        cursor.execute("SELECT version();")
        # Fetch result
        record = cursor.fetchone()
        print("You are connected to - ", record, "\n")

    except (Exception, Error) as error:
        print("Error while connecting to PostgreSQL", error)
    finally:
        if (connection):
            cursor.close()
            connection.close()
            print("PostgreSQL connection is closed")

def conn_db_mongodb():
    # Checking the Connections
    DEFAULT_CONNECTION_URL = "mongodb://localhost:27017/"

    # Establish a connection with mongoDB
    client = pymongo.MongoClient(DEFAULT_CONNECTION_URL)

    print(client)

    DB_NAME = "mydb"

    # Establish a connection with mongoDB
    client = pymongo.MongoClient(DEFAULT_CONNECTION_URL)

    # Create a DB
    dataBase = client[DB_NAME]

    def checkExistence_DB(DB_NAME, client):
        """It verifies the existence of DB"""
        DBlist = client.list_database_names()
        if DB_NAME in DBlist:
            print(f"DB: '{DB_NAME}' exists")
            return True
        print(f"DB: '{DB_NAME}' not yet present OR no collection is present in the DB")
        return False
    _ = checkExistence_DB(DB_NAME=DB_NAME, client=client)


