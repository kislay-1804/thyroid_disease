"""
    This code is about Database Operations
    Database Operations include upload as well as retrieval of data from the database
    The database that has been used here is MongoDB
"""

import os
import pandas as pd
from pymongo import MongoClient
from pymongo.errors import ConnectionError
import shutil
from application_logging.logger import App_Logger
import zipfile

class DBOperation:
    """
        This class is used for performing the database operations
    """
    
    def __init__(self):
        self.path = 'Training_Database/'
        self.badFilePath = "Training_Raw_files_validated/Bad_Raw"
        self.goodFilePath = "Training_Raw_files_validated/Good_Raw"
        self.logger = App_Logger()
        self.client = MongoClient(f"mongodb+srv://IamAni07:<password>@iamani0709.2benb.mongodb.net/?retryWrites=true&w=majority&appName=IamAni0709")
        self.db = self.client["thyroid_disease"]
        self.collection = self.db["Good_Raw_Data"]

    def databaseConnection(self):
        """
            Connect to MongoDB and return the collection object.
        """
        
        try:
            self.logger.log("Training_Logs/MongoDBConnectionLog.txt", "MongoDB database connection successful")
        except Exception as e:
            self.logger.log("Training_Logs/MongoDBConnectionLog.txt", "Error while connecting to database: %s" % e)
            raise e

    def createTableDb(self):
        """
            This function is used to create a table in the database
        """
        
        try:

            session = self.databaseConnection()
            
            try:
                integer = "int"
                var = "varchar"
                age = "age"
                sex = "sex"
                on_thyroxine = "on_thyroxine"
                query_on_thyroxine = "query_on_thyroxine"
                on_antithyroid_medication = "on_antithyroid_medication"
                sick = "sick"
                pregnant = "pregnant"
                thyroid_surgery = "thyroid_surgery"
                I131_treatment = "I131_treatment"
                query_hypothyroid = "query_hypothyroid"
                query_hyperthyroid = "query_hyperthyroid"
                lithium = "lithium"
                goitre = "goitre"
                tumor = "tumor"
                hypopituitary = "hypopituitary"
                psych = "psych"
                TSH_measured = "TSH_measured"
                TSH = "TSH"
                T3_measured = "T3_measured"
                T3 = "T3"
                TT4_measured = "TT4_measured"
                TT4 = "TT4"
                T4U_measured = "T4U_measured"
                T4U = "T4U"
                FTI_measured = "FTI_measured"
                FTI = "FTI"
                TBG_measured = "TBG_measured"
                TBG = "TBG"
                referral_source = "referral_source"
                Class = "Class"

                session.execute(f"CREATE TABLE db.Good_Raw_Data(
                                                                {age} {integer} PRIMARY KEY, 
                                                                {sex} {var}, 
                                                                {on_thyroxine} {var},
                                                                {query_on_thyroxine} {var},
                                                                {on_antithyroid_medication} {var},
                                                                {sick} {var},
                                                                {pregnant} {var},
                                                                {thyroid_surgery} {var}, 
                                                                {I131_treatment} {var}, 
                                                                {query_hypothyroid} {var}, 
                                                                {query_hyperthyroid} {var}, 
                                                                {lithium} {var}, 
                                                                {goitre} {var}, 
                                                                {tumor} {var}, 
                                                                {hypopituitary} {var}, 
                                                                {psych} {var}, 
                                                                {TSH_measured} {var}, 
                                                                {TSH} {var}, 
                                                                {T3_measured} {var}, 
                                                                {T3} {var}, 
                                                                {TT4_measured} {var}, 
                                                                {TT4} {var}, 
                                                                {T4U_measured} {var}, 
                                                                {T4U} {var}, 
                                                                {FTI_measured} {var}, 
                                                                {FTI} {var}, 
                                                                {TBG_measured} {var}, 
                                                                {TBG} {var}, 
                                                                {referral_source} {var}, 
                                                                {Class} {var});")
                
                file = open("Training_Logs/MongoDBTableLog.txt", 'a+')
                self.logger.log(file, "Tables created successfully!!")
                self.logger.log(file, "Database closed successfully")
                session.shutdown()
                file.close()
                
            except:
                file = open("Training_Logs/MongoDBTableLog.txt", 'a+')
                self.logger.log(file, "Table already present in database")
                self.logger.log(file, "Database closed successfully")
                session.shutdown()
                file.close()

        except Exception as e:
            file = open("Training_Logs/MongoDBTableLog.txt", 'a+')
            self.logger.log(file, "Error while creating table: %s " % e)
            file.close()
            file = open("Training_Logs/MongoDBTableLog.txt", 'a+')
            self.logger.log(file, "Database closed successfully")
            session.shutdown()
            file.close()
            raise e

    def insertIntoTableGoodData(self):
        """
            This function inserts good data into the MongoDB table.
        """
        
        session = self.databaseConnection()
        goodFilePath = self.goodFilePath
        badFilePath = self.badFilePath
        onlyfiles = [f for f in listdir(goodFilePath)]
        log_file = open("Training_Logs/MongoDBInsertionLog.txt", 'a+')

        age = "age"
        sex = "sex"
        on_thyroxine = "on_thyroxine"
        query_on_thyroxine = "query_on_thyroxine"
        on_antithyroid_medication = "on_antithyroid_medication"
        sick = "sick"
        pregnant = "pregnant"
        thyroid_surgery = "thyroid_surgery"
        I131_treatment = "I131_treatment"
        query_hypothyroid = "query_hypothyroid"
        query_hyperthyroid = "query_hyperthyroid"
        lithium = "lithium"
        goitre = "goitre"
        tumor = "tumor"
        hypopituitary = "hypopituitary"
        psych = "psych"
        TSH_measured = "TSH_measured"
        TSH = "TSH"
        T3_measured = "T3_measured"
        T3 = "T3"
        TT4_measured = "TT4_measured"
        TT4 = "TT4"
        T4U_measured = "T4U_measured"
        T4U = "T4U"
        FTI_measured = "FTI_measured"
        FTI = "FTI"
        TBG_measured = "TBG_measured"
        TBG = "TBG"
        referral_source = "referral_source"
        Class = "Class"

        for file in onlyfiles:
            
            try:
                data = pd.read_csv(goodFilePath + '/' + file)
                for i, row in data.iterrows():

                    query = f"insert into db.Good_Raw_Data (
                                                            {age}, 
                                                            {sex}, 
                                                            {on_thyroxine}, 
                                                            {query_on_thyroxine}, 
                                                            {on_antithyroid_medication}, 
                                                            {sick}, 
                                                            {pregnant}, 
                                                            {thyroid_surgery}, 
                                                            {I131_treatment}, 
                                                            {query_hypothyroid},
                                                            {query_hyperthyroid}, 
                                                            {lithium}, 
                                                            {goitre}, 
                                                            {tumor}, 
                                                            {hypopituitary},
                                                            {psych}, 
                                                            {TSH_measured}, 
                                                            {TSH}, 
                                                            {T3_measured}, 
                                                            {T3}, 
                                                            {TT4_measured}, 
                                                            {TT4}, 
                                                            {T4U_measured}, 
                                                            {T4U},
                                                            {FTI_measured}, 
                                                            {FTI}, 
                                                            {TBG_measured}, 
                                                            {TBG},
                                                            {referral_source}, 
                                                            {Class}) 
                                                            values (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)"

                    try:
                        session.execute(query, tuple(row))
                        self.logger.log(log_file, " %s: File loaded successfully!!" % file)

                    except Exception as e:
                        raise e

            except Exception as e:
                self.logger.log(log_file, "Error while inserting data into table: %s " % e)
                shutil.move(goodFilePath + '/' + file, badFilePath)
                self.logger.log(log_file, "File Moved Successfully %s" % file)
                log_file.close()
                session.shutdown()

        session.shutdown()
        log_file.close()

    def selectingDatafromtableintocsv(self):
        """
            This function is used to select data from the table and write it into a csv file.
        """

        self.fileFromDb = 'Training_FileFromDB/'
        self.fileName = 'InputFile.csv'
        log_file = open("Training_Logs/ExportToCsv.txt", 'a+')
        try:
            session = self.dataBaseConnection()

            main_list = []
            for i in session.execute("select * from db.Good_Raw_Data;"):
                main_list.append(i)

            # Make the CSV ouput directory
            if not os.path.isdir(self.fileFromDb):
                os.makedirs(self.fileFromDb)

            # converting main_list to data frame
            df = pd.DataFrame(main_list)

            # saving the data frame df to output directory

            df.to_csv(f"{self.fileFromDb}" + '//' + f"{self.fileName}", index=False)

            self.logger.log(log_file, "File exported successfully!!!")
            log_file.close()

        except Exception as e:
            self.logger.log(log_file, "File exporting failed. Error : %s" % e)
            log_file.close()

    def TurncateTable(self):
        """
            This function is used to truncate the table.
        """
        
        try:
            session = self.dataBaseConnection()
            session.execute("TRUNCATE TABLE db.Good_Raw_Data;")
            
            file = open("Training_Logs/MongoDBTableLog.txt", 'a+')
            self.logger.log(file, "Tables turncated successfully!!")
            file.close()
            session.shutdown()
            
        except Exception as e:
            file = open("Training_Logs/MongoTableLog.txt", 'a+')
            self.logger.log(file, "Table Turncate failed. Error : %s" % e)
            file.close()
            session.shutdown()