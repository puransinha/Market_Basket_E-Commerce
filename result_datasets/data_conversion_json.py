import json
import csv


# Conversion from CSV to JSON Standard format
import json
import csv

def csv_to_json():
    s = input('Enter the File csv File Name to Convert ...:')
    csvFilePath = r'result_datasets/' + s
    jsonFilePath = r'result_json/result1_data.json'
    jsonArray = []
    with open(csvFilePath, encoding='utf-8') as csvf:
        csvReader = csv.DictReader(csvf)
        for row in csvReader:
            jsonArray.append(row)
    with open(jsonFilePath, 'w', encoding='utf-8') as jsonf:
        jsonString = json.dumps(jsonArray, indent=4)
        jsonf.write(jsonString)
        csv_to_json(csvFilePath, jsonFilePath)

    print('done')
