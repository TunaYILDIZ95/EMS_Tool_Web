import json
import os


json_data = open('C:/Users/tunay/Desktop/EMS_Django/EMS_Web/static/json_files/countries2.json')
data1 = json.load(json_data) # deserialises it
data2 = json.dumps(data1) # json formatted string
json_data.close()


json_data = open('C:/Users/tunay/Desktop/EMS_Django/EMS_Web/static/json_files/cities.json')
data3 = json.load(json_data) # deserialises it
data4 = json.dumps(data3) # json formatted string
json_data.close()


data5 = data3['countries']
data5[0]['country']

asd = [tuple([item['code'],item['name']]) for item in data1]

city_tuple = [tuple(item.value()) for item in data3]

data = {}
for i in data5:
    data[i['country']] = i['states']

with open('data.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=4)
