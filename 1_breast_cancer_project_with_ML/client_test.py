import requests

sample_data = [
    17.99, 10.38, 122.8, 1001.0, 0.1184, 0.2776, 0.3001,
    0.1471, 0.2419, 0.07871, 1.095, 0.9053, 8.589, 153.4,
    0.006399, 0.04904, 0.05373, 0.01587, 0.03003, 0.006193,
    25.38, 17.33, 184.6, 2019.0, 0.1622, 0.6656, 0.7119,
    0.2654, 0.4601, 0.1189
]

url = 'http://127.0.0.1:8000/predict'

response = requests.post(url, json={'data':sample_data})

if response.status_code == 200:
    print("Tahmin Sonucu:",response.json())
elif response.status_code == 400:
    print("Hata: ",response.json())
else:
    print("Something went wrong")