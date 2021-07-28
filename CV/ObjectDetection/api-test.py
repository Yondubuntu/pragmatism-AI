import requests

response = requests.post("http://localhost:8080/", files={'file': open('aespa.png', 'rb')})

print(response.json())