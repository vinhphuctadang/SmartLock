import requests
import base64
BASE_URL = 'http://localhost:8080/'
def main():
    with open('image.jpg', 'rb') as f:
        image_bin = f.read()
    image_label = 'phuc'
    image_base64 = base64.b64encode(image_bin)
    myobj = { 
        'image': image_base64,
        'label': image_label
    }

    url = BASE_URL+'upload'
    x = requests.post(url, data=myobj)
    print(x.text)
    
main()