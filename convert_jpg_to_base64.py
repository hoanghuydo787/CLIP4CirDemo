import base64

with open('./upload/B000NWU8MU.jpg', 'rb') as image_file:
    encoded_string = base64.b64encode(image_file.read())
    print(encoded_string.decode('utf-8'))