import requests
import os
import zipfile


def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)    

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)

if __name__ == "__main__":
    file_id = '1mVGfN0Xoj--26_NIPgXS48ub-Rr89GIw'
    destination = os.path.dirname(os.path.abspath(__file__)) + '/networks.zip'
    extract_destination = os.path.dirname(os.path.abspath(__file__)) + '/nn'
    print('Starting networks download')
    if os.path.isfile(destination):
        print('Networks were already downloaded')
    else:
        download_file_from_google_drive(file_id, destination)
        print('Networks were succesfully downloaded.')
        with zipfile.ZipFile(destination, 'r') as zip_ref:
            zip_ref.extractall(extract_destination)
