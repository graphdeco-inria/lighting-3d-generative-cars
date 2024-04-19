import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import requests
from tqdm import tqdm
import re
import gdown
import click


def donwload_ihdri(dest, category):
    assert category in ['indoor', 'outdoor']
    os.makedirs(dest, exist_ok=True)

    count = 0
    hdri_url = f"https://www.ihdri.com/hdri-skies-{category}/"
    target_string = requests.get(hdri_url).text
    pattern = re.compile(r"<span>Download .HDR in 16K<\/span><\/a>.+?drive.google.com\/file\/d\/(.+?)\/view.+?", flags=re.DOTALL)
    for match in tqdm(pattern.finditer(target_string)):
        drive_id = match.group(1)
        filename = os.path.join(dest, f'{str(count).zfill(5)}.hdr')
        gdown.download(id=drive_id, output=filename)
        count += 1
    print('iHDRI download completed')


def download_polyhaven(dest, category, resolution="1k", ext="hdr"):
    assert resolution in ['1k', '4k']
    assert category in ['indoor', 'outdoor']
    assert ext in ['hdr', 'exr']

    url = 'https://api.polyhaven.com'
    hdris = "/assets?t=hdris"
    files = "/files"

    # get the url for the hdri json
    hdri_url = url + hdris
    if category != "all":
        hdri_url = hdri_url + "&c=" + category

    # get a list of all the hdri keys
    hdris = list(requests.get(hdri_url).json().keys())

    for i, hdri in enumerate(tqdm(hdris)):
        file_json = requests.get(url + files + "/" + hdri).json()

        try:
            print("\nurl:", file_json["hdri"][resolution][ext]["url"])
            filename = os.path.join(dest, f'{str(i).zfill(5)}.{ext}')
            with open(filename, 'wb') as hdri_file:
                response = requests.get(file_json["hdri"][resolution][ext]["url"], allow_redirects = True)
                hdri_file.write(response.content)
        except Exception as e:
            print("Download failed, possibly because", ext, "is not available for this image.")
            continue
    print('PolyHaven download completed')

@click.command()
@click.option('--polyhaven_dir',  type=str, required=True)
@click.option('--ihdri_dir',  type=str, required=True)
def main(polyhaven_dir, ihdri_dir):
    download_polyhaven(polyhaven_dir, category="outdoor")
    donwload_ihdri(ihdri_dir, category="outdoor")

if __name__ == "__main__":
    main()