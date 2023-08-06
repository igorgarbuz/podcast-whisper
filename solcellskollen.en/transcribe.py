from faster_whisper import WhisperModel
import re
import os

model_size = 'large-v2'
# model_size = 'tiny'

model = WhisperModel(model_size, device="cuda", compute_type="float16")

data_dir = '/workspace/solcellskollen_episodes/'
output_dir = '/workspace/solcellskollen_transcripts/'

test_file = 'Avsnitt #19Bonusavsnitt: Erik Dölerud, Om Hyperloop 3 december 201815:53.mp3'


def read_and_parse_files(directory, test=False):
    pattern = r'Avsnitt #(\d+)(.*?),? Om (.*?)(\d+ \w+ \d{4}(?:\d{2}.*)?\.mp3)'
    parsed_data = {}
    if test:
        return { test_file: { 'number': '19', 'guest_name': 'Erik Dölerud', 'title': 'Hyperloop', 'date': '3 december 2018' }}

    for filename in os.listdir(directory):
        if filename.endswith('.mp3'):
            match = re.match(pattern, filename)
            if match:
                number = match.group(1)
                guest_name = match.group(2).strip()
                title = match.group(3).strip().capitalize()
                date_and_time = match.group(4)
                date = re.split(r'\d{2}:\d{2}:\d{2}', date_and_time)[0].strip()


                parsed_data[filename] = { 'number': number, 'guest_name': guest_name, 'title': title, 'date': date }

    return parsed_data

def transcribe():
    data = read_and_parse_files(data_dir, test=True)
    print(f'Found {len(data)} files')
    for filename, parsed_data in data.items():
        # print(filename)
        # print(parsed_data)
        output = model.transcribe(f'{data_dir}/{filename}', task="translate", beam_size=5)
        with open(f'{output_dir}{parsed_data["number"]}_{parsed_data["guest_name"]}_{parsed_data["title"]}_{parsed_data["date"]}.txt', 'w') as f:
            f.write(output)


if __name__ == '__main__':
    transcribe()
