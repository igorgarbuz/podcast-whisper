from faster_whisper import WhisperModel
import re
import os
from tqdm import tqdm
import dateparser
# from dotenv import load_dotenv
from decouple import config
# import pdb
from deep_translator import DeeplTranslator

# translator = DeeplTranslator(source='sv', target='en', api_key=config('DEEPL_API_KEY'))


# MODEL_SIZE = 'large-v2'
MODEL_SIZE = 'tiny'

def translate_date(date):
    parsed_date = dateparser.parse(date, languages=['sv'])
    return parsed_date.strftime("%d %B %Y")

model = WhisperModel(MODEL_SIZE, device="cuda", device_index=[0, 1], compute_type="float16")

# DATA_DIR = config('DATA_DIR_DOCKER')
# OUTPUT_DIR = config('OUTPUT_DIR_DOCKER')
DATA_DIR = config['DATA_DIR']
OUTPUT_DIR = config('OUTPUT_DIR')

FILES = []


def write_vtt(segments, file, maxLineWidth=64):
    def split_text(text, maxLineWidth):
        words = text.split()
        lines = []
        current_line = []

        for word in words:
            if len(" ".join(current_line + [word])) <= maxLineWidth:
                current_line.append(word)
            else:
                lines.append(" ".join(current_line))
                current_line = [word]

        if current_line:
            lines.append(" ".join(current_line))

        return lines

    def format_time(seconds):
        hours, remainder = divmod(seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{int(hours):02d}:{int(minutes):02d}:{seconds:06.3f}"

    file.write("WEBVTT\n\n")
    for i, segment in enumerate(segments):
        start_time = format_time(segment.start)
        end_time = format_time(segment.end)
        text = segment.text
        lines = split_text(text, maxLineWidth)

        file.write(f"{start_time} --> {end_time}\n")
        file.write("\n".join(lines) + "\n\n")

def read_and_parse_files(directory):
    pattern = r'Avsnitt #(\d+)(.*?), (.*?)(\d{1,2} \w+ \d{4})(.*).mp3'
    parsed_data = {}
    files = FILES if len(FILES) > 0 else os.listdir(directory)

    for filename in files:
        if filename.endswith('.mp3'):
            match = re.match(pattern, filename)
            # pdb.set_trace()
            if match:
                number = match.group(1)
                guest_name = match.group(2).strip()
                # title = translate(match.group(3).strip().capitalize())
                title = match.group(3).strip().capitalize()
                date = translate_date(match.group(4))

                parsed_data[filename] = { 'number': number, 'guest_name': guest_name, 'title': title, 'date': date }

    return parsed_data

def transcribe(test):
    data = read_and_parse_files(DATA_DIR)
    print(f'Found {len(data)} files')
    print(data)
    for filename, parsed_data in tqdm(data.items()):
        output_filename = f'{parsed_data["number"]} - {parsed_data["guest_name"]} - {parsed_data["title"]} - {parsed_data["date"]}'
        output_file_path = f'{OUTPUT_DIR}/{output_filename}'
        file_path = f'{DATA_DIR}/{filename}'
        print(f'Processing {filename}')

        segments, _ = model.transcribe(file_path, task="translate", beam_size=5)
        with open(f'{output_file_path}.vtt', 'w') as f:
            write_vtt(segments, f)


if __name__ == '__main__':
    transcribe(test=False)
