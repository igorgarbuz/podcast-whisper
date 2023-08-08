from faster_whisper import WhisperModel
import re
import os
from tqdm import tqdm
from translate import Translator
import dateparser


# translator = Translator(from_lang="sv", provider='deepl', secret_access_key='58723f88-8486-c86f-3d38-0f0ead61e600:fx', to_lang="en", pro=True)


# def translate(text):
#     return translator.translate(text)

def translate_date(date):
    print(date)
    parsed_date = dateparser.parse(date, languages=['sv'])
    return parsed_date.strftime("%d %B %Y")

# from pyannote.audio import Pipeline

# pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization")

model_size = 'large-v2'
# model_size = 'tiny'

model = WhisperModel(model_size, device="cuda", device_index=[0, 1], compute_type="float16")

data_dir = '/root/podcast-whisper/solcellskollen.en/episodes'
output_dir = '/root/podcast-whisper/solcellskollen.en/transcripts/'

test_file = 'Avsnitt #19Bonusavsnitt: Erik Dölerud, Om Hyperloop 3 december 201815:53.mp3'

# def write_vtt(segments, file):
#     file.write("WEBVTT\n\n")
#     for i, segment in enumerate(segments):
#         # print(segment)
#         start_time = segment.start
#         end_time = segment.end
#         text = segment.text

#         file.write(f"{i+1}\n")
#         file.write(f"{start_time} --> {end_time}\n")
#         file.write(f"{text}\n\n")

# def time_str_to_seconds(time_str):
#     h, m, s = time_str.split(':')
#     s, ms = s.split('.')
#     return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000

# def add_speaker_identification(transcript, diarization):
#     output = []
#     for line in transcript.splitlines():
#         if "-->" in line:
#             start, end = line.split(" --> ")
#             start_time = time_str_to_seconds(start)
#             end_time = time_str_to_seconds(end)

#             speaker = None
#             for turn, _, speaker in diarization.itertracks(yield_label=True):
#                 if turn.start <= start_time < turn.end:
#                     break

#             output.append(f"{speaker}: {line}\n")
#         else:
#             output.append(f"{line}\n")

#     return "".join(output)

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

def read_and_parse_files(directory, test=False):
    pattern = r'Avsnitt #(\d+)(.*?),? Om (.*?)(\d{1,2} \w+ \d{4})(.*).mp3'
    parsed_data = {}
    if test:
        return { test_file: { 'number': '19', 'guest_name': 'Erik Dölerud', 'title': 'Hyperloop', 'date': '3 december 2018' }}

    for filename in os.listdir(directory):
        if filename.endswith('.mp3'):
            match = re.match(pattern, filename)
            if match:
                number = match.group(1)
                guest_name = match.group(2).strip()
                # title = translate(match.group(3).strip().capitalize())
                title = match.group(3).strip().capitalize()
                date = translate_date(match.group(4))

                parsed_data[filename] = { 'number': number, 'guest_name': guest_name, 'title': title, 'date': date }

    return parsed_data

def transcribe(test):
    data = read_and_parse_files(data_dir, test=test)
    print(f'Found {len(data)} files')
    for filename, parsed_data in tqdm(data.items()):
        output_filename = f'{parsed_data["number"]} - {parsed_data["guest_name"]} - {parsed_data["title"]} - {parsed_data["date"]}'
        output_file_path = f'{output_dir}/{output_filename}'
        file_path = f'{data_dir}/{filename}'
        print(f'Processing {filename}')

        segments, _ = model.transcribe(file_path, task="translate", beam_size=5)
        with open(f'{output_file_path}.vtt', 'w') as f:
            write_vtt(segments, f)


if __name__ == '__main__':
    transcribe(test=False)
