from pyannote.audio import Pipeline
pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization")

def write_vtt(segments, file):
    file.write("WEBVTT\n\n")
    for i, segment in enumerate(segments):
        # print(segment)
        start_time = segment.start
        end_time = segment.end
        text = segment.text

        file.write(f"{i+1}\n")
        file.write(f"{start_time} --> {end_time}\n")
        file.write(f"{text}\n\n")

def time_str_to_seconds(time_str):
    h, m, s = time_str.split(':')
    s, ms = s.split('.')
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000

def add_speaker_identification(transcript, diarization):
    output = []
    for line in transcript.splitlines():
        if "-->" in line:
            start, end = line.split(" --> ")
            start_time = time_str_to_seconds(start)
            end_time = time_str_to_seconds(end)

            speaker = None
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                if turn.start <= start_time < turn.end:
                    break

            output.append(f"{speaker}: {line}\n")
        else:
            output.append(f"{line}\n")

    return "".join(output)
