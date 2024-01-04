import librosa as lb
import csv

AUDIO_PATH = "audio/wavs/"
CSV_FILE = "audio/metadata.csv"


def getDataset(number_of_records, delimiter="|", sr=16000, min_words=20):
    rows = []
    try:
        with open(CSV_FILE, "r", newline="") as file:
            reader = csv.reader(file, delimiter=delimiter)
            for row in reader:
                if len(row) > 0 and len(row[1].split()) > min_words:
                    waveform, _ = lb.load(f"{AUDIO_PATH}{row[0]}.wav", sr=sr)
                    rows.append((waveform, row[1]))
                    if len(rows) >= number_of_records:
                        break
    except Exception as e:
        print(f"Error reading file: {e}")
    return rows
