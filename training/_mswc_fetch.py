"""Download the 3 MSWC English shards that contain 'help', extract the help
clips, decode opus->wav (48 kHz mono), and report counts + durations."""
import os, io, tarfile
import numpy as np
import soundfile as sf
from huggingface_hub import hf_hub_download

SHARDS = {'train': 33, 'dev': 4, 'test': 4}
OUT_ROOT = os.path.join('data', 'mswc_help')


def decode_opus(raw_bytes):
    try:
        x, sr = sf.read(io.BytesIO(raw_bytes), dtype='float32', always_2d=False)
        if x.ndim > 1:
            x = x.mean(axis=1)
        return x, sr
    except Exception as e:
        return None, str(e)


for split, n in SHARDS.items():
    out = os.path.join(OUT_ROOT, split)
    os.makedirs(out, exist_ok=True)
    print('=== {} : downloading shard {} ==='.format(split, n))
    path = hf_hub_download('MLCommons/ml_spoken_words',
                           'data/opus/en/{}/audio/{}.tar.gz'.format(split, n),
                           repo_type='dataset')
    print('  shard size MB:', round(os.path.getsize(path) / 1e6, 1))
    durs = []
    saved = 0
    fail = 0
    with tarfile.open(path) as t:
        for m in t:
            b = os.path.basename(m.name)
            if not (b.startswith('help_common_voice') and b.endswith('.opus')):
                continue
            f = t.extractfile(m)
            if f is None:
                continue
            x, sr = decode_opus(f.read())
            if x is None:
                fail += 1
                continue
            wav_path = os.path.join(out, os.path.splitext(b)[0] + '.wav')
            sf.write(wav_path, x, sr)
            durs.append(len(x) / sr)
            saved += 1
    durs = np.array(durs)
    if len(durs):
        print('  saved {} help clips ({} decode fails); duration mean {:.2f}s '
              'min {:.2f}s max {:.2f}s sr~{}'.format(
                  saved, fail, durs.mean(), durs.min(), durs.max(), sr))
    else:
        print('  saved 0 (fails {})'.format(fail))
print('DONE ->', OUT_ROOT)
