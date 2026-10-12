"""Self-authored CC0 text recipes; no retrieved corpus or private labels."""
import hashlib
import random


def digest(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def originals(seed, count, excluded=()):
    if count < 100:
        raise ValueError('At least 100 original groups are required')
    rng = random.Random(seed)
    seen = set(excluded)
    result = []
    emoji = ['😀', '🌈', '❤️', '☀️', '👍🏽', '👩\u200d💻', '👨\u200d🔬', '👩🏽\u200d🚀', '🏳️\u200d🌈', '🧑🏻\u200d🎨', '👨\u200d👩\u200d👧', '🫶🏾', '🧑\u200d🚀', '🧑\u200d💻', '🐱', '☕️', '🏄🏽\u200d♀️']
    attempts = 0
    while len(result) < count:
        index = attempts
        attempts += 1
        category = index % 6
        serial = rng.randrange(100000, 999999)
        if category == 0:
            text = f'虚构样本{serial} 晴川记录 {2035 + index % 8}-03-{1 + index % 28:02d}'
            kind = 'plain'
        elif category == 1:
            text = f'Orbit {serial}  cafe\u0301  rank {index % 99:02d}'
            kind = 'plain'
        elif category == 2:
            text = f'{serial}.{index % 100:02d} / 2041-11-{1 + index % 28:02d}'
            kind = 'plain'
        elif category == 3:
            text = f'[{serial}] <-> ─── ==== !? ;: /\\ _'
            kind = 'punctuation'
        elif category == 4:
            eye = rng.choice(['ω', 'Д', 'ಥ', '°', '^', '•', 'o', 'T'])
            mouth = rng.choice(['▽', '_', '∀', '益', '3', '皿'])
            text = rng.choice(['つ', '╯', 'ノ', '']) + '(' + eye + rng.choice(['', '\u0304']) + rng.choice([' ', '  ', '']) + mouth + ' ' + eye + ')' + rng.choice(['ノ', 'つ', '!', '──'])
            kind = 'face'
        else:
            # Each sequence remains an exact single original label; repeats are deduplicated.
            text = emoji[(index // 6) % len(emoji)]
            kind = 'emoji'
        key = digest(text)
        if key in seen:
            continue
        seen.add(key)
        result.append({'label': text, 'kind': kind, 'label_sha256': key, 'group': 'text:' + key})
    rng.shuffle(result)
    return result
