"""Known-input real screenshot sets: reproducible face lists, entry scripts and manifests from stored text.

A known-input set is typed by the user into the target UI with the input method they normally use,
then screenshotted. Labels are never read from pixels: the truth of each line is the text the UI
stored after typing, copied by the user into ``typed/shot-NNN.txt``. The entry script only guides
typing; where the stored text differs from it (an input method may change bracket widths), the
stored text wins and the difference is counted. Everything lives in an ignored private directory.

Retired 2026-10-15 (add-browser-rendering-evaluation): typing a generated entry script is not a usable
validation route. Real users type or paste expressions in either bracket width, so no set may require a
particular input method or width. ``build`` stays only to reproduce the retired known-input-2 listing;
``retire`` records an unused list, and ``build_manifest`` refuses any list a registered round has retired.
Real screenshots now come from everyday use with the copied comment text as truth.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import random

from tools.OcrDatasets.render import Unsupported
from tools.PretrainedOcr import grammar_corpus as gc, kaomoji_grammar as kg
from tools.PretrainedOcr import round_profile as rp
from tools.PretrainedOcr.common import utc
from tools.PretrainedOcr.data import MixedRenderer
from tools.PretrainedOcr.grammar_data import renderable_atoms
from tools.RegionKaomoji import candidates as c, real_screens as rs

LIST_SCHEMA = 'localocr.known-input-list.v1'
SCRIPT_SCHEMA = 'localocr.known-input-entry-script.v1'
RETIREMENT_SCHEMA = 'localocr.known-input-retirement.v1'
RETIREMENT_PATTERN = 'known-input-*-retirement.json'
USAGE = 'independent-evaluation-only'
VARIANTS = {'A': '句尾，空格分隔', 'B': '句尾，紧贴正文', 'C': '句首，空格分隔', 'D': '句中，两侧空格', 'E': '句中，两侧紧贴'}
# Six standalone screens of seven typed faces and one of eight (50 faces), then seven mixed screens of
# seven body lines with five embedded faces each (35 embedded).
STANDALONE_PATTERNS = ('BFFFBFFFBF',) * 6 + ('BFFFFBFFFF',)
MIXED_SCREENS, MIXED_LINES, MIXED_FACES = 7, 7, 5
PARTITION_PLAN = (('heldout', 15),)
# Widely used expressions written from the atom table (original listing, CC0); each must parse under the
# grammar and render with the registered fonts, and faces of an excluded earlier list never recur.
COMMON_FACES = (
    '(^_^)', '(^O^)', '(^▽^)', '(^∀^)', '(T_T)', '(T^T)', '(;_;)', '(>_<)', '(>▽<)', '(´∀`)', '(｀・ω・´)', '(*^_^*)', '(^_^;)',
    '(´・ω・`)', '(≧∇≦)', '(´▽`)', '(´ω`)', '(´Д`)', '(;´Д`)', '(°□°)', '(°Д°)', '(¬_¬)', '(-_-)', '(o_o)', '(O_O)', '(°o°)',
    '(^-^)', '(*´▽`*)', '(*°▽°*)', '(ノ´▽`)ノ', '(ノ^_^)ノ', 'ヽ(^o^)ノ', '(ノ°Д°)ノ', 'ヽ(°Д°)ノ', '(ಠ_ಠ)', '(;ω;)', '(◕‿◕✿)', '(-ω-)',
    '(＾ω＾)', '(^ω^)', '(≧ω≦)', '(>ω<)', '(´>ω<`)', '(〒_〒)', '(♥‿♥)', '(⊙▽⊙)', '(o_O)', '(-_-;)', '(´･_･`)', '(･ω･)', '(｀皿´)',
    '(`益´)', '(｡-_-｡)', '(｡•ω•｡)', '(｡^_^｡)', '(★ω★)', '(★▽★)', '(*°ω°*)', '(´°ω°`)', '(^▽^)/', '(T_T)/', 'ヽ(;▽;)ノ', '(´；Д；`)',
    '(;▽;)', '(°▽°)', '(•ω•)', '(•‿•)', '(•_•)', '(◕_◕)', 'ʕ•ω•ʔ', 'ʕ´•ᴥ•`ʔ', '(｡･ω･｡)ノ♡', '(´▽`)ノ', '(*^ω^*)', '(ノ≧▽≦)ノ',
    '(≧◡≦)', '(◡‿◡)', '(´◡`)', '(°ー°)', '(^ー^)', '(´ー`)', '(-ー-)', '「^_^」', '『^o^』', '【^_^】',
)
COMMON_COUNT = 35
MAXIMUM_FACE_LENGTH = 16
# Original short sentences without brackets; none parses as an expression.
BODY_LINES = (
    '今晚八点集合准时出发', '这次机制全程没有失误', '周常任务还差最后一次', '治疗压力有点大但是过了', '新装备的属性比想象中好',
    '明天继续练习开荒路线', '队友配合得非常默契', '副本掉落的材料已经分完', '记得补充药水再进本', '这个版本的坦克更好上手',
    '排队等了二十分钟才进', '今天的野队意外地顺利', '练习模式先熟悉一遍地形', '集合点在水晶旁边', '最后一个阶段差一点点',
    '下次换个职业试试看', '新手引导做完就能解锁', '周末再约一次满员团', '攻略视频看了三遍', '备注一下这次的掉落',
    '时间轴背得差不多了', '第二阶段的站位要记牢', '输出手法还要再练', '这张图的机制很有意思', '今天只打了两次就收工',
    '清完周本顺便做日常', '装备等级终于够了', '开荒第一天还算顺利', '队长临时有事先散了', '补给买够了再出发',
    '练完手法再去挑战', '想把坐骑换成新的', '下周的活动记得参加', '这次换了个新职业体验', '成功过本后合影留念',
)
README = """# 已知输入录入集（已作废，2026-10-15）

本目录位于忽略路径 `artifacts/`，内容不会进入仓库或发布包。

按录入脚本逐条键入颜文字再截图的验证方式已作废：其中的颜文字含大量无法用输入法直接打出的字符，
而且真实用户既会用输入法键入，也会从网页找来粘贴，半角全角都会出现，不能要求用户按某种方式或宽度输入。
本目录不再用于独立验证，`manifest` 会拒绝为已登记作废的输入清单构建清单。

真实字体下的评测改由浏览器真实字体渲染集完成；真实截图只来自日常使用，以复制出的评论文字为真值。
`entry-script.txt` 与 `expected/` 仅为生成记录，不需要录入。
"""


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, ensure_ascii=False, indent=2) + '\n' if not isinstance(value, str) else value
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(payload)


def excluded_texts(paths):
    texts = set()
    for path in paths:
        value = json.loads(Path(path).read_text(encoding='utf-8'))
        items = value.get('items', []) if isinstance(value, dict) else value
        texts.update(item['text'] if isinstance(item, dict) else item for item in items)
    return texts


def common_faces(rng, atoms, excluded, renderable, count=COMMON_COUNT):
    """``count`` widely used expressions that parse, render and are not excluded; partition recorded from the core."""
    pool = []
    for text in COMMON_FACES:
        parsed = kg.parse_face(text, atoms)
        if text in excluded or parsed is None or not renderable(text):
            continue
        eye1, mouth, eye2 = parsed['core']
        key = gc.core_key(parsed['bracket'], eye1, mouth, eye2)
        pool.append({'text': text, 'source': 'common-structure', 'core_partition': gc.partition(key), 'core_key': key,
                     'core_digest': gc.core_digest(key), 'structure': 'common'})
    if len(pool) < count:
        raise ValueError('Not enough renderable common expressions outside the excluded lists')
    return rng.sample(pool, count)


def compose_faces(rng, atoms, allowed, excluded, plan=PARTITION_PLAN, maximum_length=MAXIMUM_FACE_LENGTH, renderable=lambda text: True):
    """Widely used expressions first, then distinct atom-table compositions per partition plan, none in ``excluded``."""
    faces = common_faces(rng, atoms, excluded, renderable)
    seen = set(excluded) | {face['text'] for face in faces}
    for split, count in plan:
        produced = 0
        for _ in range(count * 400):
            if produced == count:
                break
            try:
                face = gc.compose_face(split, rng, atoms, allowed)
            except Unsupported:
                continue
            text = face['text']
            if text in seen or len(text) > maximum_length or kg.parse_face(text, atoms) is None or any(ch.isspace() for ch in (text[0], text[-1])):
                continue
            seen.add(text)
            faces.append({'text': text, 'source': 'atom-table-' + split + '-partition', 'core_partition': split,
                          'core_key': face['core_key'], 'core_digest': face['core_digest'], 'structure': face['structure']})
            produced += 1
        if produced != count:
            raise ValueError('Unable to compose enough distinct expressions for partition ' + split)
    return faces


def embed(body, face, variant):
    if variant == 'A':
        return body + ' ' + face
    if variant == 'B':
        return body + face
    if variant == 'C':
        return face + ' ' + body
    cut = max(2, len(body) // 2)
    return body[:cut] + (' ' + face + ' ' if variant == 'D' else face) + body[cut:]


def entry_script(faces, rng, seed, list_sha256, atoms_sha256):
    bodies = list(BODY_LINES)
    rng.shuffle(bodies)
    body_iter = iter(bodies * 4)
    screens, face_index = [], 0
    for number, pattern in enumerate(STANDALONE_PATTERNS, start=1):
        lines = []
        for position, tag in enumerate(pattern, start=1):
            if tag == 'F':
                face = faces[face_index]
                face_index += 1
                lines.append({'position': position, 'tag': 'face', 'text': face['text'], 'face_index': face_index,
                              'face': face['text'], 'core_partition': face['core_partition'], 'source': face['source']})
            else:
                lines.append({'position': position, 'tag': 'body', 'text': next(body_iter)})
        screens.append({'file': f'screens/shot-{number:03d}.png', 'section': 'standalone', 'lines': lines})
    if face_index != len(faces):
        raise ValueError('Standalone patterns must consume every face exactly once')
    embedded = list(range(len(faces)))
    rng.shuffle(embedded)
    embedded = embedded[:MIXED_SCREENS * MIXED_FACES]
    variants = list(VARIANTS)
    for number in range(MIXED_SCREENS):
        lines = []
        slots = sorted(rng.sample(range(MIXED_LINES), MIXED_FACES))
        for position in range(1, MIXED_LINES + 1):
            body = next(body_iter)
            if position - 1 in slots:
                index = embedded[number * MIXED_FACES + slots.index(position - 1)]
                variant = variants[(number * MIXED_FACES + slots.index(position - 1)) % len(variants)]
                face = faces[index]['text']
                lines.append({'position': position, 'tag': 'mixed', 'text': embed(body, face, variant), 'face_index': index + 1,
                              'face': face, 'variant': variant, 'core_partition': faces[index]['core_partition']})
            else:
                lines.append({'position': position, 'tag': 'body', 'text': body})
        screens.append({'file': f'screens/shot-{len(STANDALONE_PATTERNS) + number + 1:03d}.png', 'section': 'mixed', 'lines': lines})
    return {'schema': SCRIPT_SCHEMA, 'generated_utc': utc(), 'seed': seed, 'input_list_sha256': list_sha256,
            'atoms_sha256': atoms_sha256, 'variants': VARIANTS, 'label_rule': 'typed via the input method; truth is the stored text copied into typed/',
            'screens': screens}


def script_text(script):
    out = ['# 已知输入录入脚本 known-input-2（用输入法逐行键入到目标界面后截图；不要粘贴）', '',
           f"生成时间 {script['generated_utc']}；输入清单 sha256 {script['input_list_sha256'][:16]}…；种子 {script['seed']}。",
           '每个区块对应一张截图；区块内的行按顺序录入，每行一条记录。行首编号与 [face]/[body]/[mixed] 标记不要录入。',
           '录入后把界面存储的文本按同样顺序复制到 typed/shot-NNN.txt（每条一行）。', '']
    for screen in script['screens']:
        title = '独立颜文字行' if screen['section'] == 'standalone' else '正文内嵌颜文字行'
        out.append(f"===== {Path(screen['file']).name}（{title}，{len(screen['lines'])} 行）=====")
        for line in screen['lines']:
            note = f"    ← F{line['face_index']:02d} {line['core_partition']}" if line['tag'] != 'body' else ''
            note += f" 变体{line['variant']}" if line['tag'] == 'mixed' else ''
            out.append(f"{line['position']:2d} [{line['tag']:5s}] {line['text']}{note}")
        out.append('')
    return '\n'.join(out) + '\n'


def build(output, fonts, seed, exclude=(), count=50):
    """Create a fresh known-input directory with the list, entry script, expected texts and templates."""
    root = Path(output)
    if root.exists():
        raise ValueError('Fresh known-input output directory required')
    if count != COMMON_COUNT + sum(n for _, n in PARTITION_PLAN) or count != sum(p.count('F') for p in STANDALONE_PATTERNS):
        raise ValueError('Face count must match the partition plan and the standalone patterns')
    renderer = MixedRenderer(fonts)
    atoms = kg.default_atoms()
    allowed, excluded_atoms = renderable_atoms(renderer, atoms)
    rng = random.Random(seed)
    def renderable(text):
        try:
            renderer.shape(text)
        except Unsupported:
            return False
        return True
    faces = compose_faces(rng, atoms, allowed, excluded_texts(exclude), renderable=renderable)
    root.mkdir(parents=True)
    listing = {'schema': LIST_SCHEMA, 'atoms_sha256': atoms.sha256, 'count': len(faces), 'seed': seed, 'usage': USAGE,
               'excluded_lists': [str(Path(p).resolve()) for p in exclude], 'excluded_atoms': excluded_atoms,
               'items': [{k: face[k] for k in ('text', 'source', 'core_partition', 'core_digest', 'structure')} for face in faces],
               'note': 'Type each line into the target UI with your usual input method; screenshots and stored text stay in this ignored directory.'}
    _write(root / 'input-list.json', listing)
    _write(root / 'input-list.txt', '\n'.join(face['text'] for face in faces) + '\n')
    script = entry_script(faces, rng, seed, _sha(root / 'input-list.json'), atoms.sha256)
    _write(root / 'entry-script.json', script)
    _write(root / 'entry-script.txt', script_text(script))
    for screen in script['screens']:
        name = Path(screen['file']).stem + '.txt'
        _write(root / 'expected' / name, '\n'.join(line['text'] for line in screen['lines']) + '\n')
    (root / 'typed').mkdir()
    (root / 'screens').mkdir()
    _write(root / 'typed' / 'README.txt', '每张截图一个文件 shot-NNN.txt：界面实际存储的文本，每条记录一行，顺序与录入脚本一致。\n')
    _write(root / 'manifest-template.json', {'schema': rs.MANIFEST_SCHEMA, 'usage': USAGE, 'environment': '<界面名称 / 字体设置 / 显示缩放 / 分辨率>',
                                             'label_origin': 'typed-via-ime; truth copied from stored record text', 'input_list_sha256': script['input_list_sha256'],
                                             'screens': [{'file': 'screens/shot-001.png', 'lines': [{'kind': 'body', 'text': '<stored text>'}]}]})
    _write(root / 'README-known-input.md', README)
    return {'root': str(root), 'faces': len(faces), 'screens': len(script['screens']),
            'embedded': sum(line['tag'] == 'mixed' for screen in script['screens'] for line in screen['lines']),
            'input_list_sha256': script['input_list_sha256'], 'excluded_atoms': len(excluded_atoms)}


def embedded_faces(text, atoms=None):
    """Maximal non-overlapping substrings of ``text`` that are complete expressions, left to right."""
    atoms = kg.default_atoms() if atoms is None else atoms
    found, start, length = [], 0, len(text)
    while start < length:
        hit = None
        for end in range(min(length, start + atoms.limits['maximum_length']), start + atoms.limits['minimum_length'] - 1, -1):
            piece = text[start:end]
            if piece[0].isspace() or piece[-1].isspace():
                continue
            if kg.parse_face(piece, atoms) is not None:
                hit = end
                break
        if hit is None:
            start += 1
        else:
            found.append(text[start:hit])
            start = hit
    return found


def retired_lists(rounds=None):
    """Map input-list SHA256 to the retirement record of every list retired inside a registered round root."""
    found = {}
    for profile in (rp.ROUNDS if rounds is None else rounds):
        root = Path(profile.root)
        if not root.is_dir():
            continue
        for path in sorted(root.glob(RETIREMENT_PATTERN)):
            value = json.loads(path.read_text(encoding='utf-8'))
            if value.get('schema') == RETIREMENT_SCHEMA:
                found[value['input_list_sha256']] = path
    return found


def retire(input_dir, output, reason, change):
    """Record that an entry-script list was never typed, screenshotted or evaluated, so it can never be used."""
    root = Path(input_dir)
    script = json.loads((root / 'entry-script.json').read_text(encoding='utf-8'))
    listing_sha = _sha(root / 'input-list.json')
    if script.get('schema') != SCRIPT_SCHEMA or listing_sha != script['input_list_sha256']:
        raise ValueError('Entry script and input list do not belong together')
    screens = sorted(p.name for p in (root / 'screens').glob('*') if p.is_file()) if (root / 'screens').is_dir() else []
    typed = sorted(p.name for p in (root / 'typed').glob('shot-*.txt')) if (root / 'typed').is_dir() else []
    if screens or typed or (root / 'manifest.json').exists():
        raise ValueError('Only an unused known-input list can be retired')
    # Only a record directly in a registered round root is found by retired_lists(); refuse anything else.
    target = Path(output).resolve()
    if (not any(target.parent == Path(profile.root).resolve() for profile in rp.ROUNDS)
            or not target.match(RETIREMENT_PATTERN)):
        raise ValueError('Retirement records live directly in a registered round root as ' + RETIREMENT_PATTERN)
    # Without stored text there is no manifest, so no known-input first-use receipt can exist for this list.
    record = {'schema': RETIREMENT_SCHEMA, 'created_utc': utc(), 'change': change, 'input_dir': str(root.resolve()),
              'input_list_sha256': listing_sha, 'entry_script_sha256': _sha(root / 'entry-script.json'),
              'faces': len(json.loads((root / 'input-list.json').read_text(encoding='utf-8'))['items']),
              'screens': 0, 'typed_files': 0, 'manifest_present': False, 'first_use_receipt': None, 'reason': reason}
    _write(output, record)
    return record


def build_manifest(input_dir, environment, output, delivery=None):
    """Assemble the evaluation manifest from the entry script and the user's stored-text files."""
    root = Path(input_dir)
    script = json.loads((root / 'entry-script.json').read_text(encoding='utf-8'))
    if script.get('schema') != SCRIPT_SCHEMA:
        raise ValueError('Unknown entry script schema')
    listing_sha = _sha(root / 'input-list.json')
    if listing_sha != script['input_list_sha256']:
        raise ValueError('Input list changed after the entry script was generated')
    retired = retired_lists().get(listing_sha)
    if retired is not None:
        raise ValueError('Known-input list is retired; see ' + str(retired))
    atoms = kg.default_atoms()
    if atoms.sha256 != script['atoms_sha256']:
        raise ValueError('Atom table changed after the entry script was generated')
    screens, differences, missing_faces = [], 0, 0
    for screen in script['screens']:
        typed_path = root / 'typed' / (Path(screen['file']).stem + '.txt')
        typed = typed_path.read_text(encoding='utf-8').split('\n')
        if typed and typed[-1] == '':
            typed = typed[:-1]
        if len(typed) != len(screen['lines']) or any(not line.strip() for line in typed):
            raise ValueError('Stored text for ' + screen['file'] + ' must hold exactly one nonempty line per entry')
        if not (root / screen['file']).is_file():
            raise ValueError('Screenshot missing: ' + screen['file'])
        lines = []
        for entry, stored in zip(screen['lines'], typed):
            stored = stored.rstrip('\r')
            if stored != entry['text']:
                differences += 1
            if entry['tag'] == 'face':
                lines.append({'kind': 'face', 'text': stored, 'script_text': entry['text'], 'core_partition': entry['core_partition']})
            elif entry['tag'] == 'mixed':
                faces = embedded_faces(stored, atoms)
                if not faces:
                    missing_faces += 1
                lines.append({'kind': 'body', 'text': stored, 'faces': faces, 'script_text': entry['text'], 'script_face': entry['face'], 'variant': entry['variant']})
            else:
                lines.append({'kind': 'body', 'text': stored, 'script_text': entry['text']})
        screens.append({'file': screen['file'], 'lines': lines})
    manifest = {'schema': rs.MANIFEST_SCHEMA, 'usage': USAGE, 'environment': environment,
                'label_origin': 'typed-via-ime; truth copied from stored record text', 'input_list_sha256': listing_sha,
                'entry_script_sha256': _sha(root / 'entry-script.json'), 'typed_differences': differences,
                'mixed_lines_without_detected_face': missing_faces, 'delivery': delivery, 'screens': screens}
    _write(output, manifest)
    return {'file': str(output), 'screens': len(screens), 'typed_differences': differences, 'mixed_lines_without_detected_face': missing_faces}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    gen = sub.add_parser('generate')
    gen.add_argument('--output', required=True)
    gen.add_argument('--fonts', required=True)
    gen.add_argument('--seed', type=int, required=True)
    gen.add_argument('--exclude', action='append', default=[], help='Earlier input-list.json whose faces must not recur')
    man = sub.add_parser('manifest')
    man.add_argument('--input-dir', required=True)
    man.add_argument('--environment', required=True)
    man.add_argument('--output', required=True)
    man.add_argument('--delivery')
    ret = sub.add_parser('retire')
    ret.add_argument('--input-dir', required=True)
    ret.add_argument('--output', required=True)
    ret.add_argument('--reason', required=True)
    ret.add_argument('--change', required=True)
    args = parser.parse_args(argv)
    if args.action == 'generate':
        print(json.dumps(build(args.output, args.fonts, args.seed, args.exclude), ensure_ascii=False))
    elif args.action == 'retire':
        print(json.dumps(retire(args.input_dir, args.output, args.reason, args.change), ensure_ascii=False))
    else:
        print(json.dumps(build_manifest(args.input_dir, args.environment, args.output, args.delivery), ensure_ascii=False))


if __name__ == '__main__':
    main()
