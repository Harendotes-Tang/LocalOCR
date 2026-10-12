"""Offline original data generation, full-canvas Unicode and sealed split hashes."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import math
import regex
import shutil
import time

from PIL import Image, ImageDraw
from tools.OcrDatasets.render import Renderer, Unsupported
from tools.OcrDatasets.validate import selection_use
from .corpus import samples, text_hash, GRAMMARS, grammar_family, CN_WORDS, EN_WORDS

SPLITS = ('train', 'dev', 'heldout')
ARTIFACTS = Path(__file__).resolve().parents[2] / 'artifacts'


class MixedRenderer(Renderer):
    """Select color at exact grapheme boundaries, retaining full line advance."""
    def shape(self, text, emoji=False):
        segments=[]
        for cluster in regex.findall(r'\X',text):
            color=bool(regex.search(r'\p{Extended_Pictographic}|\p{Regional_Indicator}',cluster))
            if segments and segments[-1][1]==color:
                segments[-1]=(segments[-1][0]+cluster,color)
            else:segments.append((cluster,color))
        return [run for segment,color in segments for run in Renderer.shape(self,segment,color)]


def negative_image(split,index,rng):
    width=64+index%257;height=32+index//257
    background=(rng.randrange(180,256),rng.randrange(180,256),rng.randrange(180,256))
    image=Image.new('RGB',(width,height),background);draw=ImageDraw.Draw(image)
    if split=='dev':
        for y in range(height):
            draw.line((0,y,width-1,y),fill=tuple(max(0,c-y) for c in background))
    elif split=='heldout':
        for y in range(height):
            for x in range(width):
                delta=1+(x*17+y*13)%4
                image.putpixel((x,y),tuple(c-delta for c in background))
    kind='blank' if index%2==0 else 'graphic'
    background_recipe={'train':'solid-rgb','dev':'vertical-linear-gradient','heldout':'faint-modular-rgb-texture'}[split]
    geometry={'train':'ellipse','dev':'triangle','heldout':'sine-wave'}[split]
    if kind=='graphic':
        if split=='train':draw.ellipse((3,3,width-4,height-4),outline=(20,40,70),width=2)
        elif split=='dev':draw.polygon([(width//2,3),(width-4,height-4),(3,height-4)],outline=(20,40,70),width=2)
        else:draw.line([(x,round(height/2+height/4*math.sin(x*math.pi*4/width))) for x in range(3,width-3)],fill=(20,40,70),width=2)
    structure={'background':background_recipe,'geometry':geometry if kind=='graphic' else None}
    family='negative-grammar:'+text_hash(json.dumps(structure,sort_keys=True))
    return image,kind,family,{'background':background,'structure':structure}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + '\n', encoding='utf-8')


def pixels(image):
    rgb = image.convert('RGB')
    return hashlib.sha256(rgb.width.to_bytes(8,'big') + rgb.height.to_bytes(8,'big') + rgb.tobytes()).hexdigest()


def disk_gate(path):
    if shutil.disk_usage(path).free < 8 * 1024**3:
        raise ValueError('Disk headroom below 8GiB')


def output_root(path):
    if str(path).startswith(('\\\\', '//')):
        raise ValueError('Network paths forbidden')
    root = Path(path).resolve()
    if not root.is_relative_to(ARTIFACTS.resolve()):
        raise ValueError('Output must remain in LocalOCR artifacts')
    return root


def validate(root):
    root = Path(root).resolve(); meta = read(root/'dataset.json')
    if meta.get('schema_version')!='localocr.pretrained.dataset.v1' or meta.get('synthetic_only') is not True:
        raise ValueError('Requires synthetic-only pretrained schema')
    texts, groups, families, hashes = {}, {}, {}, {}
    counts = {}; train_tokens = set()
    for split in SPLITS:
        info = meta['splits'][split]
        relative=Path(info['file'])
        if relative.is_absolute() or relative.drive or '..' in relative.parts:raise ValueError('Split path escapes dataset')
        path = root/relative
        if sha(path) != info['sha256']:
            raise ValueError('Split changed')
        rows = read(path)
        if len(rows) != info['count']:
            raise ValueError('Split count changed')
        for row in rows:
            if row.get('recipe') is not None and row['family'] != grammar_family(split,row['recipe']):
                raise ValueError('Actual grammar AST identity differs')
            if row['split'] != split or row['provenance'] != {'source':'original','license':'CC0-1.0'}:
                raise ValueError('Invalid split/provenance')
            if text_hash(row['text']) != row['text_sha256']:
                raise ValueError('Exact Unicode label changed')
            for value, inventory in [(row['group_id'],groups),(row['family'],families)]:
                if value in inventory and inventory[value] != split:
                    raise ValueError('Cross-split group/family leakage')
                inventory[value] = split
            if row['text']:
                if row['text_sha256'] in texts and texts[row['text_sha256']] != split:
                    raise ValueError('Cross-split text leakage')
                texts[row['text_sha256']] = split
            image_path = (root/row['image']).resolve()
            if not image_path.is_relative_to(root) or sha(image_path) != row['sha256']:
                raise ValueError('Image path/hash differs')
            with Image.open(image_path) as image:
                image.load(); pixel = pixels(image)
                if image.size != (row['width'],row['height']) or pixel != row['pixel_sha256']:
                    raise ValueError('Decoded pixels changed')
            if pixel in hashes and hashes[pixel] != split:
                raise ValueError('Cross-split pixels')
            hashes[pixel] = split
            if split == 'train': train_tokens.update(row['text'])
        positive = len({r['group_id'] for r in rows if r['text']})
        if positive != info['positive_groups']:
            raise ValueError('Positive groups changed')
        counts[split] = {'count':len(rows),'positive_groups':positive,'groups':len({r['group_id'] for r in rows})}
    if meta['train_vocabulary'] != sorted(train_tokens, key=ord):
        raise ValueError('Vocabulary must derive train only')
    binding = {'schema_version':'localocr.pretrained.freeze.v1','dataset_sha256':sha(root/'dataset.json'),'split_sha256':{s:meta['splits'][s]['sha256'] for s in SPLITS},'counts':counts,'license':'CC0-1.0'}
    if (root/'freeze.json').exists() and read(root/'freeze.json') != binding:
        raise ValueError('Freeze changed')
    if (root/'heldout-seal.json').exists():
        seal = read(root/'heldout-seal.json')
        if seal['heldout_sha256'] != binding['split_sha256']['heldout'] or seal['dataset_sha256'] != binding['dataset_sha256']:
            raise ValueError('Heldout seal differs')
    return binding


def generate(output, fonts, exclusions, counts=(10000,1000,2000), variants=(3,2,2), seed=20261011):
    root = output_root(output)
    if root.exists(): raise ValueError('Fresh output required')
    disk_gate(ARTIFACTS); renderer = MixedRenderer(fonts)
    excluded = set(read(exclusions)['label_sha256'])
    old = ARTIFACTS/'ocr-datasets-v1'
    old_hashes = {p.name:sha(p) for p in old.glob('*.json') if p.name != 'selection-use.json'}
    for name in ('train','dev','test','regression','acceptance'):
        excluded.update(r['label_sha256'] for r in read(old/(name+'.json')) if r['label'])
    previous=ARTIFACTS/'pretrained-round-20261011'/'data'
    previous_frozen_hashes={}
    if previous.exists():
        previous_frozen_hashes={p.name:sha(p) for p in previous.glob('*.json')}
        for split in SPLITS:excluded.update(r['text_sha256'] for r in read(previous/(split+'.json')) if r['text'])
    root.mkdir(parents=True); (root/'images').mkdir()
    started = time.perf_counter(); excluded_reasons = {}; split_info = {}; tokens = set(); seen_pixels = {}; positive_digests=set()
    for split_index, split in enumerate(SPLITS):
        rows = []; retained = 0; rng = random.Random(seed+split_index)
        stream = samples(split, counts[split_index]*2, seed+split_index, excluded | positive_digests)
        for item in stream:
            if retained >= counts[split_index]: break
            rendered = []
            try:
                for variant in range(variants[split_index]):
                    transform={'size':(24,32,20)[variant], 'dark':bool(variant % 2),'scale':1.0 if variant==0 else .85,'blur':0 if variant==0 else .3,'jpeg':None if variant==0 else 86}
                    image, region, used, evidence = renderer.render(item['text'], emoji=item['kind']=='emoji', **transform)
                    rendered.append((image,region,used,evidence,transform))
            except Unsupported as exc:
                key=str(exc); excluded_reasons[key]=excluded_reasons.get(key,0)+1
                continue
            # A group cannot partially survive rendering failure.
            if any(pixels(image) in seen_pixels and seen_pixels[pixels(image)] != split for image,*_ in rendered):
                excluded_reasons['cross-split-pixels']=excluded_reasons.get('cross-split-pixels',0)+1; continue
            for variant,(image,region,used,evidence,transform) in enumerate(rendered):
                identifier=f'{split}-{retained:06d}-{variant}'; path=root/'images'/(identifier+'.png'); image.save(path,compress_level=1)
                pixel=pixels(image);seen_pixels[pixel]=split
                rows.append(dict(item,id=identifier,split=split,image=path.relative_to(root).as_posix(),font_id=used,pixel_sha256=pixel,width=image.width,height=image.height,region=region,sha256=sha(path),provenance={'source':'original','license':'CC0-1.0'},rendering=evidence,transform=transform))
            retained+=1;positive_digests.add(item['text_sha256'])
            if split=='train':tokens.update(item['text'])
            if retained%100==0:disk_gate(root)
            if retained%1000==0:print(json.dumps({'split':split,'groups':retained,'elapsed_seconds':time.perf_counter()-started}),flush=True)
        if retained != counts[split_index]: raise ValueError('Insufficient renderable unique groups')
        for index in range(counts[split_index]//10):
            # Unique geometry/background recipe, independent from all positive labels.
            image,kind,family,negative_transform=negative_image(split,index,rng)
            width,height=image.size
            pixel=pixels(image)
            if pixel in seen_pixels: raise ValueError('Negative pixel duplicate')
            seen_pixels[pixel]=split;identifier=f'{split}-negative-{index:05d}';path=root/'images'/(identifier+'.png');image.save(path,compress_level=1)
            rows.append({'id':identifier,'split':split,'image':path.relative_to(root).as_posix(),'text':'','text_sha256':text_hash(''),'kind':kind,'subtype':kind,'group_id':identifier,'family':family,'font_id':[],'pixel_sha256':pixel,'width':width,'height':height,'region':[0,0,width,height],'sha256':sha(path),'provenance':{'source':'original','license':'CC0-1.0'},'rendering':{'text_present':False},'transform':negative_transform})
        write(root/(split+'.json'),rows);split_info[split]={'file':split+'.json','sha256':sha(root/(split+'.json')),'count':len(rows),'positive_groups':retained,'negative_groups':counts[split_index]//10,'variants':variants[split_index]}
    font_meta=[{k:v for k,v in f.items() if k not in ('file','license_file')} for f in renderer.fonts]
    write(root/'dataset.json',{'schema_version':'localocr.pretrained.dataset.v1','synthetic_only':True,'private_inputs_consumed':False,'distribution_scope':'finite original grammar task; shared vocabulary, emoji atomic glyphs and OFL fonts; not broad language distribution','grammar_family_ast':{grammar_family(split,recipe):expression for split,recipes in GRAMMARS.items() for recipe,expression in recipes.items()},'shared_vocabulary_counts':{'cn_words':len(CN_WORDS),'en_words':len(EN_WORDS)},'license':'CC0-1.0','seed':seed,'splits':split_info,'train_vocabulary':sorted(tokens,key=ord),'fonts':font_meta,'renderer':renderer.versions,'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()+Path(__file__).with_name('corpus.py').read_bytes()).hexdigest(),'excluded_exact_digest_count':len(excluded),'exclusion_reasons':excluded_reasons,'font_independence':False,'atomic_emoji_glyph_independence':False,'usage':{'train':'training','dev':'development-selection','heldout':'sealed-first-evaluation-after-candidate-selection'}})
    train_rows=read(root/'train.json')
    ctc=[len(row['text'])+sum(a==b for a,b in zip(row['text'],row['text'][1:])) for row in train_rows]
    write(root/'train-length-statistics.json',{'source':'train only; heldout not inspected for widths/vocabulary','train_sha256':split_info['train']['sha256'],'max_label_codepoints':max(len(r['text']) for r in train_rows),'max_ctc_required_steps':max(ctc),'count_ctc_required_over40':sum(n>40 for n in ctc),'count':len(train_rows)})
    binding=validate(root);write(root/'freeze.json',binding);write(root/'heldout-seal.json',{'schema_version':'localocr.heldout-seal.v1','dataset_sha256':binding['dataset_sha256'],'heldout_sha256':binding['split_sha256']['heldout'],'usage':'first evaluation only after candidate selection frozen','labels_used_for_vocabulary':False,'results_seen':False})
    for split in ('test','acceptance'):
        if split not in read(old/'selection-use.json')['splits'] if (old/'selection-use.json').exists() else True:
            selection_use(old,None,split,'All results seen and reserved for pretrained round development/regression; not fresh acceptance')
    assert all(sha(old/name)==digest for name,digest in old_hashes.items())
    assert all(sha(previous/name)==digest for name,digest in previous_frozen_hashes.items())
    write(root/'generation.json',{'schema_version':'localocr.generation-receipt.v1','elapsed_seconds':time.perf_counter()-started,'binding':binding,'old_frozen_hashes':old_hashes,'previous_round_data_frozen_hashes':previous_frozen_hashes,'previous_round_data_bytes_unchanged':True,'old_frozen_bytes_unchanged':True,'old_receipts':read(old/'selection-use.json'),'excluded_inventory_sha256':sha(exclusions),'private_labels_exported':False,'font_files_distributed':False,'disk_free_bytes':shutil.disk_usage(root).free})
    return binding


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['generate','validate']);p.add_argument('--dataset',required=True);p.add_argument('--fonts');p.add_argument('--exclusions');p.add_argument('--counts',nargs=3,type=int,default=[10000,1000,2000]);p.add_argument('--variants',nargs=3,type=int,default=[3,2,2]);a=p.parse_args()
    print(json.dumps(generate(a.dataset,a.fonts,a.exclusions,a.counts,a.variants) if a.action=='generate' else validate(a.dataset),indent=2))
if __name__=='__main__':main()
