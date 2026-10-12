"""Original CC0 vocabulary and structural grammars, assigned before sampling."""
import hashlib
import json
import random

CN_WORDS = '晴川 星河 山林 湖水 云朵 微风 晨光 夜色 清泉 松树 花园 石桥 小路 天空 海洋 河岸 草地 麦田 竹林 雨滴 雪花 火焰 土地 沙滩 远方 家园 学校 书本 纸张 笔记 画面 音乐 灯光 桌面 窗口 房间 大门 时钟 时间 日期 数字 文字 颜色 形状 圆点 直线 曲线 方格 左边 右边 上方 下方 前面 后面 开始 结束 保存 读取 选择 更新'.split()
EN_WORDS = 'river garden cloud forest light paper table window clock color shape circle square line water stone grass flower rain snow wind sun moon star road bridge home room door book note music image number letter time date start finish save read select update clear small large left right above below'.split()
# Every expression is a different whole AST. Shared token pools/glyphs are disclosed.
GRAMMARS = {
 'train': {'cn':'{cn1} {serial} {cn2}', 'en':'{en1} {case} {en2} {serial}', 'number':'{serial}.{decimal} / 2042-03-11', 'punctuation':'({serial}) {symbols} #', 'face':'{arm}({eye}{space}{mouth}{space}{eye}) {serial}', 'combining':'cafe\u0301 {serial} o\u0304', 'emoji-zwj':'{emoji4}'},
 'dev': {'cn':'{cn2}：[{serial}] {cn1}', 'en':'{en2}[{serial}]={case}:{en1}', 'number':'2043/11/02 {serial}-{decimal}', 'punctuation':'{symbols}[{serial}]~', 'face':'{serial} [{eye}{mouth}{eye}]{arm}', 'combining':'o\u0304 [{serial}] cafe\u0301', 'emoji-zwj':'{emoji2}  {emoji2b}'},
 'heldout': {'cn':'{cn1}/{cn2}/{serial}', 'en':'{serial}:<{en1}/{en2}> {case}', 'number':'{decimal}:{serial} 2044.07.09', 'punctuation':'<{symbols}>:{serial}!', 'face':'{arm}{{{eye}{space}{mouth}{space}{eye}}}:{serial}', 'combining':'{serial}/e\u0301-o\u0304', 'emoji-zwj':'Icon[{emoji2}]={emoji3}'},
}

def text_hash(text):
 return hashlib.sha256(text.encode('utf-8')).hexdigest()

def grammar_family(split, recipe):
 # Hash actual structural expression, not its split or a renamed identifier.
 return 'grammar:' + text_hash(GRAMMARS[split][recipe])

def samples(split,count,seed,excluded=()):
 if split not in GRAMMARS:raise ValueError('Unknown split')
 rng=random.Random(seed);seen=set(excluded)
 emoji=['😀','🌈','🐱','👩\u200d💻','👨\u200d🔬','👩🏽\u200d🚀','🧑\u200d💻','🏳️\u200d🌈']
 for index in range(count):
  bucket=index%20
  kind='plain' if bucket<10 else 'punctuation' if bucket<14 else 'face' if bucket<18 else 'emoji' if bucket==18 else 'plain'
  subtype='combining' if bucket==19 else 'emoji-zwj' if bucket==18 else kind
  recipe=('cn','en','number')[index%3] if subtype=='plain' else subtype
  for attempt in range(100000):
   args={'cn1':rng.choice(CN_WORDS),'cn2':rng.choice(CN_WORDS),'en1':rng.choice(EN_WORDS),'en2':rng.choice(EN_WORDS),'case':rng.choice(['Orbit','ORBIT','orbit','OrBiT']),'serial':rng.randrange(10000000,99999999),'decimal':f'{index%100:02d}','arm':rng.choice(['つ','╯','ノ','']),'eye':rng.choice(['ω','Д','°','^','•','o','T','O']),'mouth':rng.choice(['▽','_','∀','益','3','皿','v','V']),'space':rng.choice(['',' ','  ']),'symbols':rng.choice(['!?;:','<->/\\','___===','───+-'])}
   for name,length in [('emoji4',4),('emoji3',3),('emoji2',2),('emoji2b',2)]:args[name]=''.join(rng.choice(emoji) for _ in range(length))
   text=GRAMMARS[split][recipe].format(**args)
   if kind!='emoji':text=' '*rng.randrange(4)+text.replace(' ',' '*rng.randrange(1,5))+' '*rng.randrange(4)
   digest=text_hash(text)
   if digest not in seen:
    seen.add(digest)
    yield {'text':text,'text_sha256':digest,'kind':kind,'subtype':subtype,'family':grammar_family(split,recipe),'recipe':recipe,'group_id':'text:'+digest}
    break
  else:raise ValueError('Original grammar exhausted')
