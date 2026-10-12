"""Real in-flight version switch: synthetic large image, no private samples."""
import argparse,base64,io,json,sys,threading,time
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont
from verify_live import RunningServer,wait_until
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from localocr.bundles import activate_bundle,load_active

def main():
    p=argparse.ArgumentParser();p.add_argument('--store',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--port',type=int,default=18765);args=p.parse_args()
    store,output=args.store.resolve(),args.output.resolve();output.mkdir(parents=True,exist_ok=True)
    assert load_active(store).version=='baseline-1'
    image=Image.new('RGB',(3000,4500),'white');draw=ImageDraw.Draw(image);font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',90)
    for y in range(50,4400,150):
        draw.text((50,y),'HELLO OCR 123',fill='black',font=font)
        draw.text((1500,y),'LOCAL CPU 456',fill='black',font=font)
    stream=io.BytesIO();image.save(stream,'PNG')
    body={'model':'baseline-1','messages':[{'role':'user','content':[{'type':'image_url','image_url':{'url':'data:image/png;base64,'+base64.b64encode(stream.getvalue()).decode()}}]}]}
    server=RunningServer(store,output,args.port);out={}
    def recognize():
        try: out['response']=server.client.post('/v1/chat/completions',json=body,headers=server.headers)
        except Exception as error: out['error']=repr(error)
    thread=threading.Thread(target=recognize)
    try:
        thread.start()
        wait_until(lambda:[c for c in server.children() if any(a.endswith('local-ai-ocr-worker.py') for a in c.cmdline())])
        start=time.monotonic();activate_bundle(store,'baseline-2');overlap=thread.is_alive()
        assert overlap,'Synthetic request finished before activation; no in-flight switch proof'
        assert load_active(store).version=='baseline-2'
        thread.join(timeout=95);assert not thread.is_alive() and 'error' not in out,out
        response=out['response'];assert response.status_code==200,response.text
        result=json.loads(response.json()['choices'][0]['message']['content'])
        assert result['model_version']=='baseline-1' and result['regions']
        assert server.client.get('/v1/models',headers=server.headers).json()['data'][0]['id']=='baseline-2'
        report={'actual_inflight_switch':True,'old_request_model':result['model_version'],'new_active_model':'baseline-2','activation_seconds':time.monotonic()-start,'image_pixels':3000*4500,'old_result_regions':len(result['regions']),'scope':'synthetic CPU lifecycle evidence, no quality promotion'}
        (output/'snapshot-verification.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report,indent=2))
    finally:
        server.close();thread.join(timeout=10);activate_bundle(store,'baseline-1')
if __name__=='__main__':main()
