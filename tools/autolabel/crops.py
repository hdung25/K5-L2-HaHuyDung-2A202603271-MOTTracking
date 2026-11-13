"""Crop vùng cố định qua nhiều frame (soát chi tiết). python crops.py x1 y1 x2 y2 f1,f2,... out.jpg [P..]"""
import sys, json
from PIL import Image, ImageDraw
x1,y1,x2,y2=map(int,sys.argv[1:5]); fr=[int(v) for v in sys.argv[5].split(',')]; out=sys.argv[6]
layers=sys.argv[7:]
D={}
cols=6; cw=300; ch=int(cw*(y2-y1)/(x2-x1))
s=Image.new('RGB',(cw*cols,ch*((len(fr)+cols-1)//cols)))
colors=['lime','red','cyan','magenta','orange']
for i,f in enumerate(fr):
    im=Image.open(f'../../.lab-local/mot17-09/{f+1:06d}.jpg').convert('RGB'); dr=ImageDraw.Draw(im)
    for j,p in enumerate(layers):
        fn,pp=p.split(':') if ':' in p else ('dense_raw.json',p); D.setdefault(fn,json.load(open('out/'+fn))); b=D[fn][pp].get(str(f))
        if b: dr.rectangle(b[:4],outline=colors[j],width=2)
    im=im.crop((x1,y1,x2,y2)).resize((cw,ch)); ImageDraw.Draw(im).text((4,4),str(f),fill='yellow')
    s.paste(im,((i%cols)*cw,(i//cols)*ch))
s.save('out/'+out)
