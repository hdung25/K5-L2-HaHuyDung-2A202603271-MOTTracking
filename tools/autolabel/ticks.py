"""Crop có thước pixel (vạch mỗi 20px, số mỗi 100px) để đặt box tay chính xác."""
import sys
from PIL import Image, ImageDraw
x1,y1,x2,y2=map(int,sys.argv[1:5]); fr=[int(v) for v in sys.argv[5].split(',')]; out=sys.argv[6]; sc=float(sys.argv[7]) if len(sys.argv)>7 else 2
cols=int(sys.argv[8]) if len(sys.argv)>8 else 4
cw,ch=int((x2-x1)*sc),int((y2-y1)*sc)
s=Image.new('RGB',(cw*cols,ch*((len(fr)+cols-1)//cols)))
for i,f in enumerate(fr):
    im=Image.open(f'../../.lab-local/mot17-09/{f+1:06d}.jpg').convert('RGB').crop((x1,y1,x2,y2)).resize((cw,ch))
    d=ImageDraw.Draw(im)
    for x in range((x1//20+1)*20,x2,20):
        X=(x-x1)*sc; d.line([(X,0),(X,8 if x%100 else 16)],fill='yellow'); d.line([(X,ch-8 if x%100 else ch-16),(X,ch)],fill='yellow')
        if x%100==0: d.text((X+2,18),str(x),fill='yellow')
    for y in range((y1//20+1)*20,y2,20):
        Y=(y-y1)*sc; d.line([(0,Y),(8 if y%100 else 16,Y)],fill='cyan'); d.line([(cw-(8 if y%100 else 16),Y),(cw,Y)],fill='cyan')
        if y%100==0: d.text((18,Y+2),str(y),fill='cyan')
    d.text((cw-40,4),str(f),fill='white')
    s.paste(im,((i%cols)*cw,(i//cols)*ch))
s.save('out/'+out)
