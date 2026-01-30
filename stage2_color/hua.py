"""
Author:wanyii
time:2023/10
"""
from PIL import Image, ImageDraw, ImageFont
import math, os

base_path = r"Y:\teacherguo\DDColor-master_create01\assets\landscape_dataset\gray_12.jpg"
img = Image.open(base_path).convert("RGBA")
w,h = img.size

pad_top = 20
pad_bottom = 320
canvas = Image.new("RGBA", (w, h + pad_top + pad_bottom), (255,255,255,255))
canvas.paste(img, (0, pad_top))
draw = ImageDraw.Draw(canvas)

try:
    font_title = ImageFont.truetype(r"Y:\teacherguo\DDColor-master_create01\assets\landscape_dataset\gray_12.jpg", 22)
    font = ImageFont.truetype(r"Y:\teacherguo\DDColor-master_create01\assets\landscape_dataset\gray_12.jpg", 14)
    font_small = ImageFont.truetype(r"Y:\teacherguo\DDColor-master_create01\assets\landscape_dataset\gray_12.jpg", 12)
    font_bold = ImageFont.truetype(r"Y:\teacherguo\DDColor-master_create01\assets\landscape_dataset\gray_12.jpg", 14)
except:
    font_title = ImageFont.load_default()
    font = ImageFont.load_default()
    font_small = ImageFont.load_default()
    font_bold = ImageFont.load_default()

def rounded_rect(xy, radius=12, fill=(240,248,255,255), outline=(60,90,120,255), width=2):
    x0,y0,x1,y1 = xy
    draw.rounded_rectangle([x0,y0,x1,y1], radius=radius, fill=fill, outline=outline, width=width)

def arrow(p0, p1, color=(40,60,90,255), width=3, head=10):
    draw.line([p0,p1], fill=color, width=width)
    dx = p1[0]-p0[0]; dy = p1[1]-p0[1]
    ang = math.atan2(dy, dx)
    x2,y2 = p1
    left = (x2 - head*math.cos(ang) + head*0.6*math.sin(ang),
            y2 - head*math.sin(ang) - head*0.6*math.cos(ang))
    right = (x2 - head*math.cos(ang) - head*0.6*math.sin(ang),
             y2 - head*math.sin(ang) + head*0.6*math.cos(ang))
    draw.polygon([p1, left, right], fill=color)

def dashed_arrow(p0,p1, dash=10, gap=6, color=(160,120,70,255), width=3, head=10):
    x0,y0 = p0; x1,y1 = p1
    dx,dy = x1-x0, y1-y0
    dist = math.hypot(dx,dy)
    if dist == 0: return
    ux,uy = dx/dist, dy/dist
    n = int(dist // (dash+gap))
    for i in range(n):
        a = i*(dash+gap)
        b = a + dash
        pA = (x0+ux*a, y0+uy*a)
        pB = (x0+ux*b, y0+uy*b)
        draw.line([pA,pB], fill=color, width=width)
    arrow((x1-ux*head*2, y1-uy*head*2), (x1,y1), color=color, width=width, head=head)

def text_single(text, xy, f=font, fill=(0,0,0,255), anchor=None):
    draw.text(xy, text, font=f, fill=fill, anchor=anchor)

def text_multiline(text, xy, f=font_small, fill=(0,0,0,255), spacing=3):
    draw.multiline_text(xy, text, font=f, fill=fill, spacing=spacing)

# Title
text_single("Uncertainty-Aware Colorization (Training Diagram)", (w//2, 12), f=font_title, fill=(20,40,60,255), anchor="mm")

y0 = pad_top + h + 20

loss_box = (60, y0, 500, y0+200)
rounded_rect(loss_box, fill=(245,250,255,255), outline=(70,110,150,255))
text_single("Losses (output_ab & uncertainty)", (loss_box[0]+14, loss_box[1]+16), f=font_bold, fill=(30,50,70,255))
loss_lines = [
    "• Reconstruction: L1(pred_ab, GT_ab)",
    "• NLL (σ²): 0.5*(log σ² + err²/(σ²))",
    "• Uncertainty reg: ||log(σ²)||²",
    "• Calibration (ECE-style bins)",
    "• (Optional) Ensemble diversity/consistency"
]
for i, t in enumerate(loss_lines):
    text_single(t, (loss_box[0]+18, loss_box[1]+48+i*26), f=font, fill=(10,30,50,255))

warm_box = (520, y0, 850, y0+85)
rounded_rect(warm_box, fill=(255,250,240,255), outline=(160,120,70,255))
text_single("Uncertainty warmup", (warm_box[0]+14, warm_box[1]+16), f=font_bold, fill=(90,60,20,255))
text_multiline("scale = min(1, iter / warmup_iter)\napply to uncertainty-related losses", (warm_box[0]+14, warm_box[1]+42), f=font_small, fill=(90,60,20,255))

opt_box = (520, y0+110, 850, y0+200)
rounded_rect(opt_box, fill=(245,245,245,255), outline=(120,120,120,255))
text_single("Optimizer / Backprop", (opt_box[0]+14, opt_box[1]+16), f=font_bold, fill=(40,40,40,255))
text_multiline("1) Update G (net_g)\n2) Update D (net_d, GAN)\n3) EMA update (optional)", (opt_box[0]+14, opt_box[1]+44), f=font_small, fill=(40,40,40,255))

thr_box = (60, y0+215, 500, y0+295)
rounded_rect(thr_box, fill=(240,255,245,255), outline=(80,140,100,255))
text_single("Dynamic uncertainty threshold τ", (thr_box[0]+14, thr_box[1]+16), f=font_bold, fill=(20,70,40,255))
text_multiline("high_ratio = mean(uncertainty > τ)\nadjust τ slowly after warmup", (thr_box[0]+14, thr_box[1]+44), f=font_small, fill=(20,70,40,255))

log_box = (520, y0+215, 850, y0+295)
rounded_rect(log_box, fill=(240,248,255,255), outline=(70,110,150,255))
text_single("Logging / Snapshots / Validation", (log_box[0]+14, log_box[1]+16), f=font_bold, fill=(30,50,70,255))
text_multiline("• loss components\n• uncertainty mean / max / ratio\n• save images + uncertainty heatmaps\n• periodic validation", (log_box[0]+14, log_box[1]+44), f=font_small, fill=(10,30,50,255))

# Tap points from the original architecture
tap_pred = (610, pad_top + 150)
tap_unc = (610, pad_top + 195)
loss_in1 = (loss_box[0]+10, loss_box[1]+110)
loss_in2 = (loss_box[0]+10, loss_box[1]+140)
arrow(tap_pred, loss_in1, color=(40,80,140,255))
arrow(tap_unc, loss_in2, color=(140,80,40,255))
text_single("pred_ab (μ_ab)", (tap_pred[0]-6, tap_pred[1]-8), f=font_small, fill=(40,80,140,255), anchor="rb")
text_single("uncertainty (σ²) / ensemble", (tap_unc[0]-6, tap_unc[1]-8), f=font_small, fill=(140,80,40,255), anchor="rb")

arrow((loss_box[2], loss_box[1]+130), (opt_box[0], opt_box[1]+60), color=(60,60,60,255))
text_single("total loss", (opt_box[0]+10, opt_box[1]+52), f=font_small, fill=(60,60,60,255))

dashed_arrow((warm_box[0], warm_box[1]+60), (loss_box[2]-10, loss_box[1]+60), color=(160,120,70,255))
text_single("scale", (loss_box[2]-14, loss_box[1]+52), f=font_small, fill=(160,120,70,255), anchor="rt")

dashed_arrow((loss_box[0]+260, loss_box[3]), (thr_box[0]+260, thr_box[1]), color=(80,140,100,255))
dashed_arrow((thr_box[2], thr_box[1]+40), (log_box[0], log_box[1]+40), color=(80,140,100,255))

arrow((770, pad_top + 145), (log_box[0]+10, log_box[1]+170), color=(70,110,150,255))
text_single("snapshots/val", (log_box[0]+18, log_box[1]+160), f=font_small, fill=(70,110,150,255))

text_multiline("Uncertainty head\n(returns σ² and/or ensemble preds)", (620, pad_top+218), f=font_small, fill=(140,80,40,255))

out_path = r"Y:\teacherguo\DDColor-master_create01\assets\landscape_dataset\training_diagram_uncertainty_fixed.png"
canvas.convert("RGB").save(out_path, "PNG")
out_path
