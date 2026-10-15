"""Build a one-page executive summary PDF from the verified project results."""
from pathlib import Path
from reportlab.pdfgen import canvas
from reportlab.lib.colors import HexColor, Color
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.utils import simpleSplit

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / 'Executive_Summary.pdf'
# Prefer a system font; fall back to ReportLab's bundled Vera fonts (cross-platform).
import reportlab
system_regular = Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')
system_bold = Path('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf')
font_dir = Path(reportlab.__file__).resolve().parent / 'fonts'
regular = system_regular if system_regular.exists() else font_dir / 'Vera.ttf'
bold = system_bold if system_bold.exists() else font_dir / 'VeraBd.ttf'
pdfmetrics.registerFont(TTFont('DejaVu', str(regular)))
pdfmetrics.registerFont(TTFont('DejaVu-Bold', str(bold)))
W,H=595,842
c=canvas.Canvas(str(OUTPUT), pagesize=(W,H),pageCompression=1)
navy=HexColor('#142C43');teal=HexColor('#147A80');light=HexColor('#EDF4F7');slate=HexColor('#536B7D');line=HexColor('#DBE6ED')

def txt(x,y,s,size=10,bold=False,color=navy):
 c.setFillColor(color);c.setFont('DejaVu-Bold' if bold else 'DejaVu',size);c.drawString(x,y,s)

def paragraph(x,y,text,width,size=9.2,leading=14,color=slate):
 c.setFillColor(color);c.setFont('DejaVu',size)
 lines=simpleSplit(text,'DejaVu',size,width)
 for line_text in lines:
  c.drawString(x,y,line_text);y-=leading
 return y

c.setFillColor(navy);c.rect(0,H-182,W,182,fill=1,stroke=0)
txt(43,H-47,'ACDT PROJECT 1  /  EXECUTIVE SUMMARY',9,True,HexColor('#96D5D2'))
txt(43,H-95,'BEYOND THE RATINGS',25,True,HexColor('#FFFFFF'))
txt(43,H-124,'Do better-reviewed movies earn more worldwide revenue?',10.2,False,HexColor('#DAEAF1'))
txt(43,H-151,'2022-2025  |  385 films  |  Metacritic Metascore  |  Worldwide gross in USD',8.4,False,HexColor('#BDD3E0'))

y=H-208
txt(43,y,'RESEARCH QUESTION & DESIGN',11.6,True,teal);y-=24
y=paragraph(43,y,'We tested whether higher critic Metascore is associated with greater worldwide box-office gross after adjustment for production budget, release year, and primary genre. The 385-film sample includes Action, Comedy, Drama and Horror. Gross and budget enter the analysis on the natural-log scale.',W-86,9.0,14)
y-=13
c.setStrokeColor(line);c.line(43,y,W-43,y);y-=24
txt(43,y,'KEY FINDINGS',11.6,True,teal);y-=30

cards=[('Simple OLS','-9.1%','p = 0.205'),('Multiple OLS','+22.9%','p = 0.0008'),('RF-DML','+24.3%','p ≈ 0.000085')]
card_y=y-68
for idx,(head,value,foot) in enumerate(cards):
 x=43+idx*172
 c.setFillColor(light);c.roundRect(x,card_y,159,84,10,fill=1,stroke=0)
 txt(x+11,card_y+60,head,8.5,True,slate)
 txt(x+11,card_y+29,value,20,True,teal if idx>0 else navy)
 txt(x+11,card_y+12,foot,8.2,False,slate)
y=card_y-22
y=paragraph(43,y,'All percentage figures refer to the modeled difference in worldwide gross associated with a 10-point Metascore increase. The multiple OLS model has R² = 0.582; the genre-interaction Wald test has p = 0.0493. In the interaction model, the Action estimate is +45.8% per 10 points, but other genre slopes are imprecise.',W-86,8.8,13)
y-=14
c.setStrokeColor(line);c.line(43,y,W-43,y);y-=24
txt(43,y,'MACHINE LEARNING & SYSTEM MAPS',11.6,True,teal);y-=22
y=paragraph(43,y,'Random Forest performs slightly better than linear regression in 5-fold prediction validation (mean R² 0.585 vs 0.547). Budget drives most of the model’s predictive importance. Map A presents regression-supported associations; Map B adds plausible but unmeasured industry factors; Map C separates Random Forest prediction from RF-based DML adjustment.',W-86,8.8,13)
y-=15
c.setStrokeColor(line);c.line(43,y,W-43,y);y-=24
txt(43,y,'LIMITATIONS & DECISION IMPLICATIONS',11.6,True,teal);y-=22
y=paragraph(43,y,'The dataset is observational and selected for complete reported budgets, gross and ratings. The 2025 coverage is partial; genre coding and sample selection influence moderation results. Reviews, marketing, franchises, distribution and release strategy may jointly respond to film success. These results are associations, not demonstrated causal effects. Industry decisions should not rely on ratings alone.',W-86,8.8,13)
y-=20
c.setStrokeColor(line);c.line(43,y,W-43,y);y-=20
y=paragraph(43,y,'Source: group-supplied cleaned IMDb-derived Kaggle dataset (Raed Addala), ACDT_Project1_USD_Dataset_Updated notebook. Inference: heteroskedasticity-robust HC3 OLS; cross-fitted partially linear RF-DML; scikit-learn cross-validation. See analysis notebook and System_Map_Justification.md.',W-86,7.9,12)
assert y>28,(y,'PDF content might overlap footer')
txt(43,23,'ACDT / Data-Driven Decision Making  •  Study summary — not causal identification',7.4,False,slate)
c.setTitle('Beyond the Ratings | Executive Summary')
c.setAuthor('ACDT Project 1 Research Team')
c.save()
print(OUTPUT,y)
