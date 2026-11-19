# Face Recognition with OpenCV (Haar Cascade + LBPH)

โปรแกรมจดจำใบหน้าแบบ real-time ผ่านกล้องเว็บแคม เขียนด้วย Python และ OpenCV
ใช้ **Haar Cascade** ตรวจจับตำแหน่งใบหน้า และ **LBPH (Local Binary Patterns Histograms)** จำแนกว่าเป็นใคร

- กรอบสีเขียว = จำได้ (แสดงชื่อ + ค่า confidence)
- กรอบสีแดง = ไม่รู้จัก (`Unknown`)

> ⚠️ ค่า confidence ของ LBPH **ยิ่งต่ำยิ่งแม่น** (ตรงข้ามกับความเชื่อมั่นทั่วไป)

## การทำงาน

1. อ่านรูปจาก `dataset/<ชื่อคน>/` แล้วตรวจจับใบหน้าในแต่ละรูป
2. ครอปใบหน้า → resize 150×150 → ปรับ histogram → train โมเดล LBPH
3. เปิดกล้อง ตรวจจับใบหน้าทุกเฟรม แล้วทำนายชื่อ
4. กด `q` เพื่อปิดโปรแกรม

## ติดตั้ง

ต้องใช้ Python 3.8+

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

> ต้องติดตั้ง `opencv-contrib-python` (ไม่ใช่ `opencv-python` ธรรมดา) เพราะโมดูล `cv.face` อยู่ในแพ็กเกจ contrib
> และอย่าติดตั้งทั้งสองอย่างพร้อมกัน

## เตรียมข้อมูลสำหรับ train

สร้างโฟลเดอร์ `dataset/` แล้วแยกโฟลเดอร์ย่อยตามชื่อแต่ละคน:

```
dataset/
├── Alice/
│   ├── 1.jpg
│   ├── 2.jpg
│   └── 3.jpg
└── Bob/
    └── 1.jpg
```

คำแนะนำ:

- ใช้รูปที่เห็นหน้าตรง ชัดเจน อย่างน้อย 5–10 รูปต่อคน หลายมุม หลายแสง จะแม่นกว่า 1 รูป
- ในรูป train ควรมีหน้าเจ้าของคนเดียว (โปรแกรมจะเลือกหน้าที่ใหญ่ที่สุดในรูป)
- ชื่อโฟลเดอร์ใช้ภาษาอังกฤษ เพราะ `cv.putText` ของ OpenCV ไม่รองรับภาษาไทย

## วิธีใช้งาน

```bash
python face_detect.py
```

ตัวเลือกเพิ่มเติม:

| อาร์กิวเมนต์ | ค่าเริ่มต้น | คำอธิบาย |
|---|---|---|
| `--dataset` | `dataset` | โฟลเดอร์รูปสำหรับ train |
| `--camera` | `0` | index ของกล้อง (ถ้ามีหลายตัว) |
| `--threshold` | `80` | ค่า confidence สูงสุดที่ยอมรับว่าเป็นคนนั้น ยิ่งต่ำยิ่งเข้มงวด |
| `--cascade` | ของ OpenCV | path ของไฟล์ Haar Cascade ที่ต้องการใช้เอง |

ตัวอย่าง:

```bash
python face_detect.py --dataset my_faces --camera 1 --threshold 70
```

## การปรับค่า threshold

- จำผิดคนบ่อย → ลด `--threshold` (เช่น 60–70)
- จำคนจริงไม่ได้ (ขึ้น Unknown) → เพิ่มรูป train หรือเพิ่ม `--threshold`

## ข้อจำกัด

- LBPH ไวต่อแสงและมุมหน้า ความแม่นยำสู้โมเดล deep learning (เช่น `face_recognition`, ArcFace) ไม่ได้
- Haar Cascade ตรวจจับได้ดีเฉพาะใบหน้าที่หันตรง
- ไม่มีระบบป้องกันการใช้รูปถ่ายหลอก (anti-spoofing) **ไม่ควรใช้เป็นระบบยืนยันตัวตนจริงจัง**

## ความเป็นส่วนตัว

รูปใบหน้าเป็นข้อมูลส่วนบุคคล โปรเจกต์นี้ตั้ง `.gitignore` ให้ไม่ commit โฟลเดอร์ `dataset/` และไฟล์รูปทั้งหมด
กรุณาอย่าอัปโหลดรูปของผู้อื่นขึ้น repo สาธารณะโดยไม่ได้รับอนุญาต

## เครดิต

- [OpenCV](https://opencv.org/) — Haar Cascade (`haarcascade_frontalface_default.xml`) พัฒนาโดย Rainer Lienhart
  เผยแพร่ภายใต้ Intel License Agreement ซึ่งมากับ OpenCV อยู่แล้ว โปรเจกต์นี้จึงเรียกใช้จากแพ็กเกจโดยตรง ไม่ได้แนบไฟล์ไว้ใน repo

## License

เลือก license ที่ต้องการ (แนะนำ MIT) แล้วเพิ่มไฟล์ `LICENSE`
