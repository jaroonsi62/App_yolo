import streamlit as st
import cv2
import numpy as np
from yolo_predictions import YOLO_Pred

yolo = YOLO_Pred('my_obj.onnx','my_obj.yaml') 
name = ['Agaricus', 'Amanita', 'Boletus', 'Cortinarius', 'Entoloma', 'Exidia', 'Hygrocybe',
        'Inocybe', 'Lactarius', 'Pluteus', 'Russula', 'Suillus']
st.title("การจำแนกชนิดของเห็ด : ภาพนิ่ง")
img_file = st.file_uploader("เปิดไฟล์ภาพ")

if img_file is not None:    
    file_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    #----------------------------------------------
    pred_image, obj_box = yolo.predictions(img)
    
    if len(obj_box) > 0:
        sum_conf = [0,0,0,0,0,0,0,0,0,0,0,0]
        cont_obj = [0,0,0,0,0,0,0,0,0,0,0,0]
        av = []
        for i in obj_box:
            for k in range(len(name)):
                if i[4] == name[k]:
                    sum_conf[k] += i[5]
                    cont_obj[k] += 1
        if max(cont_obj) > 0:
            for u in range(len(sum_conf)):
                try:
                    av.append(sum_conf[u]/cont_obj[u])
                except ZeroDivisionError :
                    av.append(0)
            text_obj = 'เห็ดที่ตรวจพบ : ' + name[av.index(max(av))]
        else:
            text_obj = 'ไม่พบชนิดของเห็ด'
##        obj_names = ''
##        for obj in obj_box:
##            obj_names = obj_names + obj[4] + ' '
##        text_obj = 'ตรวจพบ ' + obj_names
##    else:
##        text_obj = 'ไม่พบชนิดของเห็ด'
    #----------------------------------------------
    st.header(text_obj)
    st.image(pred_image, caption='ภาพ Output',channels="BGR")
    
