'''
python3 utils/split.py
'''
import os
import cv2

# Init
DIM = 128
dst = 'dst'
if not os.path.exists(dst):
    os.mkdir(dst)

img_path = 'assets/concrete.jpg'

img = cv2.imread(img_path)

if __name__ == "__main__":

    H,W,_ = img.shape
    cols = int(W/DIM)
    rows = int(H/DIM)
    dim = (cols*DIM,rows*DIM)
    img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    cell_no = 0
    print(f'cols: {cols} rows: {rows}')
    for r in range(rows):
        for c in range(cols):
            y_start = r*DIM
            x_start = c*DIM
            y_end = (r+1)*DIM
            x_end = (c+1)*DIM
            cell = img[y_start:y_end,x_start:x_end]
            #cv2.imwrite(f'{dst}/cell_{cell_no}.jpg',cell)
            cell_no +=1
            cv2.rectangle(img, (x_start,y_start), (x_end,y_end), (100,100,0), 2)
    
    cv2.imwrite('tmp.jpg',img)