'''
python3 utils/split.py path/to/images
'''
import os
import cv2

# Init
DIM = 128
dst = 'dst'
src = 'data'

if not os.path.exists(dst):
    os.mkdir(dst)

if __name__ == "__main__":

    images = os.listdir(src) 

    for i,image in enumerate(images):
        img_path = os.path.join(src,image)
        img = cv2.imread(img_path)

        cell_no = 0
        H,W,_ = img.shape
        cols = int(W/DIM)
        rows = int(H/DIM)
        dim = (cols*DIM,rows*DIM)
        img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA) 
        
        for r in range(rows):
            for c in range(cols):
                y_start = r*DIM
                x_start = c*DIM
                y_end = (r+1)*DIM
                x_end = (c+1)*DIM
                cell = img[y_start:y_end,x_start:x_end]
                cv2.imwrite(f'{dst}/cell_{i}_{cell_no}.jpg',cell)
                cell_no +=1
                cv2.rectangle(img, (x_start,y_start), (x_end,y_end), (100,100,0), 2)
        
        #cv2.imwrite('tmp.jpg',img)