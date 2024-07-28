import cv2 as cv
import numpy as np
import os
from copy import deepcopy

#detectare colturi si uniformizare tabla
def extrage_careu(image):
    image = cv.cvtColor(image,cv.COLOR_BGR2GRAY)
    image_m_blur = cv.medianBlur(image,3)
    image_g_blur = cv.GaussianBlur(image_m_blur, (0, 0), 5) 
    image_sharpened = cv.addWeighted(image_m_blur, 1.4, image_g_blur, -0.8, 0)
    _, thresh = cv.threshold(image_sharpened, 120, 255, cv.THRESH_BINARY)

    kernel = np.ones((5, 5), np.uint8)
    thresh = cv.dilate(thresh, kernel)

    edges =  cv.Canny(thresh ,200,400)
    refined_edges = cv.dilate(edges,kernel)
    contours, _ = cv.findContours(refined_edges,  cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    max_area = 0

    for i in range(len(contours)):
        if(len(contours[i]) >3):
            possible_top_left = None
            possible_bottom_right = None
            for point in contours[i].squeeze():
                if possible_top_left is None or point[0] + point[1] < possible_top_left[0] + possible_top_left[1]:
                    possible_top_left = point

                if possible_bottom_right is None or point[0] + point[1] > possible_bottom_right[0] + possible_bottom_right[1] :
                    possible_bottom_right = point

            diff = np.diff(contours[i].squeeze(), axis = 1)
            possible_top_right = contours[i].squeeze()[np.argmin(diff)]
            possible_bottom_left = contours[i].squeeze()[np.argmax(diff)]
            if cv.contourArea(np.array([[possible_top_left],[possible_top_right],[possible_bottom_right],[possible_bottom_left]])) > max_area:
                max_area = cv.contourArea(np.array([[possible_top_left],[possible_top_right],[possible_bottom_right],[possible_bottom_left]]))
                top_left = possible_top_left
                bottom_right = possible_bottom_right
                top_right = possible_top_right
                bottom_left = possible_bottom_left

    width = 2250
    height = 2250

    image_copy = cv.cvtColor(image.copy(),cv.COLOR_GRAY2BGR)
    cv.circle(image_copy,tuple(top_left),20,(0,0,255),-1)
    cv.circle(image_copy,tuple(top_right),20,(0,0,255),-1)
    cv.circle(image_copy,tuple(bottom_left),20,(0,0,255),-1)
    cv.circle(image_copy,tuple(bottom_right),20,(0,0,255),-1)

    puzzle = np.array([top_left,top_right,bottom_right,bottom_left], dtype = "float32")
    destination_of_puzzle = np.array([[0,0],[width,0],[width,height],[0,height]], dtype = "float32")

    M = cv.getPerspectiveTransform(puzzle,destination_of_puzzle)

    result = cv.warpPerspective(image, M, (width, height))
    
    return result

#determinare grid
lines_horizontal=[]
for i in range(0,2251,150):
    l=[]
    l.append((0,i))
    l.append((2249,i))
    lines_horizontal.append(l)
    
lines_vertical=[]
for i in range(0,2251,150):
    l=[]
    l.append((i,0))
    l.append((i,2249))
    lines_vertical.append(l)

#creeare matrice cu zone ocupate
def determina_configuratie_careu(thresh,lines_horizontal,lines_vertical):
    matrix = np.empty((15,15), dtype='bool')
    for i in range(len(lines_horizontal)-1):
        for j in range(len(lines_vertical)-1):
            y_min = lines_vertical[j][0][0] + 20
            y_max = lines_vertical[j + 1][1][0] - 20
            x_min = lines_horizontal[i][0][1] + 20
            x_max = lines_horizontal[i + 1][1][1] - 20
            patch = thresh[x_min:x_max, y_min:y_max].copy()
            Medie_patch=np.mean(patch)
            if Medie_patch>0:
                matrix[i][j]=True
            else:
                matrix[i][j]=False
    return matrix

#functie contrast litera
def literaSmooth(image):
    image_m_blur = cv.medianBlur(image,3)
    image_g_blur = cv.GaussianBlur(image_m_blur,(0,0),5)
    image_sharpened = cv.addWeighted(image_m_blur,1.4,image_g_blur,-0,8)
    return image_sharpened

#creare lista cu litere dintr-un template
def creaza_litere(result,matrix,lines_horizontal,lines_vertical):
    letterList = []
    for i in range(len(lines_horizontal) - 1):
        for j in range(len(lines_vertical) - 1):
            y_min = lines_vertical[j][0][0]
            y_max = lines_vertical[j + 1][1][0]
            x_min = lines_horizontal[i][0][1]
            x_max = lines_horizontal[i + 1][1][1]
            if matrix[i][j] == True: 
                letterList.append(result[x_min+20:x_max-20, y_min+20:y_max-20])
    return letterList

template1 = cv.imread('imagini_auxiliare/litere_1.jpg')#aici pus path pentru imaginea cu toate literele unite
template2 = cv.imread('imagini_auxiliare/litere_2.jpg')#aici pus path pentru imaginea cu toate literele distantate

result=extrage_careu(template1)
_, thresh = cv.threshold(result, 210, 255, cv.THRESH_BINARY)
matrice=determina_configuratie_careu(thresh,lines_horizontal,lines_vertical)
letterSet1 = creaza_litere(result,matrice,lines_horizontal,lines_vertical)

result=extrage_careu(template2)
_, thresh = cv.threshold(result, 210, 255, cv.THRESH_BINARY)
matrice=determina_configuratie_careu(thresh,lines_horizontal,lines_vertical)
letterSet2 = creaza_litere(result,matrice,lines_horizontal,lines_vertical)

letterList = ['A','B','C','D','E',
              'F','G','H','I','J',
              'L','M','N','O','P',
              'R','S','T','U','V',
              'X','Z','?']
scoreList = [1 ,9 ,1 ,2 ,1 ,
             8 ,9 ,10,1 ,10,
             1 ,4 ,1 ,1 ,2 ,
             1 ,1 ,1 ,1 ,8 ,
             10,10,0]

#functie arata ce litera se afla pe un patch
def aflaLitera(patch):

    maxi=-np.inf
    poz=-1
    j=0
    for letter in letterSet1:
            
        corr = cv.matchTemplate(literaSmooth(patch),literaSmooth(letter),  cv.TM_CCOEFF_NORMED)
        corr=np.max(corr)
        if corr>maxi:
            maxi=corr
            poz=j
        j+=1
    j=0
    for letter in letterSet2:      
        corr = cv.matchTemplate(literaSmooth(patch),literaSmooth(letter),  cv.TM_CCOEFF_NORMED)
        corr = np.max(corr)
        if corr > maxi:
            maxi = corr
            poz = j
        j+=1
            
    return poz
    
#functie creeare matrice litere
def matrice_valori(result,matrix,lines_horizontal,lines_vertical):
    rez = np.full((15,15),-1)
    for i in range(len(lines_horizontal) - 1):
        for j in range(len(lines_vertical) - 1):
            y_min = lines_vertical[j][0][0]
            y_max = lines_vertical[j + 1][1][0]
            x_min = lines_horizontal[i][0][1]
            x_max = lines_horizontal[i + 1][1][1]
            if matrix[i][j] == True: 
                rez[i][j]=aflaLitera(result[x_min:x_max, y_min:y_max])
    return rez

#definire tabla culori
dL=1
tL=2
dC=3
tC=4
tablaScor =[
    [tL, 0 , 0 , dC, 0 , 0 , 0 , tL, 0 , 0 , 0 , dC, 0 , 0 , tL],
    [0 , dL, 0 , 0 , 0 , tC, 0 , 0 , 0 , tC, 0 , 0 , 0 , dL, 0 ],
    [0 , 0 , dL, 0 , 0 , 0 , dC, 0 , dC, 0 , 0 , 0 , dL, 0 , 0 ],
    [dC, 0 , 0 , dL, 0 , 0 , 0 , dC, 0 , 0 , 0 , dL, 0 , 0 , dC],
    [0 , 0 , 0 , 0 , dL, 0 , 0 , 0 , 0 , 0 , dL, 0 , 0 , 0 , 0 ],
    [0 , tC, 0 , 0 , 0 , tC, 0 , 0 , 0 , tC, 0 , 0 , 0 , tC, 0 ],
    [0 , 0 , dC, 0 , 0 , 0 , dC, 0 , dC, 0 , 0 , 0 , dC, 0 , 0 ],
    [tL, 0 , 0 , dC, 0 , 0 , 0 , dL, 0 , 0 , 0 , dC, 0 , 0 , tL],
    [0 , 0 , dC, 0 , 0 , 0 , dC, 0 , dC, 0 , 0 , 0 , dC, 0 , 0 ],
    [0 , tC, 0 , 0 , 0 , tC, 0 , 0 , 0 , tC, 0 , 0 , 0 , tC, 0 ],
    [0 , 0 , 0 , 0 , dL, 0 , 0 , 0 , 0 , 0 , dL, 0 , 0 , 0 , 0 ],
    [dC, 0 , 0 , dL, 0 , 0 , 0 , dC, 0 , 0 , 0 , dL, 0 , 0 , dC],
    [0 , 0 , dL, 0 , 0 , 0 , dC, 0 , dC, 0 , 0 , 0 , dL, 0 , 0 ],
    [0 , dL, 0 , 0 , 0 , tC, 0 , 0 , 0 , tC, 0 , 0 , 0 , dL, 0 ],
    [tL, 0 , 0 , dC, 0 , 0 , 0 , tL, 0 , 0 , 0 , dC, 0 , 0 , tL]
]

#calculeaza scor cuvant
def daCuvant(tabla,xLitera,yLitera,peLinie):
    xInx=xLitera
    yInx=yLitera
    puncte = 0
    double = 0
    triple = 0
    noWord =True
    
    while xInx <= 14 and yInx <= 14 and tabla[xInx][yInx] != -1:
        if xInx != xLitera or yInx != yLitera:
            noWord = False
        litera = scoreList[tabla[xInx][yInx]]
        if tablaScor[xInx][yInx] > 4:
            speciala = tablaScor[xInx][yInx]-4
            if speciala == tL:
                triple += 1
            if speciala == dL:
                double += 1
            if speciala == tC:
                litera *= 3
            if speciala == dC:
                litera *= 2
            speciala = tablaScor[xInx][yInx]+4
        puncte+=litera      
        if peLinie == True:
            yInx+=1
        else:
            xInx+=1
            
    xInx=xLitera
    yInx=yLitera
    if peLinie == True:
        yInx-=1
    else:
        xInx-=1
    
    while xInx >= 0 and yInx >= 0 and tabla[xInx][yInx]!=-1:
        noWord = False
        litera = scoreList[tabla[xInx][yInx]]
        if tablaScor[xInx][yInx]>4:
            speciala = tablaScor[xInx][yInx]-4
            if speciala == tL:
                triple += 1
            if speciala == dL:
                double += 1
            if speciala == tC:
                litera *= 3
            if speciala == dC:
                litera *= 2
        puncte+=litera
        if peLinie == True:
            yInx-=1
        else:
            xInx-=1
            
    if noWord ==True:
        return 0
    while double>0:
        puncte*=2
        double-=1
        
    while triple>0:
        puncte*=3
        triple-=1
    return puncte
        
#calculeaza scor tabla
def calculeaza_scor(tabla, solutii):
    scor = 0
    same_lin = False
    
    if len(solutii) > 1:
        if solutii[0][0] == solutii[1][0]:
            same_lin = False
        else:
            same_lin = True
            
    for solutie in solutii:
        tablaScor[solutie[0]][solutie[1]] +=4
        scor+=daCuvant(tabla,solutie[0],solutie[1],same_lin)
        
    same_lin = not same_lin
    
    scor+=daCuvant(tabla,solutie[0],solutie[1],same_lin) 
    for solutie in solutii:
        tablaScor[solutie[0]][solutie[1]] -=4
        
    if len(solutii)>=7:
        scor+=50
    return scor

#foldere input/output
path_testare = 'evaluare/fake_test/' #path folder de testare
path_rezultat = 'evaluare/fisiere_solutie/' #path in care se creaza folderu de solutie
path_rezultat =path_rezultat+ '333_Gogu_Razvan/'
if(os.path.exists(path_rezultat)!=True):    
    os.mkdir(path_rezultat)
    
#zona main
runda=0
tablaVeche = []
files=os.listdir(path_testare)
files.sort()
for file in files:
    if file[-3:]=='jpg':
        if runda == 0:
            tablaVeche = np.full((15,15),-1)
            
        img = cv.imread(path_testare+file)
        result=extrage_careu(img)
        _, thresh = cv.threshold(result, 210, 255, cv.THRESH_BINARY)
        matrice=determina_configuratie_careu(thresh,lines_horizontal,lines_vertical)
        tabla = matrice_valori(result,matrice,lines_horizontal,lines_vertical)
        
        listaSolutii = []
        for i in range(15):
            for j in range(15):
                if tablaVeche[i][j] == -1 and tabla[i][j] != -1:
                    tupluSolutie = (i,j,tabla[i][j])
                    listaSolutii.append(tupluSolutie)
        scor = calculeaza_scor(tabla, listaSolutii)
        
        string_rez=''
        for tuplu in listaSolutii:
            string_tuplu = str(tuplu[0]+1)+chr(tuplu[1]+65)+' '+letterList[tuplu[2]]+'\n'
            string_rez = string_rez + string_tuplu
        string_rez = string_rez + str(scor)
        with open(os.path.join(path_rezultat, file[:-3]+'txt'), 'w') as fp:
            fp.write(string_rez)
            
        tablaVeche = deepcopy(tabla)    
        runda = (runda+1)%20
                    
        




