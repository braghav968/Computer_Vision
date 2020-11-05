import cv2
import numpy as np
import math
from numpy import array
from heapq import *
from numpy import*



#im = cv2.imread("maze.png",0).astype(np.uint8)
im = cv2.imread("Input_Maze_Image.jpg",0).astype(np.uint8)
#im=cv2.resize(im,(300,300))
#im=cv2.bilateralFilter(im,9,75,75)
#cv2.imshow("blur",im)
skel = np.zeros((im.shape[0],im.shape[1]),dtype = np.uint8)
temp = np.zeros((im.shape[0],im.shape[1]),dtype = np.uint8)
eroded = np.zeros((im.shape[0],im.shape[1]),dtype = np.uint8)
element = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
element2 = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7),(-1,-1))
element3 = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7),(-1,-1))
im=cv2.erode(im,element)

done = False 

eroded = cv2.erode(im,element)#,iterations=1)
temp=cv2.dilate(eroded,element)#,iterations=1)
temp=cv2.subtract(im,temp)
skel=cv2.bitwise_or(skel,temp)
im=eroded.copy()
if cv2.countNonZero(im) == 0:
    done = True 
        

while done==False:
    eroded = cv2.erode(im,element)#,iterations=1)
    temp=cv2.dilate(eroded,element)#,iterations=1)
    temp=cv2.subtract(im,temp)
    skel=cv2.bitwise_or(skel,temp)
    im=eroded.copy()
    
    if cv2.countNonZero(im)==0:
        done = True 
    
skel=cv2.dilate(skel,element2)
skel=cv2.erode(skel,element3)
#cv2.imshow("Skeleton",skel)

for i in range(1,im.shape[0] - 1):
    for j in range(1,im.shape[1] - 1):
        if im[i,j]<200 :
            im[i,j]=0
        else :
            im[i,j]=255

#cv2.imshow("thr",im)
#print(im.shape[0],im.shape[1])
#global ct
#ct=0

def thinningIteration(im,iter):
    #marker = np.zeros((im.shape[0],im.shape[1]),dtype = np.uint8)
    marker=im-im
    print(marker.shape[0],marker.shape[1],marker[10][10])
    #marker=cv2.resize(im,(im.shape[0],im.shape[1]),fx=1,fy=1)
    for i in range(1,im.shape[0] - 1):
        for j in range(1,im.shape[1] - 1):
            p2 = im[i-1,j]
            p3 = im[i-1,j+1]
            p4 = im[i,j+1]
            p5 = im[i+1,j+1]
            p6 = im[i+1,j]
            p7 = im[i+1,j-1]
            p8 = im[i,j-1]
            p9 = im[i-1,j-1]
            A = (p2 == 0 and p3 == 1)+(p3 == 0 and p4 == 1)+(p6 == 0 and p7 == 1)+(p7 == 0 and p8 == 1)+(p8 == 0 and p9 == 1)+(p9 == 0 and p2 == 1)+(p4 == 0 and p5 == 1)+(p5 == 0 and p6 == 1)
            B = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9
            
            if iter==0:
                m1=(p2*p4*p6)
                m2=(p4*p6*p8)
            else:
                m1=(p2*p4*p8)
                m2=(p2*p6*p8)
                        
            if A == 1 and (B >= 2 and B <= 6) and m1 == 0 and m2 == 0:
                marker[i,j] = 1
    #cv2.imshow("marker"+str(iter),marker)
    im = cv2.bitwise_and(im,cv2.bitwise_not(marker))

def thinning(im,ct):
    im = im / 255
    prev = np.zeros((im.shape[0],im.shape[1]),dtype = np.uint8)
    #diff = np.zeros((im.shape[0],im.shape[1],1),dtype = np.uint8)

    thinningIteration(im,0)
    thinningIteration(im,1)
    # diff=cv2.absdiff(im,prev)
    diff=abs(im-prev)
    prev=im.copy()
        
    while cv2.countNonZero(diff)>0 :
        thinningIteration(im,0)
        thinningIteration(im,1)
        #diff=cv2.absdiff(im,prev)
        diff=abs(im-prev)
        prev=im.copy()
    #cv2.imshow("diff"+str(ct),diff)
    ct+=1
    im = im * 255
    return im

#cv2.imshow("Result",thinning(skel,0))
thinned=thinning(skel,0)
cv2.imwrite("maze_thinned.png",thinned)
cv2.imshow("thin",thinned)
cv2.waitKey(0)
cv2.destroyAllWindows()


c=0
initial=[]
# Lists to store the points
circumference=[]
def drawCircle(action, x, y, flags, userdata):
  # Referencing global variables 
  global mouseX,mouseY,circumference
  # Action to be taken when left mouse button is pressed
  if action==cv2.EVENT_LBUTTONDOWN:
    # Mark the center
    cv2.circle(source, (x,y), 10, (255,255,0), 2, cv2.LINE_AA )
    mouseX,mouseY = x,y

source = cv2.imread("maze_thinned.png",0)
for i in range(1,source.shape[0] - 1):
    for j in range(1,source.shape[1] - 1):
        if source[i,j]<50 :
            source[i,j]=0
        else :
            source[i,j]=255
#print(source[114,25],source[114,25])
# Make a dummy image, will be useful to clear the drawing
dummy = source.copy()
cv2.namedWindow("Window")
# highgui function called when mouse events occur
cv2.setMouseCallback("Window", drawCircle)
k = 0
# loop until escape character is pressed

while k!=27 :
  
  cv2.imshow("Window", source)
  cv2.putText(source,"Choose center, and drag, Press ESC to exit and c to clear" ,(10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,(255,255,255), 2 );
  k = cv2.waitKey(20) & 0xFF
  # Another way of cloning
  if k==99:
    source= dummy.copy()
  elif k == ord('a'):
    print(mouseX,mouseY,source[mouseY,mouseX])
    if source[mouseY,mouseX]==255  :
        initial.append((mouseY,mouseX))
        print(initial)

  
cv2.destroyAllWindows()



image=cv2.imread('maze_thinned.png',0)
#image=cv2.imread('thin2.jpg',0)
for i in range(1,image.shape[0] - 1):
    for j in range(1,image.shape[1] - 1):
        if image[i,j]<50 :
            image[i,j]=0
        else :
            image[i,j]=255
cv2.imshow("thresh",image)
image2=cv2.imread('maze_thinned.png')
new_A=empty((image.shape[0],image.shape[1]),None)
print(image[81,137], image[81,278])

for i in range(image.shape[0]):
    for j in range(image.shape[1]):
        if image[i][j]>=100:
            new_A[i][j]=1
            image[i][j]=255
        else:
            new_A[i][j]=0
            image[i][j]=0

def heuristic(a, b):
    (x1, y1) = a
    (x2, y2) = b
    return abs(x1 - x2) + abs(y1 - y2)

def astar(array, start, goal):

    neighbors = [(0,1),(0,-1),(1,0),(-1,0),(1,1),(1,-1),(-1,1),(-1,-1)]

    close_set = set()
    came_from = {}
    gscore = {start:0}
    fscore = {start:heuristic(start, goal)}
    oheap = []

    heappush(oheap, (fscore[start], start))
    while oheap:
        
        current = heappop(oheap)[1]
    
        if current == goal:
   
            data = []
            directions=[]
            pos=0
            temp=current
            second_last = temp
            print("Printing Coordinates")
            while current in came_from:
                if(current[0]-temp[0]==1 and temp[1]-current[1]==1):
                    if(second_last[0]-temp[0]==0 and second_last[1]-temp[1]==1):
                        directions.append('RIGHT')
                    elif (temp[0]-second_last[0]==1 and temp[1]==second_last[1]):
                        directions.append('LEFT')
                elif(current[0]-temp[0]==1 and current[1]-temp[1]==1):
                    if(second_last[0]-temp[0]==0 and temp[1]-second_last[1]==1):
                        directions.append('LEFT')
                    elif(temp[0]-second_last[0]==1 and second_last[1]==temp[1]):
                        directions.append('RIGHT')
                elif(temp[0]-current[0]==1 and temp[1]-current[1]==1):
                    if(second_last[0]-temp[0]==0 and second_last[1]-temp[1]==1):
                        directions.append('LEFT')
                    elif(second_last[0]-temp[0]==1 and second_last[1]-temp[1]==0):
                        directions.append("RIGHT")
                elif(temp[0]-current[0]==1 and current[1]-temp[1]==1):
                    if(second_last[0]-temp[0]==0 and temp[1]-second_last[1]==1):
                        directions.append('RIGHT')
                    elif(second_last[0]-temp[0]==1 and second_last[1]==temp[1]):
                        directions.append('LEFT')
                data.append(current)
                second_last = temp
                temp = current
                pos=pos+1
                image2[current[0],current[1]]=[0,250,0]
                current = came_from[current]
            return directions

        close_set.add(current)
        
        for i, j in neighbors:
            print("For loop Working")
            neighbor = current[0] + i, current[1] + j            
            tentative_g_score = gscore[current] + heuristic(current, neighbor)
            if 0 <= neighbor[0] < array.shape[0]:
                print("1")
                if 0 <= neighbor[1] < array.shape[1]:       
                    print("2")         
                    if array[neighbor[0]][neighbor[1]] >= 1:
                        print("3",neighbor[0],neighbor[1])
                        if neighbor in close_set and tentative_g_score >= gscore.get(neighbor, 0):
                            continue
                        elif tentative_g_score < gscore.get(neighbor, 0) or neighbor not in [i[1]for i in oheap]:
                            came_from[neighbor] = current
                            gscore[neighbor] = tentative_g_score
                            fscore[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                            heappush(oheap, (fscore[neighbor], neighbor))                        
                else:
                    # array bound y walls
                    print("Y walls")
                    continue

            else:
                # array bound x walls
                print("X walls")
                continue          
                
                
    return False



print (astar(image,initial[0],initial[1]))
#print (astar(image,(491,543),(162,902)))
cv2.imshow("img",image)
cv2.imshow("img2",image2)
cv2.waitKey(0)
