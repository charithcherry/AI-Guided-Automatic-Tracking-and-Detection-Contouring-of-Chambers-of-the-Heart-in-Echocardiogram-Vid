import numpy as np
from scipy.spatial import ConvexHull
import pydicom
import math
import matplotlib.pyplot as plt
import cv2
plt.set_cmap('gray')

#-----------------------------------------------------------------------------------
DISPLAY = True
def showImage(im, title=None, numWin=None, show=False):
    if DISPLAY:
        if numWin != None: plt.subplot(numWin)
        plt.imshow(im), plt.title(title), plt.xticks([]), plt.yticks([])

        if show:
            plt.text(20, 20, 'press any key to continue'), plt.show(block=False)
            plt.waitforbuttonpress(), plt.close()
            return

#---------------------------------------------------------------------------
stPt, endPt = (0,0), (0,0)  # start and end point of rectangle selected by user as seed region
hoverPt = circleLoc = (0, 0)
img=0   # Will store the image to display for drawing of rectangle used as seed point
radius, bdown, saveCtr = 50, False, False

# --------------------------------------------------------------------------

font                   = cv2.FONT_HERSHEY_SIMPLEX
pos                    = (10,50)
fontScale              = 0.4
fontColor              = (255,255,255)
lineType               = 1

instructions = { \
1:  'q/a - increase/reduce circle size \n',
2:  'd - enter/exit drawing mode \n',
3:  's - save contour \n',
4:  'n/p - next/previous image \n',
5:  'z/x - zoom in/out \n',
6:  'r   - repair contour \n',
7:  'Esc - quit',
8:    'Image file name',
-1:  'Status: ' \
    }


#-------------------------------------------------------------------------------------
def saveContour(save_path, ctr, scale, crop):
    global saveCtr
    ctr = ctr / scale
    ctr[:, 0] += crop
    ctr[:, 1] += crop
    np.savetxt(save_path, ctr, fmt='%.2f')
    exit_mode = 'done'
    saveCtr = False


#-------------------------------------------------------------------------------------
def display_text(instructions, cont_name, save_path, draw_mode, img2, confirm_mode):
    if draw_mode:
        mode = "Draw mode"
    else:
        mode = "Nudge mode"
    prompt_text = "Press 'y' to confirm; 'n' to cancel"
    imFile = save_path.split('\\')[-1]
    instructions[8] = 'File name: ' + imFile
    for i, line in enumerate(instructions.values()):
        cv2.putText(img2, line, 
                (pos[0], pos[1] + i * 15),
                font, 
                fontScale,
                fontColor,
                lineType)

    cv2.putText(img2, cont_name, 
                (300, 30),
                font, 
                1.2,
                fontColor,
                lineType)
    cv2.putText(img2, mode, 
                (300, 50),
                font, 
                0.6,
                fontColor,
                lineType)
    if confirm_mode:
        cv2.putText(img2, prompt_text, 
                (120, 100), 
                font, 
                0.5,
                fontColor,
                lineType)

#-------------------------------------------------------------------------------------
# mouse callback function
def draw_circle(event, x, y, flags, param):
    global stPt, endPt, hoverPt, img, bdown, saveCtr, circleLoc
    if event == cv2.EVENT_LBUTTONDOWN:
        stPt = circleLoc = (x, y)
        bdown = True
    elif bdown == True and event == cv2.EVENT_MOUSEMOVE:
        endPt = circleLoc = (x, y)
    elif event == cv2.EVENT_LBUTTONUP:
        bdown = False
    elif event == cv2.EVENT_MOUSEMOVE:
        hoverPt = circleLoc = (x, y)
        pass
    elif event == cv2.EVENT_LBUTTONDBLCLK:
        stPt == (x, y)
        saveCtr = True

#-------------------------------------------------------------
distance = lambda p1, p2: math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)
def updateContour(ctr, centre, radius):
    #centroid = (np.mean(ctr[:, 0]), np.mean(ctr[:, 1]))
    centroid = (300.0, 300.0)
    centroid = (300.0, 300.0)
    for i, pt in enumerate(ctr):
        dist = distance(pt, centre)
        if dist < radius:  # if the point is inside the circle
            th = math.atan2(pt[1]-centroid[1], pt[0]-centroid[0])
            a, b = pt[0], pt[1]
            c, d = centre[0], centre[1]
            m, n = math.cos(th), math.sin(th)
            z = radius

            B = 2*(a*m + b*n - c*m - d*n)
            C = (c-a)**2 + (d-b)**2 - z**2
            D = (B*B - 4*C)**0.5
            r1, r2 = 0.5*(-B + D), 0.5*(-B - D)

            if abs(r1) < abs(r2):
                r = r1
            else:
                r = r2
            x,y = pt[0] + r * math.cos(th), pt[1] + r * math.sin(th)
            ctr[i] = (x,y)
    ctr = dense(ctr)
    return ctr


#-------------------------------------------------------------
def drawContour(ctr, hover_inserted=False):
    global img, bdown, stPt, hoverPt, endPt, radius
    cutoff = 8
    if stPt != (0,0):    
        if ctr is None:
            ctr = np.array([stPt], int)
        else:
            if hover_inserted == True:
                ctr = ctr[:-1]
                hover_inserted = False
            if any(np.array(stPt) != ctr[-1]) == True and not bdown:
                ctr = np.append(ctr, np.array([stPt], int), axis=0)
                endPt = stPt
            if endPt != (0,0) and bdown:
                if np.sum(np.array(endPt) != ctr[-1]) == 2:
                    if distance(endPt, ctr[-1]) > cutoff:
                        ctr = np.append(ctr, np.array([endPt], int), axis=0)
                        stPt = endPt
        
        if ctr.shape[0] > 0 and not bdown:
            if hover_inserted == True:
                ctr = ctr[:-1]
                hover_inserted = False
            ctr = np.append(ctr, np.array([hoverPt], int), axis=0)
            hover_inserted = True
    return ctr, hover_inserted


#-------------------------------------------------------------
def dense(ctr):
    dists = []
    candidates = []
    for i, pt in enumerate(ctr):
        if i != len(ctr)-1:
            dist = distance(ctr[i], ctr[i+1])
        else:
            dist = distance(ctr[i], ctr[0])
        dists.append(dist)

    cutoff = 10
    offset = 0
    for i, d in enumerate(dists):
        num = int(d/cutoff)
        if num == 0:
            continue
        if i != len(dists)-1:
            points = [ctr[i+offset] + ((f + 1)/(num + 1)) * (ctr[i+1+offset] - ctr[i+offset]) for f in range(num)]
        else:
            points = [ctr[i+offset] + ((f + 1)/(num + 1)) * (ctr[0]- ctr[i+offset]) for f in range(num)]
        points = [pt.reshape([-1, 2]) for pt in points]
        points = np.concatenate(points, axis=0)
        ctr = np.insert(ctr, i+1+offset, points, axis = 0)
        offset += points.shape[0]
    return ctr

#-------------------------------------------------------------
def repair(ctr):
    num_points = ctr.shape[0]
    hull = ConvexHull(ctr)
    x = ctr[hull.vertices, 0].reshape(-1, 1)
    y = ctr[hull.vertices, 1].reshape(-1, 1)
    ctr = np.concatenate((x, y), axis=1)
    ctr = dense(ctr)
    return ctr

#-------------------------------------------------------------
def draw_points(ctr, img2):
    for i in range(ctr.shape[0]):
        cv2.circle(img2, (ctr[i][0], ctr[i][1]), 1, (255, 0, 0), -1)

#-------------------------------------------------------------
def prompt():
    result = False
    while(True):
        key = cv2.waitKey(20)
        if key & 0xFF == ord('y'):
            result = True
            break
        if key & 0xFF == ord('n'):
            result = False
            break
    return result

#-------------------------------------------------------------
def adjustContour(img_, ctr, dim, crop, save_path):
    global img, bdown, stPt, endPt, radius, instructions
    exit_mode = 'undefined'
    img = img_.copy()
    scale = 5
    if ctr is None:
        draw = True
    else:
        draw = False
        ctr = ctr.astype(int)
        ctr = np.array(ctr)
        og_ctr = ctr.copy()
        ctr[:,0] -= crop
        ctr[:,1] -= crop
        ctr = scale * ctr
    img = cv2.resize(img, dsize=(int(scale*dim[1]), int(scale*dim[0])))
    cont_name = "INNER"

    radius_delta = 2
    min_radius = 10
    dim_win = 1000
    min_dim = 500
    dim_delta = 100
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('image', dim_win, dim_win)
    cv2.setMouseCallback('image', draw_circle)
    # Get the centre of circle and boundary points affected by the circle
    img2 = img.copy()
    translate = False
    hover_inserted = False
    open_contour = True
    check = False
    confirm_mode = False
    while(1):
        img2 = img.copy()
        if bdown == True and endPt != (0, 0) and not draw:
            ctr = updateContour(ctr, endPt, radius)
        if draw:
            if saveCtr == True:
                saveContour(save_path, ctr[:-1], scale, crop)
                exit_mode = 'done'
                break
            ctr, hover_inserted = drawContour(ctr, hover_inserted)
        if ctr is not None:
            cv2.polylines(img2, [ctr], open_contour, (255,255,0), 1)
            if check:
                cv2.fillPoly(img2, [ctr], (255,255,0))
            #draw_points(ctr, img2)
        if not draw:
            cv2.circle(img2, circleLoc, radius, (255,255,0), 1)
            instructions[-1] = 'Status: '

        display_text(instructions, cont_name, save_path, draw, img2, confirm_mode)
        cv2.imshow('image', img2)
        key = cv2.waitKey(20)
        if key & 0xFF == 27: # code 27 is for escape key
            exit_mode = 'exit'
            break
        elif key & 0xFF == ord('r'): 
            ctr = repair(ctr)
        elif key & 0xFF == ord('q'): 
            radius += radius_delta
        elif key & 0xFF == ord('a'): 
            if radius > min_radius: radius -= radius_delta
        elif key & 0xFF == ord('z'): 
            dim_win += dim_delta
            cv2.resizeWindow('image', dim_win, dim_win)
        elif key & 0xFF == ord('x'): 
            if dim_win > min_dim: dim_win -= dim_delta
            cv2.resizeWindow('image', dim_win, dim_win)
        elif key & 0xFF == ord('c'): 
            check = not check
        elif key & 0xFF == ord('p'):
            exit_mode = 'previous'
            break
        elif key & 0xFF == ord('n'):
            confirm_mode = True
            display_text(instructions, cont_name, save_path, draw, img2, confirm_mode)
            cv2.imshow('image', img2)
            yn = prompt()
            confirm_mode = False
            if yn == True:
                exit_mode = 'next'
                break
        elif key & 0xFF == ord('s'):
            if draw:    ctr = ctr[:-1]
            saveContour(save_path, ctr, scale, crop)
            exit_mode = 'done'
            break
        elif key & 0xFF == ord('d'): 
            if not draw:  
                # conditions before entering draw mode
                ctr = None
                stPt = (0,0)
                endPt = (0,0)
                open_contour = False
                draw = True
            else:
                # conditions before exiting draw mode
                if ctr is not None:
                    if hover_inserted:
                        ctr = ctr[:-1]
                    hover_inserted = False
                    open_contour = True
                    ctr = dense(ctr)
                    stPt = (0,0)
                    endPt = (0,0)
                    draw = False
                else:
                    instructions[-1] = 'Status: No contour: can not enter edit contour mode.'

    cv2.destroyAllWindows()

    return exit_mode

#------------------------------------------------------------------------------
