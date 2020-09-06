import cv2
import numpy as np

vid = cv2.VideoCapture(0)

img_base = cv2.imread('base2.jpg')
img_base = cv2.resize(img_base, (500, 500))
height_base, width_base, channel_base = img_base.shape

my_vid = cv2.VideoCapture('test.avi')
success, img_video = my_vid.read()      # capturing the first frame
img_video = cv2.resize(img_video, (width_base, height_base))


orb = cv2.ORB_create(nfeatures=1000)
kp1, des1 = orb.detectAndCompute(img_base, None)
#img_base = cv2.drawKeypoints(img_base, kp1, None)


# for the video detection
detection = False
frame_counter = 0


def tackImages(scale,imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range ( 0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape [:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None,scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor= np.hstack(imgArray)
        ver = hor
    return ver


def stackImages(imgArray, scale, lables=[]):
    sizeW= imgArray[0][0].shape[1]
    sizeH = imgArray[0][0].shape[0]
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    if rowsAvailable:
        for x in range(0, rows):
            for y in range(0, cols):
                imgArray[x][y] = cv2.resize(imgArray[x][y], (sizeW,sizeH), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((sizeH, sizeW, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
            hor_con[x] = np.concatenate(imgArray[x])
        ver = np.vstack(hor)
        ver_con = np.concatenate(hor)
    else:
        for x in range(0, rows):
            imgArray[x] = cv2.resize(imgArray[x], (sizeW, sizeH), None, scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor= np.hstack(imgArray)
        hor_con= np.concatenate(imgArray)
        ver = hor
    if len(lables) != 0:
        eachImgWidth= int(ver.shape[1] / cols)
        eachImgHeight = int(ver.shape[0] / rows)
        print(eachImgHeight)
        for d in range(0, rows):
            for c in range (0,cols):
                cv2.rectangle(ver,(c*eachImgWidth,eachImgHeight*d),(c*eachImgWidth+len(lables[d])*13+27,30+eachImgHeight*d),(255,255,255),cv2.FILLED)
                cv2.putText(ver,lables[d],(eachImgWidth*c+10,eachImgHeight*d+20),cv2.FONT_HERSHEY_COMPLEX,0.7,(255,0,255),2)
    return ver

while True:
    suc, img_cam = vid.read()
    img_aug = img_cam.copy()
    img_black = np.zeros((img_cam.shape[0], img_cam.shape[1]), np.uint8)
    kp2, des2 = orb.detectAndCompute(img_cam, None)
    #img_cam = cv2.drawKeypoints(img_cam, kp2, None)


    # for the video part
    if detection == False:
        my_vid.set(cv2.CAP_PROP_POS_FRAMES, 0)  # initializing video to frame 0
        frame_counter = 0
    else:
        if frame_counter == my_vid.get(cv2.CAP_PROP_FRAME_COUNT):  # getting total number of frames
            my_vid.set(cv2.CAP_PROP_POS_FRAMES, 0)
            frame_counter = 0

    successs, img_video = my_vid.read()
    img_video = cv2.resize(img_video, (width_base, height_base))

    # comparing both the descriptors
    brute_force = cv2.BFMatcher()
    matches = brute_force.knnMatch(des1, des2, k=2)
    good_matches = []

    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)
    print(len(good_matches))        # to estimate how well the target is detected


    # drawing the matches to see what kind of matches we are finding
    img_matches = cv2.drawMatches(img_base, kp1, img_cam, kp2, good_matches, None, flags=2)

    # finding the homography matrix
    if len(good_matches) > 20:
        detection = True
        source_points = np.float32([kp1[m.queryIdx].pt
                                    for m in good_matches]).reshape(-1, 1, 2)
        destination_points = np.float32([kp2[m.trainIdx].pt
                                         for m in good_matches]).reshape(-1, 1, 2)

        matrix, mask = cv2.findHomography(source_points, destination_points, cv2.RANSAC, 5)
        print(matrix)

        # Finding the bounding box around the image in webcam

        points = np.float32([[0, 0], [0, height_base], [width_base, height_base],
                            [width_base, 0]]).reshape(-1, 1, 2)

        # finding the points where it thinks it has found the image in webcam
        dest_points = cv2.perspectiveTransform(points, matrix)

        # drawing the points and creating a bounding box
        img_new = cv2.polylines(img_cam, [np.int32(dest_points)], True, (250, 0, 255), 3)

        # warping the img_video based on the matrix
        img_warp = cv2.warpPerspective(img_video, matrix, (img_cam.shape[1], img_cam.shape[0]))

        # creating an appropriate mask
        mask_new = np.zeros((img_cam.shape[0], img_cam.shape[1]), np.uint8)
        cv2.fillPoly(mask_new, [np.int32(dest_points)], (255, 255, 255))   # the bounded part is white
        # but we need the inverse of what we got

        # creating the inverse mask
        mask_inverse = cv2.bitwise_not(mask_new)

        # creating the augmentation and filling the white parts with webcam footages
        img_aug = cv2.bitwise_and(img_aug, img_aug, mask=mask_inverse)
        img_final = cv2.bitwise_or(img_warp, img_aug)
        #img_stacked = stackImages(([img_cam, img_video], [img_warp, my_vid]), .7)



    else:
        img_new = img_cam.copy()
        img_warp = img_black.copy()
        #mask_new = np.zeros((img_cam.shape[0], img_cam.shape[1]), np.uint8)
        #mask_inverse = cv2.bitwise_not(mask_new)
        #img_aug = cv2.bitwise_and(img_aug, img_aug, mask=mask_inverse)
        img_aug = img_black.copy()

        # adding the first image from video with img_aug
        #img_final = cv2.bitwise_or(img_warp, img_aug)
        img_final = img_cam.copy()

    img_stacked = tackImages(1, ([img_cam, img_final]))

    #cv2.imshow('mask', img_final)
    #cv2.imshow('box', img_new)
    #cv2.imshow('warped', img_warp)
    #cv2.imshow('Matches', img_matches)
    #cv2.imshow('webcam', img_cam)
    #cv2.imshow('Base', img_base)
    #cv2.imshow('video', img_video)
    cv2.imshow('Stacked', img_stacked)
    cv2.waitKey(1)
    frame_counter += 1

