import cv2
import mediapipe as mp
import os
import time
import HandTrackingModule as htm

wCam, hCam = 640, 480
cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

# folderPath = "Image Folder"
# myList = os.listdir(folderPath)
# print(myList)
# overlayList = []
# for imPath in myList:
#     image = cv2.imread(f'{folderPath}/{imPath}')
#     # print(f'{folderPath}/{imPath}')
#     overlayList.append(image)

# print(len(overlayList))
pTime = 0;

detector = htm.handDetector(detectionCon=0.75)
tipIds = [4, 8, 12, 16, 20]

while True:
    success, img = cap.read()
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=True)
    #print(lmList)

    if len(lmList) != 0:
        # a
        if lmList[tipIds[0]][2] < lmList[tipIds[0]-1][2] and lmList[tipIds[1]][2] > lmList[tipIds[1]-1][2] and lmList[tipIds[2]][2] > lmList[tipIds[2]-1][2] and lmList[tipIds[3]][2] > lmList[tipIds[3]-1][2] and lmList[tipIds[4]][2] > lmList[tipIds[4]-1][2] and lmList[tipIds[0]][1] >= lmList[tipIds[0]-1][1]:
            detectedChar = 'a'  
            print(detectedChar)
            cv2.putText(img, detectedChar, (45, 375), cv2.FONT_HERSHEY_PLAIN,
                    7, (255, 0, 0), 10)

        # b
        if lmList[tipIds[0]][1] < lmList[tipIds[0]-1][1] and lmList[tipIds[1]][2] < lmList[tipIds[1]-1][2] and lmList[tipIds[2]][2] < lmList[tipIds[2]-1][2] and lmList[tipIds[3]][2] < lmList[tipIds[3]-1][2] and lmList[tipIds[4]][2] < lmList[tipIds[4]-1][2]:
            detectedChar = 'b'
            print(detectedChar)
            cv2.putText(img, detectedChar, (45, 375), cv2.FONT_HERSHEY_PLAIN,
                    7, (255, 0, 0), 10)

        # c
        if lmList[tipIds[0]][1] > lmList[tipIds[0]-1][1] and lmList[tipIds[1]][1] > lmList[tipIds[1]-1][1] and lmList[tipIds[2]][1] > lmList[tipIds[2]-1][1] and lmList[tipIds[3]][1] > lmList[tipIds[3]-1][1] and lmList[tipIds[4]][1] > lmList[tipIds[4]-1][1]:
            detectedChar = 'c'
            print(detectedChar)
            cv2.putText(img, detectedChar, (45, 375), cv2.FONT_HERSHEY_PLAIN,
                    7, (255, 0, 0), 10)

        # d
        if lmList[tipIds[0]][1] > lmList[tipIds[0]-1][1] and lmList[tipIds[1]][2] < lmList[tipIds[1]-1][2] and lmList[tipIds[2]][2] > lmList[tipIds[2]-1][2] and lmList[tipIds[3]][2] > lmList[tipIds[3]-1][2] and lmList[tipIds[4]][2] > lmList[tipIds[4]-1][2]: 
            detectedChar = 'd'
            print(detectedChar)
            cv2.putText(img, detectedChar, (45, 375), cv2.FONT_HERSHEY_PLAIN,
                    7, (255, 0, 0), 10)
        
        # # đ
        # if lmList[tipIds[0]][1] > lmList[tipIds[0]-1][1] and lmList[tipIds[1]][2] < lmList[tipIds[1]-1][2] and lmList[tipIds[2]][2] > lmList[tipIds[2]-1][2] and lmList[tipIds[3]][2] > lmList[tipIds[3]-1][2] and lmList[tipIds[4]][2] > lmList[tipIds[4]-1][2] and lmList[tipIds[1]][1] > lmList[tipIds[1]-1][1] and lmList[tipIds[1]-2][2] < lmList[tipIds[1]-3][2]: 
        #     detectedChar = 'đ'
        #     print(detectedChar)
        #     cv2.putText(img, detectedChar, (45, 375), cv2.FONT_HERSHEY_PLAIN,
        #             7, (255, 0, 0), 10)

        # e
        if lmList[tipIds[0]][1] < lmList[tipIds[0]-1][1] and lmList[tipIds[1]][2] > lmList[tipIds[1]-1][2] and lmList[tipIds[2]][2] > lmList[tipIds[2]-1][2] and lmList[tipIds[3]][2] > lmList[tipIds[3]-1][2] and lmList[tipIds[4]][2] > lmList[tipIds[4]-1][2]:
            detectedChar = 'e'  
            print(detectedChar)
            cv2.putText(img, detectedChar, (45, 375), cv2.FONT_HERSHEY_PLAIN,
                    7, (255, 0, 0), 10)

        # g
        if lmList[tipIds[0]][1] > lmList[tipIds[0]-1][1] and lmList[tipIds[1]][1] > lmList[tipIds[1]-1][1] and lmList[tipIds[2]][1] < lmList[tipIds[2]-1][1] and lmList[tipIds[3]][1] < lmList[tipIds[3]-1][1] and lmList[tipIds[4]][1] < lmList[tipIds[4]-1][1]:
            detectedChar = 'g'  
            print(detectedChar)
            cv2.putText(img, detectedChar, (45, 375), cv2.FONT_HERSHEY_PLAIN,
                    7, (255, 0, 0), 10)
        
        # h
        if lmList[tipIds[0]][1] < lmList[tipIds[0]-1][1] and lmList[tipIds[1]][1] < lmList[tipIds[1]-1][1] and lmList[tipIds[2]][1] > lmList[tipIds[2]-1][1] and lmList[tipIds[3]][2] > lmList[tipIds[3]-1][2] and lmList[tipIds[4]][2] > lmList[tipIds[4]-1][2]:
            detectedChar = 'h'
            print(detectedChar)
            cv2.putText(img, detectedChar, (45, 375), cv2.FONT_HERSHEY_PLAIN,
                    7, (255, 0, 0), 10)

        # i
        if lmList[tipIds[0]][1] < lmList[tipIds[0]-1][1] and lmList[tipIds[1]][2] > lmList[tipIds[1]-1][2] and lmList[tipIds[2]][2] > lmList[tipIds[2]-1][2] and lmList[tipIds[3]][2] > lmList[tipIds[3]-1][2] and lmList[tipIds[4]][2] < lmList[tipIds[4]-1][2]:
            detectedChar = 'i'
            print(detectedChar)
            cv2.putText(img, detectedChar, (45, 375), cv2.FONT_HERSHEY_PLAIN,
                    7, (255, 0, 0), 10)

        # k
        if lmList[tipIds[0]][2] < lmList[tipIds[0]-1][2] and lmList[tipIds[1]][2] < lmList[tipIds[1]-1][2] and lmList[tipIds[2]][1] > lmList[tipIds[2]-1][1] and lmList[tipIds[3]][2] > lmList[tipIds[3]-1][2] and lmList[tipIds[4]][2] > lmList[tipIds[4]-1][2]:
            detectedChar = 'k'
            print(detectedChar)
            cv2.putText(img, detectedChar, (45, 375), cv2.FONT_HERSHEY_PLAIN,
                    7, (255, 0, 0), 10)

        # l
        if lmList[tipIds[0]][1] > lmList[tipIds[0]-1][1] and lmList[tipIds[1]][2] < lmList[tipIds[1]-1][2] and lmList[tipIds[2]][2] > lmList[tipIds[2]-1][2] and lmList[tipIds[3]][2] > lmList[tipIds[3]-1][2] and lmList[tipIds[4]][2] > lmList[tipIds[4]-1][2] and lmList[tipIds[0]][1] >= lmList[tipIds[0]-1][1]:
            detectedChar = 'l'  
            print(detectedChar)
            cv2.putText(img, detectedChar, (45, 375), cv2.FONT_HERSHEY_PLAIN,
                    7, (255, 0, 0), 10)

        # m
        if lmList[tipIds[0]-1][2] > lmList[tipIds[0]-2][2] and lmList[tipIds[1]-2][2] > lmList[tipIds[1]-3][2] and lmList[tipIds[2]-2][2] > lmList[tipIds[2]-3][2] and lmList[tipIds[3]-2][2] > lmList[tipIds[3]-3][2] and lmList[tipIds[4]-2][2] > lmList[tipIds[4]-3][2]:
            detectedChar = 'm'  
            print(detectedChar)
            cv2.putText(img, detectedChar, (45, 375), cv2.FONT_HERSHEY_PLAIN,
                    7, (255, 0, 0), 10)


        else: 
            detectedChar = ' '
        
        

    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime

    cv2.putText(img, f'FPS: {int(fps)}', (400, 70), cv2.FONT_HERSHEY_PLAIN,
                3, (255, 0, 0), 3)

    cv2.imshow("Alphabet Recognition", img)
    cv2.waitKey(1)