import cv2
import mediapipe as mp
import time

# Setting up the camera to fit the entire screen
cap = cv2.VideoCapture(0)
cap.set(3, 1280)  # Full width
cap.set(4, 720)  # Full height

# mediapipe hand detection 
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

# keyboard size, position and layout
class Button:
    def __init__(self, pos, text, size=[50, 50]):
        self.pos = pos
        self.text = text
        self.size = size

keys = [["1","2","3","4","5","6","7","8","9","0"],
        ["Q", "W", "E", "R", "T", "Y", "U", "I", "O", "P", "^", "$"],
        ["A", "S", "D", "F", "G", "H", "J", "K", "L", "%", "*"],
        ["Z", "X", "C", "V", "B", "N", "M", ",", ";", ":", "!", ".", "?"]]

buttonList = []
startY = 350  

button_width = 55  
button_height = 55  
button_spacing = 10  

for i in range(len(keys)):
    for j, key in enumerate(keys[i]):
        x_pos = (button_width + button_spacing) * j + 50
        y_pos = (button_height + button_spacing) * i + startY
        buttonList.append(Button([x_pos, y_pos], key, [button_width, button_height]))

#spacebar and delete buttons to go BELOW the keyboard
spacebar_width = 375
delete_width = 375
buttonList.append(Button([50, startY + 3 * (button_height + button_spacing) + 75], "Space", [spacebar_width, button_height]))
buttonList.append(Button([50 + spacebar_width + button_spacing, startY + 3 * (button_height + button_spacing) + 75], "Delete", [delete_width, button_height]))

finalText = ""
last_click_time = 0
click_delay = 0.8  #time taken for keyboard click to register

# keyboard style/colours
def drawAll(img, buttonList):
    for button in buttonList:
        x, y = button.pos
        w, h = button.size
        # Pink keys
        cv2.rectangle(img, button.pos, (x + w, y + h), (100, 100, 255), cv2.FILLED)
        # purple outline
        cv2.rectangle(img, button.pos, (x + w, y + h), (128, 0, 128), 2)  
        
        #font for letters
        font_scale = 1.5 if w >= 100 else 1  
        cv2.putText(img, button.text, (x + int(w/4), y + int(h/1.5)), cv2.FONT_HERSHEY_PLAIN, font_scale, (50, 50, 50), 2)  # Darker text
    return img

# detecting hand points
def handLandmarks(colorImg):
    if colorImg is None:
        return []
    
    # openCV uses BGR, mp uses RGB. convert colours back n forth here
    imgRGB = cv2.cvtColor(colorImg, cv2.COLOR_BGR2RGB)
    landmarkList = []
    handData = hands.process(imgRGB)
    
    if handData.multi_hand_landmarks:
        for hand in handData.multi_hand_landmarks:
            for id, lm in enumerate(hand.landmark):
                h, w, _ = colorImg.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                landmarkList.append([id, cx, cy])
            mpDraw.draw_landmarks(colorImg, hand, mpHands.HAND_CONNECTIONS)
    return landmarkList

# Main loop
while True:
    success, img = cap.read()
    if not success:
        continue
    
    img = cv2.flip(img, 1)
    
    lmlist = handLandmarks(img)
    img = drawAll(img, buttonList)
    
    if lmlist:
        for button in buttonList:
            x, y = button.pos
            w, h = button.size
            if x < lmlist[8][1] < x + w and y < lmlist[8][2] < y + h:
                #button highlight to know which button user is on
                cv2.rectangle(img, button.pos, (x + w, y + h), (255, 105, 180), cv2.FILLED)
                
                font_scale = 1.5 if w >= 100 else 1
                cv2.putText(img, button.text, (x + int(w/4), y + int(h/1.5)), cv2.FONT_HERSHEY_PLAIN, font_scale, (255, 255, 255), 2)
                
                #when 2 fingerpoints are on one button, the button click is registered
                if lmlist[8][2] < lmlist[7][2] and lmlist[12][2] < lmlist[11][2]:
                    if time.time() - last_click_time > click_delay:
                        if button.text == "Space":
                            finalText += " "
                        elif button.text == "Delete":
                            finalText = finalText[:-1]
                        else:
                            finalText += button.text
                        last_click_time = time.time()
    
    #keyboard drawin w BGR
    cv2.rectangle(img, (50, 250), (810, 320), (100, 100, 255), cv2.FILLED)  
    cv2.rectangle(img, (50, 250), (810, 320), (128, 0, 128), 2)  
    cv2.putText(img, finalText, (60, 300), cv2.FONT_HERSHEY_PLAIN, 3, (50, 50, 50), 3)  
    
    cv2.namedWindow('Virtual Keyboard', cv2.WINDOW_NORMAL)
    cv2.setWindowProperty('Virtual Keyboard', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow("Virtual Keyboard", img)
    
    #exit camera with the escape key
    if cv2.waitKey(1) & 0xFF == 27:  # 27 is the ASCII code for ESC
        break

cap.release()
cv2.destroyAllWindows()