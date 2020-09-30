import cv2
import keras
import numpy as np
import tensorflow as tf
from random import choice
from keras.models import load_model
 
# Declare the finite moves allowed.
# The sequence is decided according to the data fed to the model

ALLOWED_MOVES = {
    0: 'paper',
    1: 'rock',
    2: 'scissors',
    3: 'none'
}


def played_move(move):
    '''
        Decide the move based on the given ALLOWED_MOVES
    ''' 
    return ALLOWED_MOVES[move]


def decide_winner(user, comp):
    '''
        Decide the winner from the user and computer moves.
    '''
    if user == comp:
        return "Tie"

    if user == "rock":
        if comp == "scissors":
            return "User"
        if comp == "paper":
            return "Computer"

    if user == "paper":
        if comp == "rock":
            return "User"
        if comp == "scissors":
            return "Computer"

    if user == "scissors":
        if comp == "paper":
            return "User"
        if comp == "rock":
            return "Computer"


model = tf.keras.models.load_model("RockPaperScissorsModel.h5")

cap = cv2.VideoCapture(0)

previous_move = None

while True:
    
    _, frame = cap.read()
    
    # Simulating mirror image
    frame = cv2.flip(frame, 1)
    
    # Drawing the ROI
    # cv2.rectangle(frame, (x1, y1), (x2, y2), (r, g, b), thickness)
    cv2.rectangle(frame, (350, 225), (600, 450), (255, 0, 0), 2) # Rectangle for the user moves
    cv2.rectangle(frame, (50, 225), (300, 450), (255, 0, 0), 2) # Rectangle for the computer moves
    
    # Extracting the ROI
    # frame[y1:y2, x1:x2]
    img = frame[225:450, 350:600]   
    # cv2.imshow("test", img) # testing the ROI

    img = cv2.resize(img, (150, 150))

    # predict the move made
    pred = model.predict(np.array([img]))
    move_code = np.argmax(pred[0]) 
    user_move_name = played_move(move_code)

    # predict the winner (human vs computer)
    if previous_move != user_move_name:
        if user_move_name != "none":
            computer_move_name = choice(['paper', 'rock', 'scissors'])
            winner = decide_winner(user_move_name, computer_move_name)
        else:
            computer_move_name = "none"
            winner = "Waiting..."
    previous_move = user_move_name

    # display the information
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, user_move_name, (375, 210), font, 1, (0, 0, 0), 2, cv2.LINE_AA)  # Predicted user move
    cv2.putText(frame, computer_move_name, (80, 210), font, 1, (255, 255, 255), 2, cv2.LINE_AA)  # Randomly generated comp move
    cv2.putText(frame, "Winner: " + winner, (60, 90), font, 2, (199, 105, 21), 3, cv2.LINE_AA)  # Calculated Winner

    if computer_move_name != "none":
        icon = cv2.imread(
            "images/{}.png".format(computer_move_name))
        icon = cv2.resize(icon, (230, 205))
        frame[235:440, 60:290] = icon

    cv2.imshow("Rock Paper Scissors", frame)

    k = cv2.waitKey(10)
    if k == 27 or k == 32: # Press SpaceBar or Esc key to break the loop and exit the window.
        break

cap.release()
cv2.destroyAllWindows()