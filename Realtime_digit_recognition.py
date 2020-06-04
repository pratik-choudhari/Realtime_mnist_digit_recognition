#code by pratik choudhari
import cv2
import pygame
import numpy as np
from keras.models import load_model

#define colors
white = [255,255,255]
black=[0,0,0]
color = (255, 128, 0)

#flag to keep writing
draw_on = False
#last position of mouse
last_pos = (0, 0)
#radius of tracer
radius = 20
#initial prediction value to display
pred='Draw Something!'

#image size
width = 640
height = 640

#load model
model = load_model('MNIST_digit_recognizer.h5')

# init screen
screen = pygame.display.set_mode((width, height))
screen.fill(white)
#init font
pygame.font.init()
myfont = pygame.font.SysFont('Comic Sans MS', 20)

#get prediction
def predict(fname):
    img = cv2.imread(fname)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.bitwise_not(img)
    img = cv2.resize(img,(28,28))
    img = img.reshape(-1,28,28,1)
    img = img.astype("float")
    img = img/255.0
    return np.argmax(model.predict(img),axis=1)[0]

#get roi
def crop(orginal):
    cropped = pygame.Surface((width, height))
    cropped.blit(orginal, (0, 0), (0, 0, width, height))
    return cropped

#trace mouse cursor
def roundline(srf, color, start, end, radius=1):
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    distance = max(abs(dx), abs(dy))
    for i in range(distance):
        x = int(start[0] + float(i) / distance * dx)
        y = int(start[1] + float(i) / distance * dy)
        pygame.draw.circle(srf, color, (x, y), radius)


try:
    while True:
        # get all events
        e = pygame.event.wait()

        #set predicted text
        textsurface = myfont.render('My guess: '+str(pred), False, (0, 0, 0))
        screen.blit(textsurface,(0,0))

        # clear screen after right click
        if(e.type == pygame.MOUSEBUTTONDOWN and e.button == 3):
            pred='Draw Something!'
            screen.fill(white)

        # quit
        if e.type == pygame.QUIT:
            raise StopIteration

        # start drawing after left click
        if(e.type == pygame.MOUSEBUTTONDOWN and e.button == 1):
            color = black
            pygame.draw.circle(screen, color, e.pos, radius)
            draw_on = True

        #trace mouse
        if e.type == pygame.MOUSEMOTION:
            if draw_on:
                pygame.draw.circle(screen, color, e.pos, radius)
                roundline(screen, color, e.pos, last_pos, radius)
            last_pos = e.pos

        # stop drawing after releasing left click
        if e.type == pygame.MOUSEBUTTONUP and e.button == 1:
            draw_on = False
            fname = "out.png"
            #get roi
            img = crop(screen)
            #save img
            pygame.image.save(img, fname)
            #predict
            pred = str(predict(fname))
            #fill none value and display new pred
            screen.fill(pygame.Color("white"),(0,0,255,40))
            textsurface = myfont.render('My guess: '+str(pred), False, (0, 0, 0))
            screen.blit(textsurface,(0,0))

        pygame.display.flip()

except StopIteration:
    pass

pygame.quit()