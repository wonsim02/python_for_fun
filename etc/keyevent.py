import pygame, keyboard
import time

pygame.init()

screen = pygame.display.set_mode((200, 200))
font = pygame.font.SysFont("consolas", 180)
pygame.display.set_caption("wow")

done, key = False, None

arrow = {'up' : '↑', 'down' : '↓', 'left' : '←', 'right' : '→',
         'w' : 'W', 's' : 'S', 'a' : 'A', 'd' : 'D'}

def printText(msg):
    textSurface = font.render(msg, True, pygame.Color('BLACK'), None)
    textRect = textSurface.get_rect()
    textRect.topleft = (50, 10)

    screen.blit(textSurface, textRect)

while not done:
    keyboard.start_recording()
    time.sleep(0.01)
    e = keyboard.stop_recording()
    key = e[0].name if len(e)>0 and e[0].name in arrow else key
    
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            done = True

    screen.fill((255, 255, 255))
    printText(arrow.get(key, ''))
    pygame.display.flip()

pygame.quit()
