import pygame
import tensorflow as tf
import numpy as np

WHITE = 0xFFFFFF
BLACK = 0x000000
WINDOW_WIDTH = WINDOW_HEIGHT = 600
MODEL_PATH = "digit_recognition_128_128_10.model"

class Tile:
    def __init__(self, x, y, width, height):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.color = WHITE;

    def draw(self, canvas):
        """
        Draw this individual tile as a rectangle using its x, y, width, and height to determine rect bounds
        """
        pygame.draw.rect(canvas, self.color, (self.x, self.y, self.x + self.width, self.y + self.height))

class Canvas:
    def __init__(self, rows, columns, width, height):
        self.rows = rows
        self.cols = columns
        self.len = rows * columns
        self.width = width
        self.height = height
        self.tiles = []

    def draw(self, canvas):
        """
        Call the draw function of every tile within the canvas
        """
        for tileRow in self.tiles:
            for tile in tileRow:
                tile.draw(canvas)

    def getTile(self, clickPosition): 
        """
        Returns the tile that was clicked on by the user based on the click's location on the canvas
        """
        x = clickPosition[0]
        y = clickPosition[1]
        # integer division by tile width and height to get clicked tile's row and col
        col = int(x) // self.tiles[0][0].width
        row = int(y) // self.tiles[0][0].height
        return self.tiles[row][col]
    
    def convert_to_feature(self):
        """
        Converts the current canvas data to an array for the model to use for prediction
        """


