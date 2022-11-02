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
        self.initTiles()

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

    def initTiles(self):
        """
        Initializes the tiles in this canvas and assigns them evenly spaced x and y coordinates 
        """
        tile_width = self.width // self.cols
        tile_height = self.height // self.rows

        for row in range(self.rows):
            self.tiles.append([])
            for column in range(self.cols):
                # initialize each tile with x and y values as tile width and height * row and column for automatic spacing
                self.tiles[row].append(Tile(tile_width * column, tile_height * row, tile_width, tile_height))

    
    def convert_to_feature(self):
        """
        Converts the current canvas data to an array for the model to use for prediction
        """
        current_tiles = self.tiles

        feature = [[] for i in range(len(current_tiles))]

        # Build feature matrix one tile at a time based on color
        for i in range(len(current_tiles)):
            for j in range(len(current_tiles[i])):
                if current_tiles[i][j].color == WHITE:
                    feature[i].append(0)
                else:
                    feature[i].append(1)

        # TF requires another surrounding [] for correct dimensions
        # results in (, 28, 28) array
        tf_compatable_feature = []
        tf_compatable_feature.append(feature)
        return tf_compatable_feature


