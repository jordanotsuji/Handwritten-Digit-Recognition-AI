import tensorflow as tf
import numpy as np

WHITE = 0xFFFFFF
BLACK = 0x000000
MODEL_PATH = "digit_recognition_128_128_10.model"

class Tile:
    def __init__(self, x, y, width, height):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.color = WHITE;


class Canvas:
    def __init__(self, rows, columns, width, height):
        self.rows = rows
        self.cols = columns
        self.len = rows * columns
        self.width = width
        self.height = height
        self.tiles = []
