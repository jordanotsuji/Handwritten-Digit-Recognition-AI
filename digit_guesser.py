import pygame
import tensorflow as tf
import numpy as np

WHITE = 0xFFFFFF
BLACK = 0x000000
WINDOW_WIDTH = WINDOW_HEIGHT = 600
MODEL_PATH = "digit_recognition_128_128_10.model"
model = tf.keras.models.load_model(MODEL_PATH)

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
        self.columns = columns
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
        try:
            x = clickPosition[0]
            y = clickPosition[1]
            # integer division by tile width and height to get clicked tile's row and col
            col = int(x) // self.tiles[0][0].width
            row = int(y) // self.tiles[0][0].height
            return self.tiles[row][col]
        except:
            pass


    def initTiles(self):
        """
        Initializes the tiles in this canvas and assigns them evenly spaced x and y coordinates 
        """
        tile_width = self.width // self.columns
        tile_height = self.height // self.rows

        for row in range(self.rows):
            self.tiles.append([])
            for column in range(self.columns):
                # initialize each tile with x and y values as tile width and height * row and column for automatic spacing
                self.tiles[row].append(Tile(tile_width * column, tile_height * row, tile_width, tile_height))

    def clear(self):
        """
        Clears the current canvas
        """
        for i in range(self.rows):
            for j in range(self.columns):
                self.tiles[i][j].color = WHITE

    
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


def main():
    """
    Main loop detecting and responding to events 
    """
    while True:
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if(event.key == pygame.K_RETURN):
                    # if enter is pressed, print the model's guess
                    probabilities = model.predict(canvas.convert_to_feature())
                    print(f'Predicted Probabilities: \n\t{probabilities[0]}')
                    prediction = np.argmax(probabilities[0])
                    print(f"Model Prediction: {prediction}")
                elif(event.key == pygame.K_r):
                    canvas.clear()

            if pygame.mouse.get_pressed()[0]:
                # if left click, color the tile black
                pos = pygame.mouse.get_pos()
                clicked = canvas.getTile(pos)
                clicked.color = BLACK;

            if event.type == pygame.QUIT:
                pygame.quit()
                quit(0)

        # redraw the canvas and update
        canvas.draw(window)
        pygame.display.update()

canvas = Canvas(28, 28, WINDOW_WIDTH, WINDOW_HEIGHT)

# Pygame Window creation
pygame.init()
window = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("Handwritten Digit Classification AI")

# main loop
main()

# Exit gracefully 
pygame.quit()
quit()
