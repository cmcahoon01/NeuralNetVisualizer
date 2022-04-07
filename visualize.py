import numpy as np
import pygame


class Visualize:
    def __init__(self, screen, window_size, clock, font):
        self.screen = screen
        self.window_size = window_size
        self.clock = clock
        self.font = font

    def visualize(self, net):
        show_net(net, self.screen, self.window_size, self.font)
        self.clock.tick(1)


def start_pygame(window_size, show_number=False):
    pygame.init()
    screen = pygame.display.set_mode(window_size)
    pygame.display.set_caption('Visualize NN')
    clock = pygame.time.Clock()
    pygame.font.init()
    font = pygame.font.SysFont('Comic Sans MS', 30) if show_number else None
    visualize = Visualize(screen, window_size, clock, font)
    return visualize


def show_net(net, screen, window_size, font=None):
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            quit()

    background_color = 196, 196, 196
    screen.fill(background_color)

    layer_width = window_size[0] / (len(net.layers) + 1)
    current_x = layer_width / 2
    previous_layer, previous_colors, previous_biases = get_positions(window_size, net.layers[0].input, current_x, background_color)
    current_x += layer_width
    for i, layer in enumerate(net.layers):
        next_layer, next_colors, next_biases = get_positions(window_size, layer.output, current_x, background_color, layer.biases)
        draw_weights(screen, layer.weights, previous_layer, next_layer, background_color)
        draw_nodes(screen, previous_layer, previous_colors, bias_colors=previous_biases)
        if font is not None:
            biases = None if i == 0 else net.layers[i - 1].biases
            draw_weight_values(screen, previous_layer, next_layer, layer.weights, font)
            draw_node_values(screen, previous_layer, layer.input, biases, font)
        previous_layer, previous_colors, previous_biases = next_layer, next_colors, next_biases
        current_x += layer_width
    draw_nodes(screen, previous_layer, previous_colors, bias_colors=previous_biases)
    if font is not None:
        draw_node_values(screen, previous_layer, net.layers[-1].output, net.layers[-1].biases, font)
    pygame.display.update()


def get_positions(window_size, nodes, x, background_color, biases=None):
    node_height = window_size[1] / len(nodes)
    current_y = node_height / 2
    positions = []
    colors = []
    bias_colors = [] if biases is not None else None

    clip = 2
    if biases is not None:
        biases = np.clip(biases, -clip, clip)

    for i, node in enumerate(nodes):
        color = [255 - int(node * 255)] * 3
        positions.append((int(x), int(current_y)))
        colors.append(color)
        if biases is not None:
            bias = biases[0][i]
            if bias > 0:
                blue = [0, 0, 255]
                bias = bias / clip
                color = np.average([blue, background_color], weights=[bias, 1 - bias], axis=0)
            else:
                red = [255, 0, 0]
                bias = -bias / clip
                color = np.average([red, background_color], weights=[bias, 1 - bias], axis=0)
            bias_colors.append(color)
        current_y += node_height
    return positions, colors, bias_colors


def draw_nodes(screen, positions, colors, bias_colors=None):
    for position, color in zip(positions, colors):
        color = np.clip(color, 0, 255)
        pygame.draw.circle(screen, color, position, 30)
    if bias_colors is not None:
        for positions, color in zip(positions, bias_colors):
            color = np.clip(color, 0, 255)
            pygame.draw.circle(screen, color, positions, 35, 5)


def draw_weights(screen, weights, previous_layer, next_layer, background_color):
    clip = 2
    weights = np.clip(weights, -clip, clip)
    for i in range(len(weights)):
        for j in range(len(weights[i])):
            if weights[i][j] > 0:
                blue = [0, 0, 255]
                weight = weights[i][j] / clip
                color = np.average([blue, background_color], weights=[weight, 1 - weight], axis=0)
            else:
                red = [255, 0, 0]
                weight = -weights[i][j] / clip
                color = np.average([red, background_color], weights=[weight, 1 - weight], axis=0)
            pygame.draw.line(screen, color.astype(int), previous_layer[i], next_layer[j], 5)


def draw_weight_values(screen, previous_layer, next_layer, weights, font):
    for i in range(len(weights)):
        for j in range(len(weights[i])):
            distance = 1 / 3
            x_pos = previous_layer[i][0] * (1 - distance) + next_layer[j][0] * distance
            y_pos = previous_layer[i][1] * (1 - distance) + next_layer[j][1] * distance
            text = font.render(str(round(weights[i][j], 1)), True, (122, 76, 237))
            position = (x_pos - text.get_width() / 2, y_pos + 10)
            screen.blit(text, position)


def draw_node_values(screen, positions, nodes, biases, font):
    if biases is not None:
        while len(biases.shape) > 1:
            biases = biases[0]
    for i in range(len(nodes)):
        text = font.render(str(round(nodes[i], 1)), True, (122, 76, 237))
        position = (positions[i][0] - text.get_width() / 2, positions[i][1] - 20)
        screen.blit(text, position)
        if biases is not None:
            text = font.render(str(round(biases[i], 2)), True, (122, 76, 237))
            position = (positions[i][0] - text.get_width() / 2, positions[i][1] + 30)
            screen.blit(text, position)
