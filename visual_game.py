
import os
import json
import random
import pygame
import agent as ai

# Load the learned Q-table
if os.path.exists("q_table.json"):
    with open("q_table.json", "r") as file:
        ai.q_table = json.load(file)
    print("Trained AI loaded successfully!")
else:
    print("Training AI for the first time...")
    ai.train_agent()
    ai.save_learning()

pygame.init()

# Window settings
CELL = 100
HEADER = 110
SIZE = ai.SIZE
WIDTH = CELL * SIZE
HEIGHT = HEADER + CELL * SIZE + 80

screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Self-Learning AI Game")
clock = pygame.time.Clock()

font = pygame.font.SysFont("arial", 20, bold=True)
small_font = pygame.font.SysFont("arial", 15)

# Colors
BG = (25, 30, 42)
EMPTY = (235, 240, 245)
BLUE = (35, 115, 245)
GREEN = (50, 175, 100)
YELLOW = (250, 195, 55)
DARK = (60, 65, 75)
WHITE = (255, 255, 255)

# Game state
state = ai.START
steps = 0
reward_total = 0
paused = False
finished = False
won = False
last_move = 0
delay = 450


def draw_text(text, x, y, color=WHITE, f=small_font):
    image = f.render(str(text), True, color)
    screen.blit(image, (x, y))


def reset_game():
    global state, steps, reward_total
    global paused, finished, won, last_move

    state = ai.START
    steps = 0
    reward_total = 0
    paused = False
    finished = False
    won = False
    last_move = 0


def move_ai():
    global state, steps, reward_total
    global finished, won

    if finished:
        return

    # Select the highest-valued learned action
    values = ai.get_q_values(state)
    highest = max(values)

    best_actions = [
        action for action, value in enumerate(values)
        if value == highest
    ]

    action = random.choice(best_actions)

    next_state, reward, done = ai.move_agent(
        state, action
    )

    state = next_state
    steps += 1
    reward_total += reward

    print(
        "Step:", steps,
        "| Action:", ai.ACTION_NAMES[action],
        "| Position:", state,
        "| Reward:", reward
    )

    if done:
        finished = True
        won = True
        print("Congratulations! AI reached the goal.")

    elif steps >= 50:
        finished = True
        won = False
        print("Move limit reached. Press R to retry.")


def draw_game():
    screen.fill(BG)

    draw_text(
        "SELF-LEARNING AI",
        15, 12, WHITE, font
    )

    draw_text(
        f"Steps: {steps}    Reward: {reward_total}",
        15, 43
    )

    if finished and won:
        status = "GOAL REACHED!"
        color = YELLOW
    elif finished:
        status = "Run ended - press R"
        color = (255, 100, 100)
    elif paused:
        status = "PAUSED"
        color = YELLOW
    else:
        status = "AI navigating the grid"
        color = WHITE

    draw_text(status, 15, 72, color)

    # Draw grid cells
    for row in range(SIZE):
        for col in range(SIZE):
            cell = (row, col)

            x = col * CELL
            y = HEADER + row * CELL

            color = EMPTY

            if cell == ai.START:
                color = GREEN
            elif cell == ai.GOAL:
                color = YELLOW
            elif cell in ai.OBSTACLES:
                color = DARK

            rect = pygame.Rect(
                x + 2, y + 2, CELL - 4, CELL - 4
            )

            pygame.draw.rect(
                screen, color, rect, border_radius=8
            )

            if cell == ai.START:
                label = "START"
                label_color = WHITE
            elif cell == ai.GOAL:
                label = "GOAL"
                label_color = BG
            elif cell in ai.OBSTACLES:
                label = "BLOCK"
                label_color = WHITE
            else:
                label = ""

            if label:
                image = small_font.render(
                    label, True, label_color
                )
                screen.blit(
                    image, image.get_rect(center=rect.center)
                )

    # Draw the moving AI
    row, col = state
    cx = col * CELL + CELL // 2
    cy = HEADER + row * CELL + CELL // 2

    pygame.draw.circle(screen, BLUE, (cx, cy), 29)
    pygame.draw.circle(screen, WHITE, (cx, cy), 29, 3)

    image = font.render("AI", True, WHITE)
    screen.blit(image, image.get_rect(center=(cx, cy)))

    # Controls
    footer = HEADER + SIZE * CELL + 10
    draw_text("SPACE: Pause/Resume", 10, footer)
    draw_text("N: Next move   R: Restart", 10, footer + 23)
    draw_text("ESC: Quit", 10, footer + 46)

    pygame.display.flip()


# Main game loop
running = True

while running:
    now = pygame.time.get_ticks()

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False

            elif event.key == pygame.K_SPACE:
                paused = not paused

            elif event.key == pygame.K_r:
                reset_game()

            elif event.key == pygame.K_n:
                move_ai()

    # Move automatically every 450 milliseconds
    if (
        not paused
        and not finished
        and now - last_move >= delay
    ):
        move_ai()
        last_move = now

    draw_game()
    clock.tick(60)

pygame.quit()
