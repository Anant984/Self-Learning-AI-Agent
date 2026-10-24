
import random
import json

# =====================================
# SELF-LEARNING AI AGENT USING Q-LEARNING
# =====================================

SIZE = 5

START = (0, 0)
GOAL = (4, 4)

OBSTACLES = {
    (1, 1),
    (1, 2),
    (2, 3),
    (3, 3)
}

# Actions: Up, Down, Left, Right
ACTIONS = {
    0: (-1, 0),
    1: (1, 0),
    2: (0, -1),
    3: (0, 1)
}

ACTION_NAMES = ["UP", "DOWN", "LEFT", "RIGHT"]

# Learning settings
LEARNING_RATE = 0.1
DISCOUNT_FACTOR = 0.9
EPSILON = 1.0
MIN_EPSILON = 0.05
EPSILON_DECAY = 0.995
EPISODES = 5000

# Q-table: stores what the agent learns
q_table = {}


def get_q_values(state):
    """Return action values for a state."""
    key = str(state)

    if key not in q_table:
        q_table[key] = [0.0, 0.0, 0.0, 0.0]

    return q_table[key]


def move_agent(state, action):
    """Move the agent and return reward and status."""

    row, col = state
    dr, dc = ACTIONS[action]

    new_state = (row + dr, col + dc)

    # Prevent leaving the grid
    if not (0 <= new_state[0] < SIZE
            and 0 <= new_state[1] < SIZE):
        return state, -5, False

    # Avoid obstacles
    if new_state in OBSTACLES:
        return state, -5, False

    # Reward for reaching the goal
    if new_state == GOAL:
        return new_state, 20, True

    # Small penalty for every ordinary move
    return new_state, -1, False


def choose_action(state, epsilon):
    """Balance exploration and exploitation."""

    if random.random() < epsilon:
        # Exploration: try a random action
        return random.choice(list(ACTIONS.keys()))

    # Exploitation: choose the best-known action
    values = get_q_values(state)
    best_value = max(values)

    best_actions = [
        action
        for action, value in enumerate(values)
        if value == best_value
    ]

    return random.choice(best_actions)


def train_agent():
    """Train the AI using Q-learning."""

    global EPSILON

    print("Training your AI agent...")

    for episode in range(1, EPISODES + 1):
        state = START
        total_reward = 0

        for step in range(100):
            action = choose_action(state, EPSILON)

            next_state, reward, done = move_agent(
                state, action
            )

            current_values = get_q_values(state)
            old_value = current_values[action]

            if done:
                future_value = 0
            else:
                future_value = max(
                    get_q_values(next_state)
                )

            # Q-learning update formula
            new_value = old_value + LEARNING_RATE * (
                reward
                + DISCOUNT_FACTOR * future_value
                - old_value
            )

            current_values[action] = new_value

            state = next_state
            total_reward += reward

            if done:
                break

        # Gradually reduce random exploration
        EPSILON = max(
            MIN_EPSILON,
            EPSILON * EPSILON_DECAY
        )

        if episode % 500 == 0:
            print(
                f"Episode {episode}/{EPISODES}"
                f" | Reward: {total_reward}"
                f" | Exploration: {EPSILON:.3f}"
            )

    print("\nTraining completed!")


def display_grid(state):
    """Display the game board."""

    print()

    for row in range(SIZE):
        line = []

        for col in range(SIZE):
            cell = (row, col)

            if cell == state:
                symbol = " A "
            elif cell == START:
                symbol = " S "
            elif cell == GOAL:
                symbol = " G "
            elif cell in OBSTACLES:
                symbol = " # "
            else:
                symbol = " . "

            line.append(symbol)

        print("".join(line))

    print()


def test_agent():
    """Test the trained agent without random exploration."""

    print("\nTesting the trained AI agent...")

    state = START
    path = [state]
    total_reward = 0

    display_grid(state)

    for step in range(50):
        values = get_q_values(state)

        # Pick the best learned action
        action = max(
            range(len(values)),
            key=lambda a: values[a]
        )

        next_state, reward, done = move_agent(
            state, action
        )

        print("Action:", ACTION_NAMES[action])

        state = next_state
        path.append(state)
        total_reward += reward

        display_grid(state)

        if done:
            print("SUCCESS! The AI reached the goal.")
            break
    else:
        print("The AI did not reach the goal this time.")

    print("Steps taken:", len(path) - 1)
    print("Total reward:", total_reward)


def save_learning():
    """Save learned knowledge to a JSON file."""

    with open("q_table.json", "w") as file:
        json.dump(q_table, file, indent=4)

    print("\nLearned knowledge saved to q_table.json")



if __name__ == "__main__":
    train_agent()
    test_agent()
    save_learning()
