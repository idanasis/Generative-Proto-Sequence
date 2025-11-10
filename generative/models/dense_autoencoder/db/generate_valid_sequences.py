import torch


def generate_valid_sequences() -> list[tuple[int, ...]]:
    # mapping = {0: 'up', 1: 'down', 2: 'right', 3: 'left'}
    opposite_actions = {0: 1, 1: 0, 2: 3, 3: 2}
    max_length = 10

    def is_valid_transition(prev: int, current: int) -> bool:
        return current != opposite_actions.get(prev)

    def backtrack(sequence: list[int], prev_action: int, transitions_left: int) -> list[tuple[int, ...]]:
        if len(sequence) == max_length or (sequence and sequence[-1] == 4):
            padded_sequence = sequence + [4] * (max_length - len(sequence))
            return [tuple(padded_sequence)]

        valid_sequences = []

        for action in range(5):  # 0, 1, 2, 3, 4 (end of sequence)
            if action == 4:
                new_sequence = sequence + [4]
                valid_sequences.extend(backtrack(new_sequence, prev_action, transitions_left))
            elif is_valid_transition(prev_action, action):
                new_sequence = sequence + [action]
                if action != prev_action and transitions_left > 0:
                    valid_sequences.extend(backtrack(new_sequence, action, transitions_left - 1))
                elif action == prev_action:
                    valid_sequences.extend(backtrack(new_sequence, action, transitions_left))

        return valid_sequences

    all_sequences = []
    for start_action in range(4):
        all_sequences.extend(backtrack([start_action], start_action, 1))

    return all_sequences


def count_sequences_by_length(sequences: list[tuple[int, ...]]) -> dict[int, int]:
    sequence_counts = {i: 0 for i in range(1, 11)}
    for sequence in sequences:
        # Determine the effective length of the sequence before padding
        effective_length = sequence.index(4) if 4 in sequence else 10
        sequence_counts[effective_length] += 1

    return sequence_counts


def sequence_to_one_hot(sequence: tuple[int, ...], num_actions: int = 5) -> torch.Tensor:
    """
    Convert a single sequence to its one-hot representation.
    num_actions includes the end-of-sequence token (4 in this case).
    """
    one_hot = torch.zeros(len(sequence), num_actions)
    for i, action in enumerate(sequence):
        one_hot[i, action] = 1
    return one_hot

