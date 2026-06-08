from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from .dense_var_auto_encoder_vin_1 import DenseVAE


class ActionGen:

    def __init__(self, pretrained_decoder: DenseVAE, n_act_seq_len: int, device: torch.device, maze_n_actions: int = 4,
                 use_gumble: bool = True, penalize_cyclic_position_revisits: bool = False,
                 deterministic_inference: bool = True):
        self.n_words = maze_n_actions + 1
        self.n_action_seq_length = n_act_seq_len
        self.model = pretrained_decoder
        self.device = device
        self.reverse_action_mapping = {0: 1, 1: 0, 2: 3, 3: 2}
        self.use_gumble = use_gumble
        self.penalize_cyclic_position_revisits = penalize_cyclic_position_revisits
        self.deterministic_inference = deterministic_inference

    @staticmethod
    def temperature_scaled_softmax(logits, temperature=1.0):
        logits = logits / temperature
        return torch.softmax(logits, dim=2)

    def gen_action_seq(self, gen_input, get_actions_as_one_hot: bool = False,
                       exclude_decoder_from_computation_graph: bool = False, deterministic_mode: bool = False):
        """Autoregressively decode an action sequence from the conditioning latent ``z``.

        The decoder is now a causal Transformer (``mingpt_decoder.GPT``) rather than a
        single-shot MLP, so the sequence is produced token-by-token: at each step the
        Transformer is conditioned on ``z`` (the prefix token) plus the tokens generated
        so far, and emits a distribution over the next token. The per-step distribution is
        turned into the same representation the old code returned -- temperature-scaled
        probabilities, a Gumbel-Softmax one-hot, a straight-through one-hot, or a plain
        argmax one-hot -- and the chosen token is fed back in for the next step.

        Gradients still reach ``z`` because the latent prefix conditions *every* step;
        the discrete tokens fed back are detached (we never differentiate through the
        sampling history), mirroring standard straight-through / Gumbel practice.

        Returns:
            ``(None, scaled_probs)`` when ``get_actions_as_one_hot`` is False, else
            ``(one_hot, None)``. Both sequence tensors keep the historical shape
            ``(batch, n_action_seq_length, n_words)``.
        """
        z = gen_input

        def emit_scaled_probs(step_logits: torch.Tensor) -> torch.Tensor:
            # Temperature-scaled softmax over the vocabulary for a single step.
            # temperature_scaled_softmax expects (batch, seq, vocab) and softmaxes dim=2.
            return self.temperature_scaled_softmax(step_logits.unsqueeze(1), temperature=0.01).squeeze(1)

        def emit_argmax_one_hot(step_logits: torch.Tensor) -> torch.Tensor:
            # Deterministic argmax one-hot (no gradient) -- used for inference / when the
            # decoder is excluded from the computation graph.
            indices = torch.argmax(step_logits, dim=-1)
            return F.one_hot(indices, num_classes=self.n_words).float()

        def emit_gumbel_one_hot(step_logits: torch.Tensor) -> torch.Tensor:
            # Differentiable hard one-hot via the Gumbel-Softmax trick.
            return F.gumbel_softmax(step_logits, tau=1.0, hard=True)

        def emit_straight_through_one_hot(step_logits: torch.Tensor) -> torch.Tensor:
            # Argmax one-hot with a straight-through estimator for backprop.
            probs = torch.softmax(step_logits, dim=-1)
            indices = torch.argmax(probs, dim=-1)
            one_hot = F.one_hot(indices, num_classes=self.n_words).float()
            return (one_hot - probs).detach() + probs

        # Select the per-step emission function, preserving the original branching logic.
        if not get_actions_as_one_hot:
            step_emit = emit_scaled_probs
        elif deterministic_mode:  # in_inference_mode
            step_emit = emit_argmax_one_hot if self.deterministic_inference else emit_gumbel_one_hot
        elif exclude_decoder_from_computation_graph:
            step_emit = emit_argmax_one_hot
        elif self.use_gumble:
            step_emit = emit_gumbel_one_hot
        else:
            step_emit = emit_straight_through_one_hot

        # --- Autoregressive generation loop ---
        generated_rows = []  # per-step emitted rows, each (batch, n_words)
        idx = None           # integer tokens fed back to the Transformer (detached)
        for _ in range(self.n_action_seq_length):
            # logits: (batch, t + 1, n_words); the final position predicts the next token.
            logits, _ = self.model.decoder(z, idx)
            step_logits = logits[:, -1, :]

            row = step_emit(step_logits)
            generated_rows.append(row)

            # Condition the next step on the token we just produced (greedy w.r.t. the
            # emitted row, detached so no gradient flows through the discrete choice).
            next_token = torch.argmax(row, dim=-1, keepdim=True).detach()
            idx = next_token if idx is None else torch.cat([idx, next_token], dim=1)

        sequence = torch.stack(generated_rows, dim=1)  # (batch, n_action_seq_length, n_words)

        if not get_actions_as_one_hot:
            return None, sequence
        return sequence, None

    def run_forward_actions(self, current_env, act_list):
        state = None
        done = False
        for act in act_list:
            observation, reward, truncated, _ = current_env.step(act)
            done = truncated
            state = torch.tensor(observation, dtype=torch.float32, device=self.device)
        return current_env, state, done


    def get_reward_per_sequence(self, current_env, act_list, old_state,
                                visited_positions_stats: Optional[dict] = None):
        def get_global_visited_positions(visited_positions_stats):
            """Extract global visited positions."""
            if visited_positions_stats is None:
                return None

            assert "prev_visited_positions" in visited_positions_stats
            assert "global_visited_positions" in visited_positions_stats
            return visited_positions_stats["global_visited_positions"] - visited_positions_stats[
                "prev_visited_positions"]

        def simulate_step(env, act):
            """Simulate the agent's next position based on the action."""
            row, col = env.agent_xy
            dx, dy = env.MOVES[act]
            target_row, target_col = row + dx, col + dy

            # Check if the move is within bounds and free
            is_valid = env.is_in_bounds(target_row, target_col) and env.is_free(target_row, target_col)
            return (target_row, target_col), is_valid

        def is_cycle(target_row, target_col, visited_positions, visited_positions_stats):
            """Check if the target position creates a cycle."""
            if (target_row, target_col) in visited_positions:
                return True  # Revisiting a position in the current sequence

            return False

        global_visited_positions = get_global_visited_positions(visited_positions_stats)
        total_reward = 0
        observation = current_env.envs[0].encode().transpose(2, 0, 1)[np.newaxis, ...] if visited_positions_stats is not None else None  # (1, 3, maze_size, maze_size), prevents None in case of cycle on the first step
        terminations = np.array([False])
        truncations = np.array([False])

        total_infos = {'agent_xy': np.empty((1, ), dtype=object), '_agent_xy': np.empty((0, ), dtype=bool)}
        n_steps = 0
        forward_actions = []
        infos = {'agent_xy': current_env.envs[0].agent_xy} if visited_positions_stats is not None else {}  # Handle the case of cycle on the first step

        observation_l = []
        reward_l = []
        terminations_l = []

        visited_positions = set([old_state])
        current_sequence_revisits_count = 0
        prev_sequence_revisits_count = 0
        global_sequence_revisits_count = 0

        # mapping = {0: 'up', 1: 'down', 2: 'left', 3: 'right'}
        for idx, act in enumerate(act_list.tolist()):
            if visited_positions_stats is not None:
                _env = current_env.envs[0].unwrapped  # Unwrapped environment

                # Simulate the step to get the next position
                (target_row, target_col), is_valid = simulate_step(_env, act)

                # If the move is invalid, don't check for cycles
                if is_valid:
                    # Check for cycles if the move is valid
                    if is_cycle(target_row, target_col, visited_positions, visited_positions_stats):
                        break

            observation, reward, terminations, truncations, infos = current_env.step([act])
            observation_l.append(observation)
            reward_l.append(reward)
            terminations_l.append(terminations)
            n_steps += 1
            total_reward += reward

            state_xy = infos['agent_xy'][0]
            if not state_xy == old_state:
                forward_actions.append(act)
                old_state = state_xy

                # Check if this position has been visited before
                if state_xy in visited_positions:
                    current_sequence_revisits_count += 1

                if visited_positions_stats is not None:
                    if state_xy in visited_positions_stats["prev_visited_positions"]:
                        prev_sequence_revisits_count += 1

                    if state_xy in global_visited_positions:
                        global_sequence_revisits_count += 1

            # Add current position to visited positions
            visited_positions.add(state_xy)

            for key, val in infos.items():
                total_infos[key] = np.concatenate((total_infos.get(key, np.empty((0, ), dtype=bool)), val))

            if terminations or truncations:
                if self.penalize_cyclic_position_revisits:
                    total_reward -= -5 * current_sequence_revisits_count

                if visited_positions_stats is not None:
                    visited_positions_stats.update(
                        {
                            'global_sequence_revisits_count': global_sequence_revisits_count,
                            'prev_sequence_revisits_count': prev_sequence_revisits_count,
                            'current_sequence_revisits_count': current_sequence_revisits_count,
                            'visited_positions': visited_positions,
                        }
                    )


                return observation, total_reward, terminations, truncations, total_infos, n_steps, forward_actions, True, infos, observation_l, reward_l, terminations_l

        if self.penalize_cyclic_position_revisits:
            total_reward -= -5 * current_sequence_revisits_count


        if visited_positions_stats is not None:
            visited_positions_stats.update(
                {
                    'global_sequence_revisits_count': global_sequence_revisits_count,
                    'prev_sequence_revisits_count': prev_sequence_revisits_count,
                    'current_sequence_revisits_count': current_sequence_revisits_count,
                    'visited_positions': visited_positions,
                }
            )

        return observation, total_reward, terminations, truncations, total_infos, n_steps, forward_actions, True, \
               infos, observation_l, reward_l, terminations_l


def get_decoder_api(decoder_model_path: str, decoder_seq_len: int, device: torch.device, maze_n_actions: int = 4,
                    var_for_sample: int = 1, use_gumble_in_decoder: bool = True, penalize_cyclic_position_revisits: bool = False,
                    deterministic_inference: bool = False, load_pretrained_weights: bool = True) -> ActionGen:
    decoder = get_decoder(decoder_f_name=decoder_model_path, decoder_seq_len=decoder_seq_len, device=device,
                          maze_n_actions=maze_n_actions, var_for_sample=var_for_sample, 
                          load_pretrained_weights=load_pretrained_weights)
    return ActionGen(pretrained_decoder=decoder, n_act_seq_len=decoder_seq_len, device=device,
                     maze_n_actions=maze_n_actions, use_gumble=use_gumble_in_decoder,
                     penalize_cyclic_position_revisits=penalize_cyclic_position_revisits,
                     deterministic_inference=deterministic_inference)


def get_decoder(decoder_f_name: str, decoder_seq_len: int, device: torch.device, maze_n_actions: int,
                var_for_sample: int = 1, load_pretrained_weights: bool = True):
    decoder = DenseVAE(input_length=decoder_seq_len, n_words=maze_n_actions + 1, device=device,
                       variance_for_sample=var_for_sample).to(device)
    if load_pretrained_weights:
        # Load state dict directly to the correct device
        decoder.load_state_dict(torch.load(decoder_f_name, map_location=device))
        decoder.to(device)
        decoder.eval()
    else:
        # Random initialization - ensure on correct device and keep in training mode
        decoder.to(device)
        decoder.train()
    return decoder
