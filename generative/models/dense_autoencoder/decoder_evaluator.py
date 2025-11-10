from dataclasses import dataclass

import pandas as pd
import torch
from sklearn.manifold import TSNE
from typing import Tuple, List

from generative.models.dense_autoencoder.decoder_utils import get_decoder_api
from dense_var_auto_encoder_vin_1 import get_model_performance_on_set, MazeDataLoaderV2, evaluate_k_seq_in_decoder
from ac_simplegrid_levels import get_device, set_seed
import tyro

import matplotlib.pyplot as plt
import seaborn as sns


@dataclass
class Config:
    seed: int = 123
    decoder_model_path: str = "denseAE_generic_2act_seq_VAE_v7_bs=32_epochs=19999-20000_lr=0.0001_end_size_16_var_1_leakyrelu_normalized_by_seq_len_without_aug_instance_norm.pt"
    n_actions_in_seq: int = 10
    torch_deterministic: bool = True
    actor_n_output_channels: int = 16
    n_samples: int = 10000
    mps: bool = False
    calc_tsne_embedding_representation: bool = True


def convert_embedding_to_2d_tsne(emb_tup_list: List[Tuple[list, torch.Tensor]],
                                 action_seq_length_list: List[int]) -> pd.DataFrame:
    embedding_batch = torch.stack([emb_tup[1] for emb_tup in emb_tup_list]).squeeze(1).cpu().numpy()
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    data_2d = tsne.fit_transform(embedding_batch)
    tsne_df = pd.DataFrame(data_2d).rename(columns={0: 'original_dim_1', 1: 'original_dim_2'})
    tsne_df['action_seq_len'] = action_seq_length_list
    tsne_df['action_seq'] = [emb_tup[0] for emb_tup in emb_tup_list]
    return tsne_df


if __name__ == "__main__":
    args = tyro.cli(Config)

    # TRY NOT TO MODIFY: seeding
    set_seed(seed=args.seed, deterministic_torch=args.torch_deterministic)

    device = get_device(args)
    decoder = get_decoder_api(decoder_model_path=args.decoder_model_path, decoder_seq_len=args.n_actions_in_seq,
                              device=device, maze_n_actions=4)
    evaluate_k_seq_in_decoder(decoder=decoder, action_db_path='db/action_sequence_dict_by_sequence_length_entire_db.pkl.gzip', n_samples=args.n_samples,
                              decoder_input_size=args.actor_n_output_channels, compress_prints=True,
                              noise_cat_list=['normal_0', 'normal_-0.9', 'normal_0.9', 'normal_0-3', 'uniform', 'bernoulli', 'exponential', 'laplace'])

    f_names = [
        'action_seq_db_1_to_10_2_act_in_seq_v1.pickle',
    ]

    mzl = MazeDataLoaderV2(f_names_list=f_names,
                           merge_all_into_train=True)  # , batch_size=batch_size, n_actions=n_words - 1, use_dynamic_length=True, seq_len=20)
    dataset_iter, _, _, dataset_seq_len, _, _ = mzl.get_train_val_test_data_eos_version(reserve_two_words=False)

    if args.calc_tsne_embedding_representation:
        _, _, _, embedding_l = get_model_performance_on_set(seq_set=dataset_iter, eval_model=decoder.model, c_mzl=mzl,
                                                            seq_set_len=dataset_seq_len, extract_embedding=True)

        tsne_df = convert_embedding_to_2d_tsne(emb_tup_list=embedding_l, action_seq_length_list=dataset_seq_len)
        # tsne_df.to_csv(
        #     'entire_dataset_tsne_2d_new_normalized_loss_decoder_with_seq_len_trained_on_aug_data_no_aug_data_in_tsne_prod_decoder.csv',
        #     index=False)


    t_loss, t_n_errors, t_n_errors_in_set = get_model_performance_on_set(seq_set=dataset_iter,
                                                                         eval_model=decoder.model, c_mzl=mzl,
                                                                         seq_set_len=dataset_seq_len)
    print(
        f'loss in entire seq dataset set: {t_loss}, # wrong sequences: {t_n_errors}, avg errors in seq: {t_n_errors_in_set}')



    long_seq_dataset_iter = [a for a in dataset_iter if
                             len(decoder.model.trim_action_sequence_from_eos_tokens(a.argmax(-1))[0]) > 5]
    long_seq_dataset_len = [len(decoder.model.trim_action_sequence_from_eos_tokens(a.argmax(-1))[0]) for a in
                            dataset_iter if len(
            decoder.model.trim_action_sequence_from_eos_tokens(a.argmax(-1))[0]) > 5]
    t_loss, t_n_errors, t_n_errors_in_set = get_model_performance_on_set(seq_set=long_seq_dataset_iter,
                                                                         eval_model=decoder.model, c_mzl=mzl,
                                                                         seq_set_len=long_seq_dataset_len)
    print(
        f'loss in long seq dataset set: {t_loss}, # wrong sequences: {t_n_errors}, avg errors in seq: {t_n_errors_in_set}')

    short_seq_dataset_iter = [a for a in dataset_iter if
                              len(decoder.model.trim_action_sequence_from_eos_tokens(a.argmax(-1))[0]) <= 5]
    short_seq_dataset_len = [len(decoder.model.trim_action_sequence_from_eos_tokens(a.argmax(-1))[0]) for a in
                             dataset_iter if len(
            decoder.model.trim_action_sequence_from_eos_tokens(a.argmax(-1))[0]) <= 5]
    t_loss, t_n_errors, t_n_errors_in_set = get_model_performance_on_set(seq_set=short_seq_dataset_iter,
                                                                         eval_model=decoder.model, c_mzl=mzl,
                                                                         seq_set_len=short_seq_dataset_len)
    print(
        f'loss in short seq dataset set: {t_loss}, # wrong sequences: {t_n_errors}, avg errors in seq: {t_n_errors_in_set}')

one_seq_l = [[1, 4, 4, 4, 4, 4, 4, 4, 4, 4], [2, 4, 4, 4, 4, 4, 4, 4, 4, 4], [3, 4, 4, 4, 4, 4, 4, 4, 4, 4],
             [0, 4, 4, 4, 4, 4, 4, 4, 4, 4]]

for sub in one_seq_l:
    re_construct = decoder.model.get_reconstructed_action_list_with_embedding(sub)[0].tolist()
    print(sub, re_construct)

two_seq_l = [[1, 1, 4, 4, 4, 4, 4, 4, 4, 4], [2, 2, 4, 4, 4, 4, 4, 4, 4, 4], [3, 3, 4, 4, 4, 4, 4, 4, 4, 4],
             [0, 0, 4, 4, 4, 4, 4, 4, 4, 4],
             [1, 2, 4, 4, 4, 4, 4, 4, 4, 4], [2, 1, 4, 4, 4, 4, 4, 4, 4, 4], [3, 0, 4, 4, 4, 4, 4, 4, 4, 4],
             [0, 3, 4, 4, 4, 4, 4, 4, 4, 4],
             [1, 3, 4, 4, 4, 4, 4, 4, 4, 4], [3, 1, 4, 4, 4, 4, 4, 4, 4, 4], [0, 2, 4, 4, 4, 4, 4, 4, 4, 4],
             [2, 0, 4, 4, 4, 4, 4, 4, 4, 4]]

for sub in two_seq_l:
    re_construct = decoder.model.get_reconstructed_action_list_with_embedding(sub)[0].tolist()
    print(sub, re_construct)


# # code for action sequence color length representation:
# import matplotlib.pyplot as plt
# import seaborn as sns
# a4_dims = (11.7, 8.27)
# a4_dims = (12, 8)
# fig, ax = plt.subplots(figsize=a4_dims)
#
# tsne_df_v5 = tsne_df.rename(columns={'original_dim_1': 'dimension_1', 'original_dim_2': 'dimension_2'})
#
# # Use nice 10-color palette
# palette = sns.color_palette("flare", n_colors=10)
#
# sns.stripplot(
#     ax=ax,
#     data=tsne_df_v5,
#     x="dimension_1",
#     y="dimension_2",
#     hue="action_seq_len",
#     palette=palette,
#     size=4
# ).set_title(
#     "T-SNE 2D represnetation of entire sequence dataset coloured by sequence length"
# )
#
# # Place legend in bottom right of the plot
# handles, labels = ax.get_legend_handles_labels()
# ax.legend(
#     handles=handles[:10],
#     labels=[str(i) for i in range(1, 11)],
#     title="Action Sequence Length",
#     loc='lower right',
#     bbox_to_anchor=(1, 0)
# )
#
# plt.tight_layout()
# plt.show()

# # code for action sequence embedding representation - run in notebook:
# import plotly.graph_objects as go
#
# fig = go.Figure(
#     data=[
#         go.Scatter(
#             x=tsne_df["original_dim_1"],
#             y=tsne_df["original_dim_2"],
#             mode="markers",
#             name="original",
#         ),
#     ],
#     layout=go.Layout(
#         title=".                                                 T-SNE 2D represnetation of entire sequence dataset embedding",
#         xaxis=dict(title="First Dimension"),
#         yaxis=dict(title="Second Dimension"),
#         legend=dict(title="Encoders"),
#         font=dict(size=18),
#     )
# )
# fig.update_traces(text=tsne_df['action_seq'], mode='markers+text', textposition='top center', textfont=dict(color='rgba(0, 0, 0, 0)'))
# fig.show()