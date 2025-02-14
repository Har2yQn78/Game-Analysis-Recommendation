import pandas as pd
import ast
import random
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler

TOP_TAGS_TO_USE = 5


class GameDataset(Dataset):
    def __init__(self, parquet_file, mode="train"):
        """
        Loads the parquet file, builds vocabularies for developer, publisher, genres, and tags,
        normalizes numeric features, and prepares training samples (anchor, positive, negative)
        based on the 'recommendations' column.
        """
        self.df = pd.read_parquet(parquet_file)
        self.dev_vocab = self.build_vocab(self.df['developers'])
        self.pub_vocab = self.build_vocab(self.df['publishers'])

        self.gen_vocab = self.build_vocab(self.df['genres'])
        self.tag_vocab = self.build_vocab(self.df['tags'])

        scaler = MinMaxScaler()
        self.df[['pct_pos_total', 'num_reviews_total']] = scaler.fit_transform(
            self.df[['pct_pos_total', 'num_reviews_total']]
        )

        self.df = self.df.dropna(subset=['recommendations'])
        self.df.reset_index(drop=True, inplace=True)
        self.mode = mode

    def build_vocab(self, series):
        """
        Builds a simple vocabulary from a pandas series.
        Supports values that are lists, dictionaries, or string representations thereof.
        """
        unique_vals = set()
        for val in series.dropna().unique():
            if isinstance(val, list):
                unique_vals.update(val)
            elif isinstance(val, dict):
                unique_vals.update(val.keys())
            else:
                try:
                    parsed = ast.literal_eval(val)
                    if isinstance(parsed, list):
                        unique_vals.update(parsed)
                    elif isinstance(parsed, dict):
                        unique_vals.update(parsed.keys())
                    else:
                        unique_vals.add(parsed)
                except:
                    unique_vals.add(val)
        vocab = {val: idx for idx, val in enumerate(sorted(unique_vals))}
        return vocab

    def parse_list(self, val):
        """Converts a string representation of a list into an actual list if needed."""
        if isinstance(val, list):
            return val
        try:
            return ast.literal_eval(val)
        except:
            return []

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        """
        Returns a tuple (anchor, positive, negative), where each element is a dictionary of features.
        A positive example is chosen from the 'recommendations' column,
        and a random game (different from the anchor) is used as a negative example.
        """
        row = self.df.iloc[idx]
        anchor = self.process_row(row)

        recs = self.parse_list(row['recommendations'])
        if recs:
            pos_appid = random.choice(recs)
            pos_row = self.df[self.df['AppID'] == pos_appid]
            if pos_row.empty:
                pos_row = row
            else:
                pos_row = pos_row.iloc[0]
        else:
            pos_row = row
        positive = self.process_row(pos_row)

        neg_idx = random.randint(0, len(self.df) - 1)
        while neg_idx == idx:
            neg_idx = random.randint(0, len(self.df) - 1)
        neg_row = self.df.iloc[neg_idx]
        negative = self.process_row(neg_row)

        return anchor, positive, negative

    def process_row(self, row):
        """
        Processes a single row into a feature dictionary:
          - Developer and publisher are single-valued.
          - Genres: a multi-valued feature using the entire genre list.
          - Tags: sort the tag dictionary by vote count and select the top TOP_TAGS_TO_USE.
          - Numeric: [pct_pos_total, num_reviews_total]
        """
        dev = self.dev_vocab.get(row['developers'], 0)
        pub = self.pub_vocab.get(row['publishers'], 0)

        gens_list = self.parse_list(row['genres'])
        gens = [self.gen_vocab.get(g, 0) for g in gens_list if g in self.gen_vocab]
        if not gens:
            gens = [0]

        tag_dict = row['tags']
        if isinstance(tag_dict, str):
            try:
                tag_dict = ast.literal_eval(tag_dict)
            except:
                tag_dict = {}
        elif not isinstance(tag_dict, dict):
            tag_dict = {}
        sorted_tags = sorted(tag_dict.items(), key=lambda x: x[1], reverse=True) if tag_dict else []
        top_tags = [tag for tag, _ in sorted_tags[:TOP_TAGS_TO_USE]]
        if not top_tags:
            top_tags = [0]
        tags = [self.tag_vocab.get(tag, 0) for tag in top_tags]

        numeric = [row['pct_pos_total'], row['num_reviews_total']]

        return {
            'developer': dev,
            'publisher': pub,
            'genres': gens,
            'tags': tags,
            'numeric': numeric
        }


def collate_fn(batch):
    """
    Collate function to pad variable-length multi-valued features.
    The batch is a list of tuples: (anchor, positive, negative)
    """

    def pad_sequences(seq_list, pad_value=0):
        max_len = max(len(seq) for seq in seq_list)
        padded = [seq + [pad_value] * (max_len - len(seq)) for seq in seq_list]
        return torch.tensor(padded, dtype=torch.long)

    anchor_dev = torch.tensor([item[0]['developer'] for item in batch], dtype=torch.long)
    anchor_pub = torch.tensor([item[0]['publisher'] for item in batch], dtype=torch.long)
    anchor_gens = pad_sequences([item[0]['genres'] for item in batch])
    anchor_tags = pad_sequences([item[0]['tags'] for item in batch])
    anchor_num = torch.tensor([item[0]['numeric'] for item in batch], dtype=torch.float)

    positive_dev = torch.tensor([item[1]['developer'] for item in batch], dtype=torch.long)
    positive_pub = torch.tensor([item[1]['publisher'] for item in batch], dtype=torch.long)
    positive_gens = pad_sequences([item[1]['genres'] for item in batch])
    positive_tags = pad_sequences([item[1]['tags'] for item in batch])
    positive_num = torch.tensor([item[1]['numeric'] for item in batch], dtype=torch.float)

    negative_dev = torch.tensor([item[2]['developer'] for item in batch], dtype=torch.long)
    negative_pub = torch.tensor([item[2]['publisher'] for item in batch], dtype=torch.long)
    negative_gens = pad_sequences([item[2]['genres'] for item in batch])
    negative_tags = pad_sequences([item[2]['tags'] for item in batch])
    negative_num = torch.tensor([item[2]['numeric'] for item in batch], dtype=torch.float)

    return {
        'anchor': {
            'developer': anchor_dev,
            'publisher': anchor_pub,
            'genres': anchor_gens,
            'tags': anchor_tags,
            'numeric': anchor_num
        },
        'positive': {
            'developer': positive_dev,
            'publisher': positive_pub,
            'genres': positive_gens,
            'tags': positive_tags,
            'numeric': positive_num
        },
        'negative': {
            'developer': negative_dev,
            'publisher': negative_pub,
            'genres': negative_gens,
            'tags': negative_tags,
            'numeric': negative_num
        }
    }
