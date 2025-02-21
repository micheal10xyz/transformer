import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


class ZhEnDataset(Dataset):
    def __init__(self, file_path):
        super().__init__()
        self.df = pd.read_csv(file_path)
    
    def __getitem__(self, index):
        return self.df['0'][index], self.df['1'][index]
    

    def __len__(self):
        return 2
        # return len(self.df)
    
def get_zh_en_data_loader():
    file_path = 'dataset/damo_mt_testsets_zh2en_news_wmt18.csv'
    dataset = ZhEnDataset(file_path)
    return DataLoader(dataset, 2, shuffle=True)


if __name__ == '__main__':
    for src, tgt in get_zh_en_data_loader():
        print('src ', src)
        print('tgt', tgt)  