from data_provider.data_loader import TimeSeriesDataset
from data_provider.data_factory import data_provider
from torch.utils.data import DataLoader
data_set = TimeSeriesDataset(root_path='./data/', data_path='temp.csv', target='target')
data_loader = DataLoader(
        data_set,
        batch_size=1,
        shuffle=False,
    )

i = 0
for seq_x, seq_y in data_loader:
    i += 1
    if i > 2:
        break
    print( seq_y.shape )
    print( seq_x.shape )