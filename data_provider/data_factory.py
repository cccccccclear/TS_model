from data_provider.data_loader import TimeSeriesDataset
from torch.utils.data import DataLoader

data_dict = {
    'IC' : TimeSeriesDataset,
    'IM' : TimeSeriesDataset,
}

def data_provider(args, flag):
    Data = data_dict[args.data]

    if flag == 'test':
        shuffle_flag = False
        # drop_last = True
        batch_size = 1  # bsz=1 for evaluation
    else:
        shuffle_flag = True
        # drop_last = True
        batch_size = args.batch_size  # bsz for train and valid
    
    data_set = Data(
        root_path=args.root_path,
        data_path=args.data_path,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        target=args.target,
    )
    print(flag, len(data_set))
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
    )
    return data_set, data_loader
