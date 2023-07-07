import torch


# dyanmic paddingのために用意
class Collate:
    def __init__(self,table_flag):
        self.table_flag = table_flag

    def __call__(self, batch):
        if self.table_flag:
            ids, cat_data, num_data, mask, length, questions, labels, table_data = zip(*batch)
        else:
            ids, cat_data, num_data, mask, length, questions, labels = zip(*batch)

        max_len = max([len(m) for m in mask])
        # padding
        cat_pad, num_pad, mask_pad = [], [], []
        device = cat_data[0].device
        for cat_array, num_array, mask_array in zip(cat_data, num_data, mask):
            shortage = max_len - cat_array.size(0)
            if shortage > 0:
                cat_array = torch.cat([cat_array, torch.zeros((shortage, cat_array.size(1)), device=device, dtype=torch.long)], dim=0)
                num_array = torch.cat([num_array, torch.zeros((shortage,), device=device, dtype=torch.float)], dim=0)
                mask_array = torch.cat([mask_array, torch.zeros((shortage,), device=device, dtype=torch.float)], dim=0)
            cat_pad.append(cat_array); num_pad.append(num_array); mask_pad.append(mask_array)
                
        cat_data, num_data, mask = torch.stack(cat_pad, dim=0), torch.stack(num_pad, dim=0), torch.stack(mask_pad, dim=0)
        length, questions, labels = torch.stack(length, dim=0), torch.stack(questions, dim=0), torch.stack(labels, dim=0)
        collated_data = [ids, cat_data, num_data, mask, length, questions, labels]

        if self.table_flag:
            table_data = torch.stack(table_data, dim=0)
            collated_data.append(table_data)

        return collated_data