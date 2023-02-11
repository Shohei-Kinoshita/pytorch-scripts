class LSTMDataset:
    def __init__(self, df, pred_range, sequence_rows=10, is_train=True):
        self.df = df.reset_index(drop=True)
        self.pred_range = pred_range
        self.sequence_rows = sequence_rows
        self.pred_range = pred_range
        self.is_train = is_train
        self.x, self.y = self.prepare_data()
    
    def __len__(self):
        return len(self.x)
    
    def prepare_data(self):
        list_x = []
        list_y = []
        for i in range(self.sequence_rows, len(self.df) - self.pred_range):
            x_seq = self.df[i-self.sequence_rows:i][list_x + list_y].values
            y_seq = self.df.loc[i:i + self.pred_range - 1][list_y].values
            list_x.append(x_seq)
            list_y.append(y_seq)
        return list_x, list_y

    def __getitem__(self, idx):
        inputs = torch.tensor(self.x[idx], dtype=torch.float)
        if self.is_train:
            outputs = torch.tensor(self.y[idx], dtype=torch.float)
            return inputs, outputs
        return inputs


class MyLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, batch_size):
        super().__init__()
        self.input_size = input_dim
        self.hidden_layers_size = hidden_dim
        self.batch_size = batch_size
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=0.1)
        self.linear = nn.Linear(hidden_dim, output_dim)
        self.hidden_cell = (torch.zeros(1, self.batch_size, self.hidden_layers_size),
                            torch.zeros(1, self.batch_size, self.hidden_layers_size))
    
    def forward(self, input_seq, hidden0=None):
        batch_size, seq_len = input_seq.shape[0], input_seq.shape[1]
        lstm_out, self.hidden_cell = self.lstm(input_seq, hidden0)
        predictions = self.linear(lstm_out[:, -1, :])
        return predictions


def train_fn(train_loader, model, criterion, optimizer, device):
    model.train()
    train_loss = 0.0
    train_metric = 0.0

    for features, targets in tqdm(train_loader):
        optimizer.zero_grad()
        features = features.to(device)
        targets = targets.to(device)
        outputs = model(features)
        loss = criterion(outputs, targets)
        preds = outputs.detach().cpu().numpy()
        labels = targets.detach().cpu().numpy()

        train_loss += loss.item()
        train_metric += mean_squared_error(labels, preds, squared=False)
        loss.backward()
        optimizer.step()

    train_loss /= len(train_loader)
    train_metric /= len(train_loader)
    return train_loss, train_metric


@torch.no_grad()
def valid_fn(valid_loader, model, criterion, device):
    model.eval()
    valid_loss = 0.0
    valid_metric = 0.0
    valid_preds = []

    for features, targets in tqdm(valid_loader):
        features = features.to(device)
        targets = targets.to(device)
        outputs = model(features)
        loss = criterion(outputs, targets)
        preds = outputs.detach().cpu().numpy()
        labels = targets.detach().cpu().numpy()
        valid_loss += loss.item()
        valid_metric += mean_squared_error(labels, preds, squared=False)
        valid_preds.append(preds)

    valid_loss /= len(valid_loader)
    valid_metric /= len(valid_loader)
    valid_preds = np.concatenate(valid_preds)
    return valid_loss, valid_metric, valid_preds


@torch.inference_mode()
def inference_fn(dataloader, model, device):
    model.eval()
    preds = []

    for features in tqdm(dataloader):
        features = features.to(device)
        outputs = model(features)
        preds.append(outputs.detach().cpu().numpy())

    preds = np.concatenate(preds)
    return preds
