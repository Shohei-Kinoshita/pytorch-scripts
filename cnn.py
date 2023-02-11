class CNNDataset(Dataset):
    def __init__(self, df, transform=None, is_train=True):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.is_train = is_train

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        img_path = self.df['path'][idx]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform:
            image_aug = self.transform(image=image)
            image = image_aug['image']
        image_tensor = torch.tensor(image, dtype=torch.float32)
        
        csv_data = np.array(self.df.iloc[idx][LIST_META_COL].values, dtype=np.float32)

        if self.is_train:
            labels = self.df['cancer'][idx]
            label_tensor = torch.tensor(labels, dtype=torch.float32)
            return {'image': image_tensor, 'meta': csv_data, 'target': label_tensor}
        else:
            return {'image': image_tensor, 'meta': csv_data}


def get_transform(*, data):
    assert data == 'train' or data == 'valid', 'Please enter train or valid'
    if data == 'train':
        compose = A.Compose([
            A.Resize(224, 224),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    elif data == 'valid':
        compose = A.Compose([
            A.Resize(224, 224),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    return compose


def data_to_device(dict_data, device):
    dict_data_to_device = {key: dict_data[key].to(device) for key in dict_data.keys()}
    return dict_data_to_device


class ResNet50NetWork(nn.Module):
    def __init__(self, meta_input_size, meta_output_size, output_size):
        super().__init__()
        self.meta_input_size = meta_input_size
        self.meta_output_size = meta_output_size
        self.output_size = output_size
        self.cnn = timm.create_model(model_name='resnet50d', pretrained=True)
        self.meta_vec = nn.Sequential(nn.Linear(self.meta_input_size, self.meta_output_size),
                                                nn.BatchNorm1d(self.meta_output_size),
                                                nn.ReLU(),
                                                nn.Dropout(p=0.1))
        self.classification = nn.Linear((self.cnn.num_classes + self.meta_output_size), self.output_size)
    
    def forward(self, image, meta):
        image = self.cnn(image)
        meta = self.meta_vec(meta)
        feature_concat = torch.cat((image, meta), dim=1)
        out = self.classification(feature_concat)
        return out


def train_fn(train_loader, model, criterion, optimizer, device):
    model.train()
    train_loss = 0.0
    train_metric = 0.0

    for dict_data in tqdm(train_loader):
        optimizer.zero_grad()
        dict_data = data_to_device(dict_data, device)
        outputs = model(dict_data['image'], dict_data['meta'])
        loss = criterion(outputs, dict_data['target'].unsqueeze(dim=1))
        preds = outputs.detach().cpu().numpy()
        labels = dict_data['target'].unsqueeze(dim=1).detach().cpu().numpy()
        train_loss += loss.item()
        train_metric += pfbeta_torch(labels, preds, beta=1)
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

    for dict_data in tqdm(valid_loader):
        dict_data = data_to_device(dict_data, device)
        outputs = model(dict_data['image'], dict_data['meta'])
        loss = criterion(outputs, dict_data['target'].unsqueeze(dim=1))
        preds = outputs.detach().cpu().numpy()
        labels = dict_data['target'].unsqueeze(dim=1).detach().cpu().numpy()
        valid_loss += loss.item()
        valid_metric += pfbeta_torch(labels, preds, beta=1)
        valid_preds.append(preds)

    valid_loss /= len(valid_loader)
    valid_metric /= len(valid_loader)
    valid_preds = np.concatenate(valid_preds)
    return valid_loss, valid_metric, valid_preds


@torch.inference_mode()
def inference_fn(dataloader, model, device):
    model.eval()
    preds = []

    for dict_data in tqdm(dataloader):
        dict_data = data_to_device(dict_data, device)
        outputs = model(dict_data['image'], dict_data['meta'])
        preds.append(outputs.detach().cpu().numpy())

    preds = np.concatenate(preds)
    return preds


sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0)
preds = []
va_idxs = []
for i, (train_idx, valid_idx) in enumerate(sgkf.split(np.zeros(len(df_train)), df_train['cancer'], df_train['patient_id']), start=1):
    model = ResNet50NetWork(META_INPUT_SIZE, META_OUTPUT_SIZE, OUTPUT_SIZE)
    model.to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    print(f'fold: {i}')
    print('='*50)
    train, valid = df_train.loc[train_idx, :], df_train.loc[valid_idx, :]
    train_dataset = RSNADataset(train, get_transform(data='train'), is_train=True)
    valid_dataset = RSNADataset(valid, get_transform(data='valid'), is_train=True)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=2, shuffle=True, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, num_workers=2, shuffle=False, pin_memory=True)

    for epoch in range(EPOCHS):
        print(f'epoch: {epoch}')
        train_loss, train_metric = train_fn(train_loader, model, criterion, optimizer, DEVICE)
        valid_loss, valid_metric, valid_preds = valid_fn(valid_loader, model, criterion, DEVICE)
        print(f'train_loss: {train_loss:.4f}, train_metric: {train_metric:.4f}')
        print(f'valid_loss: {valid_loss:.4f}, valid_metric: {valid_metric:.4f}')
        print('-'*50)
