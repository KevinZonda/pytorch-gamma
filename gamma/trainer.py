import torch
from sklearn.model_selection import train_test_split

class Trainer:
    def __init__(self, model, optimizer, loss_fn, after_train=None, device=None):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.after_train = after_train
        self.device = device
        if self.device is None:
            self.device = model.device

    def train(self, train_data, val_data, split_val=False, epochs=10, batch_size=32):
        has_val = True
        if split_val:
            X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
        else:
            X_train, y_train = train_data
            if val_data is None:
                has_val = False
            else:
                X_val, y_val = val_data

        optimizer = self.optimizer

        history = {'train_loss': [], 'val_loss': []}

        for epoch in range(epochs):
            self.model.train()
            running_loss = 0.0

            for i in range(0, len(X_train), batch_size):
                X_batch = X_train[i:i+batch_size]
                y_batch = y_train[i:i+batch_size]

                optimizer.zero_grad()

                y_hat = self.model(X_batch)
                loss = self.criterion(y_hat, y_batch)

                loss.backward()
                optimizer.step()

                running_loss += loss.item() * len(X_batch)

            train_loss = running_loss / len(X_train)
            history['train_loss'].append(train_loss)

            if has_val:
                self.model.eval()
                val_loss = 0.0

                with torch.no_grad():
                    for i in range(0, len(X_val), batch_size):
                        X_batch = X_val[i:i+batch_size]
                        y_batch = y_val[i:i+batch_size]

                        outputs = self.model(X_batch)
                        loss = self.criterion(outputs, y_batch)

                        val_loss += loss.item() * len(X_batch)

                val_loss = val_loss / len(X_val)
                history['val_loss'].append(val_loss)

            print(f'Epoch {epoch+1}/{epochs} - train_loss: {train_loss:.4f} - val_loss: {val_loss:.4f}')

        return history
    
    def evaluate(self, X, y, batch_size=32):
        running_loss = 0.0
        self.model.eval()
        with torch.no_grad():
            for i in range(0, len(X), batch_size):
                X_batch = X[i:i+batch_size]
                y_batch = y[i:i+batch_size]
                loss = self.loss_fn(self.model(X_batch), y_batch)
                running_loss += loss.item() * len(X_batch)
        return running_loss / len(X)
