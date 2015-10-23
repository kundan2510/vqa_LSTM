lstm = require 'LSTM_lib'

lstm.create_model(3,5,5)

ab = lstm.train_one(torch.Tensor({1,0,0}),{torch.Tensor({0.7,0.73,0}),torch.Tensor({0.4,0.73,0.4}),torch.Tensor({0.5,0.5,0.79})},{torch.Tensor({0.7,0.73,0}),torch.Tensor({0.4,0.73,0.4})})

return ab