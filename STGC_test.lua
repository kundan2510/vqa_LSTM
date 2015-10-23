require 'nn'
require 'dp'
require 'rnn'

require 'SingleTargetSequencerCriterion'

tm = nn.SingleTargetSequencerCriterion(nn.ClassNLLCriterion())

out = tm:forward({torch.Tensor({-1,-2,-3}),torch.Tensor({-1,-4,-2}),torch.Tensor({-5,-2,-3})},torch.Tensor({1,2,3}))

err = tm:backward({torch.Tensor({-1,-2,-3}),torch.Tensor({-1,-4,-2}),torch.Tensor({-5,-2,-3})},torch.Tensor({1,2,3}))
print (out)
