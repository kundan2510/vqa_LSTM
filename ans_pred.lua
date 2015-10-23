require 'dp'
require 'rnn'

require 'SingleTargetSequencerCriterion'

cmd = torch.CmdLine()
cmd:text()
cmd:option('--learningRate', 0.05, 'learning rate at t=0')
cmd:option('--minLR', 0.00001, 'minimum learning rate')
cmd:option('--saturateEpoch', 400, 'epoch at which linear decayed LR will reach minLR')
cmd:option('--momentum', 0.9, 'momentum')
cmd:option('--maxOutNorm', -1, 'max l2-norm of each layer\'s output neuron weights')
cmd:option('--cutoffNorm', -1, 'max l2-norm of concatenation of all gradParam tensors')
cmd:option('--batchSize', 32, 'number of examples per batch')

cmd:option('--maxEpoch', 1000, 'maximum number of epochs to run')
cmd:option('--maxTries', 50, 'maximum number of epochs to try to find a better local minima for early-stopping')
cmd:option('--progress', false, 'print progress bar')
cmd:option('--silent', false, 'don\'t print anything to stdout')
cmd:option('--uniform', 0.1, 'initialize parameters using uniform distribution between -uniform and uniform. -1 means default initialization')

cmd:option('--uniform', 0.1, 'initialize parameters using uniform distribution between -uniform and uniform. -1 means default initialization')

-- recurrent layer 
cmd:option('--lstm', false, 'use Long Short Term Memory (nn.LSTM instead of nn.Recurrent)')
cmd:option('--rho', 30, 'back-propagate through time (BPTT) for rho time-steps')
cmd:option('--hiddenSize', '{200}', 'number of hidden units used at output of each recurrent layer. When more than one is specified, RNN/LSTMs are stacked')
cmd:option('--zeroFirst', false, 'first step will forward zero through recurrence (i.e. add bias of recurrence). As opposed to learning bias specifically for first step.')
cmd:option('--dropout', false, 'apply dropout after each recurrent layer')
cmd:option('--dropoutProb', 0.5, 'probability of zeroing a neuron (dropout probability)')

-- data
cmd:option('--trainEpochSize', -1, 'number of train examples seen between each epoch')
cmd:option('--validEpochSize', -1, 'number of valid examples used for early stopping and cross-validation') 

cmd:text()

opt = cmd:parse(arg or {})

opt.hiddenSize = dp.returnString(opt.hiddenSize)

if not opt.silent then
   table.print(opt)
end

lm = nn.Sequential()

for i,hiddenSize in ipairs(opt.hiddenSize) do 
	local rnn
	rnn = nn.Sequencer(nn.FastLSTM(inputSize, hiddenSize))
	lm:add(rnn)

	if opt.dropout then -- dropout it applied between recurrent layers
      lm:add(nn.Sequencer(nn.Dropout(opt.dropoutProb)))
   	end

   	inputSize = hiddenSize
end

lm:remember(opt.lstm and 'both' or 'eval')
opt.decayFactor = (opt.minLR - opt.learningRate)/opt.saturateEpoch

train = dp.Optimizer{
   loss = nn.ModuleCriterion(
            nn.SingleTargetSequencerCriterion(nn.ClassNLLCriterion()), 
            nn.Identity(), 
            nn.Identity()
         ),
   epoch_callback = function(model, report) -- called every epoch
      if report.epoch > 0 then
         opt.learningRate = opt.learningRate + opt.decayFactor
         opt.learningRate = math.max(opt.minLR, opt.learningRate)
         if not opt.silent then
            print("learningRate", opt.learningRate)
            if opt.meanNorm then
               print("mean gradParam norm", opt.meanNorm)
            end
         end
      end
   end,
   callback = function(model, report) -- called every batch
      if opt.cutoffNorm > 0 then
         local norm = model:gradParamClip(opt.cutoffNorm) -- affects gradParams
         opt.meanNorm = opt.meanNorm and (opt.meanNorm*0.9 + norm*0.1) or norm
      end
      model:updateGradParameters(opt.momentum) -- affects gradParams
      model:updateParameters(opt.learningRate) -- affects params
      model:maxParamNorm(opt.maxOutNorm) -- affects params
      model:zeroGradParameters() -- affects gradParams 
   end,
   feedback = dp.Perplexity(),  
   sampler = dp.TextSampler{epoch_size = opt.trainEpochSize, batch_size = opt.batchSize}, 
   acc_update = opt.accUpdate,
   progress = opt.progress
}