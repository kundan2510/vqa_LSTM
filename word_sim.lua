require 'nn'

local word_sim = torch.class('nn.word_sim', 'nn.Module')

local w2_vec = require 'w2vutils'

function word_sim:updateOutput(input)
	self.output = w2_vec.distance_all(w2vec,input)
	return self.output
end

function word_sim:updateGradInput(input, gradOutput)
  self.gradInput = w2_vec.multiply_axis(w2vec,gradOutput)
  return self.gradInput
end