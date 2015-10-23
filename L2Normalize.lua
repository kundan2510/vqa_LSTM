
--[[ 
   This layer expects an [n x d] Tensor and normalizes each
   row to have unit L2 norm.
]]--

-- This file has been taken from Andrej Karpathy Github (https://gist.github.com/karpathy/f3ee599538ff78e1bbe9) and modified

local L2Normalize, parent = torch.class('nn.L2Normalize', 'nn.Module')
function L2Normalize:__init()
   parent.__init(self)
end
function L2Normalize:updateOutput(input)
   self.normSquared = torch.sum(torch.cmul(input, input))
   self.buffer = math.sqrt(self.normSquared)
   self.output = input/self.buffer
   return self.output
end

function L2Normalize:updateGradInput(input, gradOutput)
   self.normSquared = torch.sum(torch.cmul(input, input))
   self.buffer = math.sqrt(self.normSquared)
   self.gradInput = (-torch.cmul(input,input) + 1)/(self.normSquared*self.buffer)
   self.gradInput:cmul(gradOutput) 
   return self.gradInput
end
