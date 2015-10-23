
-- Applies a criterion to only last element in input table and last target.
------------------------------------------------------------------------


local SingleTargetSequencerCriterion, parent = torch.class('nn.SingleTargetSequencerCriterion','nn.SequencerCriterion')

function SingleTargetSequencerCriterion:__init(criterion)
   parent.__init(criterion)
   self.criterion = criterion
   if torch.isTypeOf(criterion, 'nn.ModuleCriterion') then
      error("Not Supported")
   end
   self.gradInput = {}
end

function SingleTargetSequencerCriterion:updateOutput(inputTable, targetTable)
   self.output = self.criterion:forward(inputTable[#inputTable], targetTable[#inputTable])
   return self.output
end

function SingleTargetSequencerCriterion:updateGradInput(inputTable, targetTable)
   for i,input in ipairs(inputTable) do
      self.gradInput[i] = nn.rnn.recursiveCopy(self.gradInput[i], self.criterion:backward(input, targetTable[i]))
      -- print (input)
		-- if (i < #inputTable) then
		-- 	-- self.gradInput[i] = input:clone()
		--    -- self.gradInput[i]:fill(0)
		-- else
		-- 	self.gradInput[i] = nn.rnn.recursiveCopy(self.gradInput[i], self.criterion:backward(input, targetTable[i]))
		-- end
   end
   return self.gradInput
end