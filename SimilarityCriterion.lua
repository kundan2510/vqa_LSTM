local SimilarityCriterion, Criterion = torch.class('nn.SimilarityCriterion', 'nn.Criterion')

function SimilarityCriterion:__init()
	Criterion.__init(self)
end

function SimilarityCriterion:updateOutput(input, target)
	if 0 > (0.95 - torch.dot(input,target)) then
		self.output = 0
	else
		self.output = 0.95 - torch.dot(input,target)
	end
	return self.output
end

function SimilarityCriterion:updateGradInput(input, target)
	self.output = self:updateOutput(input, target)
	self.gradInput = input:clone()
	if(self.output == 0) then
		self.gradInput = self.gradInput:fill(0)
	else
		self.gradInput = target:mul(-1)
	end
	return self.gradInput
end