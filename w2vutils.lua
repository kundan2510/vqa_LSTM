torch.setdefaulttensortype('torch.FloatTensor')

cmd = torch.CmdLine()
cmd:text('Options')
cmd:option('-binfilename','/home/kundan/word_models/GoogleNews-vectors-negative300.bin','Name of the bin file.')
cmd:option('-outfilename','/home/kundan/word_models/GoogleNews-vectors-negative300.t7','Name of the output t7 file.')
opt = cmd:parse(arg)
w2vutils = {}
if not paths.filep(opt.outfilename) then
	w2vutils = require 'bintot7'
else
	w2vutils = torch.load(opt.outfilename)
	print('Done reading word2vec data.')
end


w2vutils.distance = function (self,vec,k)
	local k = k or 1	
	self.zeros = self.zeros or torch.zeros(self.M:size(1));
	local norm = vec:norm(2)
	vec:div(norm)
	local distances = torch.addmv(self.zeros,self.M ,vec)
	distances , oldindex = torch.sort(distances,1,true)
	local returnwords = {}
	local returndistances = {}
	for i = 1,k do
		table.insert(returnwords, w2vutils.v2wvocab[oldindex[i]])
		table.insert(returndistances, distances[i])
	end
	return returndistances, returnwords
end

w2vutils.distance_all = function (self,vec)
	local k = k or 1	
	self.zeros = self.zeros or torch.zeros(self.M:size(1));
	local distances = torch.addmv(self.zeros,self.M ,vec)
	return distances
end


w2vutils.multiply_axis = function(self,vec)
	local new_vec = vec:clone():resize(1,vec:size()[1])
	local rval = new_vec*self.M
	return rval:resize(rval:size()[2])
end


w2vutils.nearest_words = function(self,word,k)
	return self.distance(self,self.vector(self,word),k)
end

w2vutils.vector = function(self,word)
	local id = self.w2vvocab[word]
	-- assert(id,word .. " not found in dictionary\n")
	return self.M[id]
end

return w2vutils
