require 'nn'
require 'L2Normalize'
require 'SimilarityCriterion'
require 'optim'
require 'num2str'

torch.manualSeed(0)

local LSTM = require 'LSTM_new'
local model_utils = require 'model_utils'
local w2vec = require 'w2vutils'
-- require 'word_sim'
require 'string_split.lua'
local TM = {}
TM.modules = {}
TM.clones_q = {}
TM.clones_ans = {}
TM.clones_l2 = {}
TM.clones_criterion = {}

TM.word_buffer = {}

TM.net, TM.cnn_activation = dofile('/home/kundan/overfeat-torch-master/run.lua')

TM.vqa = require 'vqa_utils'
TM.rand_ans_indices = {}
TM.done_till = 965+5200+ 4160
TM.num_answers = 0
TM.num_epochs = 0
function TM.create_model(layersize,max_q_len,max_ans_len)
	--TODO
	outputsize = layersize
	--TODO
	TM.layersize = layersize
	TM.outputsize = outputsize

	TM.modules.img_activ_to_embed = nn.Sequential():add(nn.Linear(4096,layersize))

	TM.modules.lstm_q = LSTM.lstm(layersize)
	TM.modules.lstm_ans = LSTM.lstm(layersize,true)
	TM.modules.l2 = nn.Sequential():add(nn.L2Normalize())
	TM.modules.criterion = nn.SimilarityCriterion()

	TM.params, TM.grad_params = model_utils.combine_all_parameters(TM.modules.lstm_q, TM.modules.lstm_ans, TM.modules.l2, TM.img_activ_to_embed)


	--TODO: check this part and try to remove max_q_len
	TM.clones_ans = model_utils.clone_many_times(TM.modules.lstm_ans, max_ans_len+1, not TM.modules.lstm_ans.parameters)
	TM.clones_q = model_utils.clone_many_times(TM.modules.lstm_q, max_q_len+1, not TM.modules.lstm_q.parameters)
	TM.clones_l2 = model_utils.clone_many_times(TM.modules.l2, max_ans_len+1, not TM.modules.l2.parameters)
	TM.clones_criterion = model_utils.clone_many_times(TM.modules.criterion, max_ans_len+1, not TM.modules.criterion.parameters)
end

function TM.train_one(Image_vector,Qsn_table,Answer_table)
	----TODO
	
	----TODO
	print('stop 10')
	local lstm_q_c = {[0]=torch.zeros(TM.layersize)}
	lstm_q_c[0][1] = 1
	local lstm_q_h = {[0]=torch.zeros(TM.layersize)}
	lstm_q_h[0][1] = 1
	local end_of_question = torch.zeros(TM.layersize)
	end_of_question[#end_of_question] = 1
	print('stop 10b')
	local lstm_ans_c = {}
	local lstm_ans_h = {}
	local lstm_ans_out = {}
	print('stop 10c')
	local q_length = #Qsn_table
	local a_length = #Answer_table
	print('stop 10d')
	img_emb = TM.modules.img_activ_to_embed:forward(Image_vector)

	lstm_q_h[1],lstm_q_c[1] = unpack(TM.clones_q[1]:forward({img_emb,lstm_q_h[0],lstm_q_c[0]}))
	print('stop 10e')
	for i = 2,q_length+1 do
		lstm_q_h[i],lstm_q_c[i] = unpack(TM.clones_q[i]:forward({Qsn_table[i-1],lstm_q_h[i-1],lstm_q_c[i-1]}))
	end
	print('stop 10f')
	lstm_ans_h[0] = lstm_q_h[q_length+1]
	lstm_ans_c[0] = lstm_q_c[q_length+1]
	local out_vec = {}
	local loss = {}
	print('stop 10g')
	lstm_ans_c[1], lstm_ans_h[1],lstm_ans_out[1] = unpack(TM.clones_ans[1]:forward({end_of_question,lstm_ans_h[0],lstm_ans_c[0]}))
	
	out_vec[1] = TM.clones_l2[1]:forward(lstm_ans_out[1])

	loss[1] = TM.clones_criterion[1]:forward(out_vec[1], Answer_table[1])
	print('stop 10h')
	for i = 2,a_length do
		lstm_ans_h[i],lstm_ans_c[i], lstm_ans_out[i] = unpack(TM.clones_ans[i]:forward({Answer_table[i-1],lstm_ans_h[i-1],lstm_ans_c[i-1]}))
		out_vec[i] = TM.clones_l2[i]:forward(lstm_ans_out[i])
		loss[i] = TM.clones_criterion[i]:forward(out_vec[i], Answer_table[i])
	end
	print('stop 10i')
	--Backward pass
	local d_ans_h = {[a_length] = torch.zeros(TM.layersize)}     -- d loss / d lstm_ans_h
	local  d_ans_c = {[a_length] = torch.zeros(TM.layersize)}    -- d loss/ d lstm_ans_c
	print('stop 10j')
	local d_q_h = {}     -- d loss / d lstm_q_h
	local  d_q_c = {}    -- d loss/ d lstm_q_c
	local d_ans_out = {}  -- d loss/ d lstm_ans_out
	local d_out_vec = {} 
	for i = a_length,1,-1 do
		d_out_vec[i] = TM.clones_criterion[i]:backward(out_vec[i], Answer_table[i])
		d_ans_out[i] = TM.clones_l2[i]:backward(out_vec[i],d_out_vec[i])

		_,d_ans_h[i-1],d_ans_c[i-1] = unpack(TM.clones_ans[i]:backward({Answer_table[i-1],lstm_ans_h[i-1],lstm_ans_c[i-1]},{d_ans_out[i],d_ans_h[i],d_ans_c[i]}))
	end

	d_q_h[q_length+1] = torch.zeros(TM.layersize)
	d_q_c[q_length+1] = torch.zeros(TM.layersize)
	print('stop 10k')
	print(string.format("question table: %d, lstm_q_h : %d, lstm_q_c : %d",#Qsn_table,#lstm_q_h,#lstm_q_c))
	for i = q_length+1,2,-1 do
		print(i)
		_,d_q_h[i-1],d_q_c[i-1] = unpack(TM.clones_q[i]:backward({Qsn_table[i-1],lstm_q_h[i-1],lstm_q_c[i-1]},{d_q_h[i],d_q_c[i]}))
	end
	print('stop 10l')
	d_Image,_,_ = unpack(TM.clones_q[1]:backward({Image_vector,lstm_q_h[0],lstm_q_c[0]},{d_q_h[1],d_q_c[1]}))
	print('stop 10m')
	TM.modules.img_activ_to_embed:backward(Image_vector,d_Image)
	print('stop 11')
	collectgarbage()
	return torch.Tensor(loss):sum()
end

function TM.train_example(ans_id)
	print('stop 6')
	local ans = {}
	local question = {}
	local image_path = {}
	local ans_vec = {}
	local q_vec = {}
	ans = split(TM.vqa.answers[ans_id]['multiple_choice_answer']:lower(), '[^a-z0-9]')
	question = split(TM.vqa.questions_MCQ[TM.vqa.id2qsn[TM.vqa.answers[ans_id]['question_id']]]['question']:lower(),'[^a-z0-9]')
	image_path = '/home/kundan/train2014/COCO_train2014_' .. string.format('%012.f',TM.vqa.answers[ans_id]['image_id'])..'.jpg'
	ans_vec = {}
	q_vec = {}
	k = 0
	for i = 1,#ans do
		ans[i] = to_word(ans[i])
		ans_vec[i-k] = TM.word_buffer[ans[i]] or w2vec.vector(w2vec,ans[i])
		if not ans_vec[i-k] then
			k = k + 1
		else
			if not TM.word_buffer[ans[i]] then
				TM.word_buffer[ans[i]] = ans_vec[i-k]:clone()
			end
		end
	end

	print(TM.vqa.questions_MCQ[TM.vqa.id2qsn[TM.vqa.answers[ans_id]['question_id']]]['question'])
	print(TM.vqa.answers[ans_id]['multiple_choice_answer'])
	k = 0
	for i = 1,#question do
		question[i] = to_word(question[i])
		q_vec[i-k] = TM.word_buffer[question[i]] or w2vec.vector(w2vec,question[i])
		if not q_vec[i-k] then
			k = k + 1
		else
			if not TM.word_buffer[question[i]] then
				TM.word_buffer[question[i]] = q_vec[i-k]:clone()
			end
		end
	end
	if #ans_vec <= 0 or #q_vec <= 0 or #ans_vec >= 15 or #q_vec >= 30 then
		return TM.train_example(TM.next_ans())
	end
	-- print('stop 9')
	print(string.format('num_word_ans = %d, num_word_ques = %d, image_path = %s',#ans_vec,#q_vec,image_path))
	local Img = TM.cnn_activation(image_path,TM.net,'small')
	local Image_vector = {}
	if not Img then
		return TM.train_example(TM.next_ans())
	else
		Image_vector = Img:resize(4096)
	end
	print("stop 13")
	collectgarbage()
	return Image_vector,q_vec,ans_vec
end

function TM.next_ans()
	-- print('stop 4')
	TM.done_till = TM.done_till + 1
	if TM.done_till > TM.num_answers then
		TM.num_epochs = TM.num_epochs + 1
		TM.done_till = 1
	end
	-- print('stop 5')
	return TM.rand_ans_indices[TM.done_till]
end

function TM.loss(params)
	if TM.params ~= params then
		TM.params = params
	end
	TM.grad_params:zero()
	local loss = TM.train_one(TM.train_example(TM.next_ans()))
	TM.grad_params:clamp(-4, 4)
	return loss, TM.grad_params
end

function TM.get_ready()
	TM.vqa.load_answers()
	TM.vqa.load_MCQ()
	TM.vqa.load_open_questions()
	TM.vqa.create_index_for_q_and_a()
	TM.num_answers = #TM.vqa.answers
	TM.rand_ans_indices = torch.randperm(TM.num_answers)
end

function TM.train(iterations)
	--TODO
	TM.get_ready()
	TM.create_model(300,30,15)
	--TODO
	local optim_state = {learningRate = 1e-1, momentum = 0.8}
	local print_every = 4
	local loss_t = torch.Tensor(print_every)
	loss_t[1] = 0.0
	for i = 1,iterations do
		-- print('stop 1')
		local _, loss_i = optim.sgd(TM.loss, TM.params, optim_state)
		-- print('stop 2')
		loss_t[i%print_every + 1] = loss_i[1]
		if i % print_every == 0 then
			print(string.format('num_epoch = %d, num_iterations = %d, training error = %.4f\n',TM.num_epochs,i,loss_t:sum()/print_every))
		end
		if i % 10000  == 0 then
			torch.save(string.format('models/model_%d_%d_%.4f.modules',TM.num_epochs,i,loss_t:sum()/print_every),TM.modules)
		end
		-- print('stop 3')
	end
end


return TM