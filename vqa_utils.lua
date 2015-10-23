local json = require 'cjson'

local vqa_utils = {}

vqa_utils.questions_open = {}

vqa_utils.questions_MCQ = {}

vqa_utils.answers = {}
vqa_utils.q2ans = {}
vqa_utils.id2qsn = {}

function vqa_utils.load_open_questions(filepath)
	filepath = filepath or '/home/kundan/q_data/OpenEnded_mscoco_train2014_questions.json'
	local f = assert(torch.DiskFile(filepath,'r'),'Unable to open file ' .. filepath)
	local txt = f:readString('*a')
	vqa_utils.questions_open = (json.decode(txt))['questions']
	txt = nil
	collectgarbage()
end

function vqa_utils.load_MCQ(filepath)
	filepath= filepath or '/home/kundan/q_data/MultipleChoice_mscoco_train2014_questions.json'
	local f = assert(torch.DiskFile(filepath,'r'),'Unable to open file ' .. filepath)
	local txt = f:readString('*a')
	vqa_utils.questions_MCQ = (json.decode(txt))['questions']
	txt = nil
	collectgarbage()
end


function vqa_utils.load_answers(filepath)
	filepath = filepath or '/home/kundan/ans_data/mscoco_train2014_annotations.json'
	local f = assert(torch.DiskFile(filepath,'r'),'Unable to open file ' .. filepath)
	local txt = assert(f:readString('*a'),'Unable to read the file '..filepath)
	vqa_utils.answers = (json.decode(txt))['annotations']
	txt = nil
	collectgarbage()
end


function vqa_utils.get_question(id)
	return vqa_utils.questions_MCQ[id] or vqa_utils.questions_open[id]
end

function vqa_utils.create_index_for_q_and_a()
	for i,v in ipairs(vqa_utils.answers) do
		vqa_utils.q2ans[v['question_id']] = i
	end
	for i,v in ipairs(vqa_utils.questions_MCQ) do
		vqa_utils.id2qsn[v['question_id']] = i
	end
end

return vqa_utils
