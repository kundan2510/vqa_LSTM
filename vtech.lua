vqa = require 'vqa_utils'

vqa.load_MCQ()
vqa.load_answers()
vqa.load_open_questions()


count = 0;

num_to_print_each = 100
for i = 1, #vqa.questions_MCQ do
	qs = vqa.questions_MCQ[i]['question']
	qs = qs:lower()
	io.write(string.format("%s\n",qs))
	-- is_common = string.find(qs,'what is in') or string.find(qs,'what is on')
	-- if is_common then
	-- 	count = count + 1
	-- 	if count <= num_to_print_each then
	-- 		print(qs)
	-- 	end
	-- end
end

for i = 1, #vqa.questions_open do
	qs = vqa.questions_MCQ[i]['question']
	qs = qs:lower()
	is_common = string.find(qs,'what is in') or string.find(qs,'what is on')
	io.write(string.format("%s\n",qs))
	-- if is_common then
	-- 	count = count + 1
	-- 	if num_to_print_each > 0 then
	-- 		print(qs)
	-- 		num_to_print_each = num_to_print_each - 1
	-- 	end
	-- end
end

-- print(count)

-- print(count/(#vqa.questions_MCQ + #vqa.questions_open))