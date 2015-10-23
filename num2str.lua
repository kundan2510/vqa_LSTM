function to_word(str)
	if not str:match('[0-9]') then
		return str
	end
	if str:find('0') == 1 then
		return "zero"
	elseif str:find('1') == 1 then
		return "one"
	elseif str:find('2') == 1 then
		return "two"
	elseif str:find('3') == 1 then
		return "three"
	elseif str:find('4') == 1 then
		return "four"
	elseif str:find('5') == 1 then
		return "five"
	elseif str:find('6') == 1 then
		return "six"
	elseif str:find('7') == 1 then
		return "seven"
	elseif str:find('8') == 1 then
		return "eight"
	elseif str:find('9') == 1 then
		return "nine"
	else
		return "ten"
	end
end