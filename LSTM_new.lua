require 'nn'
require 'nnx'
require 'nngraph'

local LSTM_new = {}

function LSTM_new.lstm(layer_size,output)
	local x = nn.Identity()()
    local prev_c = nn.Identity()()
    local prev_h = nn.Identity()()

    function new_input_sum()
        -- transforms input
        local i2other = nn.Linear(layer_size, layer_size)(x)
        -- transforms previous timestep's output
        local h2other = nn.Linear(layer_size, layer_size)(prev_h)

        local c2other  = nn.Linear(layer_size, layer_size)(prev_c)
        return nn.CAddTable()({i2other, h2other,c2other})
    end

    local in_gate = nn.Sigmoid()(new_input_sum())
    local f_gate = nn.Sigmoid()(new_input_sum())
    local in_transform = nn.Tanh()(new_input_sum())

    local next_c = nn.CAddTable()({
        nn.CMulTable()({f_gate, prev_c}),
        nn.CMulTable()({in_gate,     in_transform})
    })

    local out_gate = nn.Tanh()(
    	nn.CAddTable()({
	    	nn.Linear(layer_size, layer_size)(x),
	    	 nn.Linear(layer_size, layer_size)(prev_h),
	    	 nn.Tanh()(next_c)
	    	 })
    	 )
    local c_transform = nn.Tanh()(next_c)

    local hidden_gate = nn.CMulTable()({out_gate,c_transform})
    if not output then
        return nn.gModule({x, prev_c, prev_h}, {next_c, hidden_gate})
    else
        return nn.gModule({x, prev_c, prev_h}, {next_c, hidden_gate,out_gate})
    end
end

return LSTM_new