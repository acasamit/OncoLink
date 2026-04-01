local MACRO = require("MACRO")
local dataset = require("dataset")
local parsing = require("parsing")
local ai = require("ai")

local reload_dataset = false

local args = {...}

function check_for_reload()
	if reload_dataset or not io.open("../data_train.csv", "r") or not io.open("../data_val.csv", "r") then
		dataset.reload()
	end
end

function check_for_training()
	ai.start_train()
end

function main()
	parsing.parse_args(args)

	check_for_reload()
	check_for_training()
end

main()
